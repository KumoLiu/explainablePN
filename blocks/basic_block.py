from sys import implementation
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from .attention_block import Attention_block
from .cbam import CBAM
# from monai.networks.nets.basic_unet import TwoConv

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding
from monai.networks.layers.factories import Conv


class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        use_aspp: bool = False
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()
        conv_0 = Convolution(dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        conv_1 = SimpleASPP(dim, out_chns, out_chns) if use_aspp else Convolution(dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        halves: bool = True,
        use_attention: bool = False,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)
        self.attention_block = Attention_block(
            spatial_dims=dim,
            F_g=up_chns,
            F_l=up_chns,
            F_int=out_chns
        )
        self.use_attention = use_attention

    def forward(self, x: torch.Tensor, x_e: torch.Tensor, masks: torch.Tensor=None):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)
        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
        if self.use_attention:
            _slice = int((masks.shape[1]+1)/2-1)
            x_0 = self.attention_block(x_0, x_e, masks[:,_slice:_slice+1,...])

        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x
    
class ResidualUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    For example:

    .. code-block:: python

        from monai.networks.blocks import ResidualUnit

        convs = ResidualUnit(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="AN",
            act=("prelu", {"init": 0.2}),
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(convs)

    output::

        ResidualUnit(
          (conv): Sequential(
            (unit0): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
            (unit1): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (residual): Identity()
        )

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        adn_ordering: Union[Sequence[str], str] = ["NDA", "ND"],
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        use_cbam: bool = False,
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(dim=dimensions, in_planes=out_channels, use_mask=use_mask) if use_cbam else None
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                dimensions,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering[su],
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[Conv.CONV, dimensions]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        if self.cbam is not None:
            cx = self.cbam(cx, masks)
        out = self.relu(cx + res) # add the residual to the output
        
        return out

class SimpleASPP(nn.Module):
    """
    A simplified version of the atrous spatial pyramid pooling (ASPP) module.

    Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
    https://arxiv.org/abs/1802.02611

    Wang et al., A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions
    from CT Images. https://ieeexplore.ieee.org/document/9109297
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        conv_out_channels: int,
        kernel_sizes: Sequence[int] = (1, 3, 3, 3),
        dilations: Sequence[int] = (1, 2, 4, 6),
        norm_type: Optional[Union[Tuple, str]] = "BATCH",
        acti_type: Optional[Union[Tuple, str]] = "LEAKYRELU",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            conv_out_channels: number of output channels of each atrous conv.
                The final number of output channels is conv_out_channels * len(kernel_sizes).
            kernel_sizes: a sequence of four convolutional kernel sizes.
                Defaults to (1, 3, 3, 3) for four (dilated) convolutions.
            dilations: a sequence of four convolutional dilation parameters.
                Defaults to (1, 2, 4, 6) for four (dilated) convolutions.
            norm_type: final kernel-size-one convolution normalization type.
                Defaults to batch norm.
            acti_type: final kernel-size-one convolution activation type.
                Defaults to leaky ReLU.

        Raises:
            ValueError: When ``kernel_sizes`` length differs from ``dilations``.

        See also:

            :py:class:`monai.networks.layers.Act`
            :py:class:`monai.networks.layers.Conv`
            :py:class:`monai.networks.layers.Norm`

        """
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError(
                "kernel_sizes and dilations length must match, "
                f"got kernel_sizes={len(kernel_sizes)} dilations={len(dilations)}."
            )
        pads = tuple(same_padding(k, d) for k, d in zip(kernel_sizes, dilations))

        self.dim = spatial_dims
        self.mean = Pool["ADAPTIVEAVG", spatial_dims]((1))
        self.img_conv = Convolution(
            dimensions=spatial_dims,
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=1,
            act=acti_type,
            norm=norm_type,
        )

        self.convs = nn.ModuleList()
        for k, d, p in zip(kernel_sizes, dilations, pads):
            _conv = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=conv_out_channels, kernel_size=k, dilation=d, padding=p
            )
            self.convs.append(_conv)

        out_channels = conv_out_channels * (len(pads)+1)  # final conv. output channels
        self.conv_k1 = Convolution(
            dimensions=spatial_dims,
            in_channels=out_channels,
            out_channels=conv_out_channels,
            kernel_size=1,
            act=acti_type,
            norm=norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
        """
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.img_conv(image_features)
        if self.dim == 2:
            image_features = F.upsample(image_features, size=size, mode='bilinear')
        else:
            image_features = F.upsample(image_features, size=size, mode='trilinear')

        x_out = torch.cat([conv(x) for conv in self.convs], dim=1)
        x_out = torch.cat([x_out, image_features], dim=1)
        x_out = self.conv_k1(x_out)
        return x_out