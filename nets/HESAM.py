import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union, Callable

from utils_cw.utils import check_dir
import tqdm
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as mpl_color_map
from monai.networks.nets.basic_unet import Down

# from monai.networks.nets import DynUNet
from strix.models.cnn import DynUNet
from monai_ex.networks.layers import Act, Norm, Conv, Pool
from blocks.basic_block import TwoConv, UpCat, ResidualUnit, SimpleASPP
from nets.utils import save_activation
from ..utils import save_attention

class MultiChannelLinear(nn.Module):
    def __init__(self, in_features: int, n_channels: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.n_channels = n_channels

        self.weights = Parameter(torch.Tensor(self.n_channels, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.n_channels))
        else:
            self.bias = None

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        def apply_along_axis(x1, func1, func2, axis):
            if self.bias is not None:
                return torch.stack([
                    func2(func1(x_i, self.weights[i:i+1].t()), self.bias[i]) for i, x_i in enumerate(torch.unbind(x1, dim=axis))
                ], dim=axis)
            else:
                return torch.stack([
                    func1(x_i, self.weights[i:i+1].t()) for i, x_i in enumerate(torch.unbind(x1, dim=axis))
                ], dim=axis)
        return apply_along_axis(x, torch.mm, torch.add, 1)


class HESAM(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (64, 128, 256, 256),
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.RELU,
        norm=Norm.BATCH,
        dropout=0.0,
        upsample: str = "deconv",
        use_attention: bool = False,
        use_cbam: bool = False,
        use_mask: bool = False,
        save_attentionmap_fpath: str = None,
        save_latent=False,
    ) -> None:
        """
        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            features: 4 integers as numbers of features.
                Defaults to ``(32, 64, 128, 256)``,
                - the first five values correspond to the five-level encoder feature sizes.
            last_feature: number of feature corresponds to the feature size after the last upsampling.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        """
        super().__init__()
        print(f"HESAM features: {features}.")
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dimensions]
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dimensions]

        self.save_latent = save_latent

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, features[0], features[1], act, norm, dropout)
        self.down_2 = Down(dimensions, features[1], features[2], act, norm, dropout)
        self.down_3 = Down(dimensions, features[2], features[3], act, norm, dropout)

        if len(features) == 6:
            self.down_4 = Down(dimensions, features[3], features[4], act, norm, dropout)
            self.down_5 = Down(dimensions, features[4], features[5], act, norm, dropout)
            self.upcat_5 = UpCat(dimensions, features[5], features[4], features[3], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
            self.upcat_4 = UpCat(dimensions, features[3], features[3], features[2], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
            self.upcat_3 = UpCat(dimensions, features[2], features[2], features[1], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        else:
            self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False, use_attention=use_attention)

        self.final_conv = Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.res_block1 = ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
                )
        self.res_block2 = ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
            )

        self.sam = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], out_channels)
        self.save_attentionmap_fpath = save_attentionmap_fpath
        self.features = features

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, label: torch.Tensor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        if len(self.features) == 6:
            x4 = self.down_4(x3)
            x5 = self.down_5(x4)
            u5 = self.upcat_5(x5, x4, masks)
            u4 = self.upcat_4(u5, x3, masks)
        else:
            x4 = u4 = x3
        
        u3 = self.upcat_3(u4, x2, masks)
        u2 = self.upcat_2(u3, x1, masks)
        u1 = self.upcat_1(u2, x0, masks)

        out = self.final_conv(u1)

        res_out = self.res_block1(out, masks)
        res_out = self.res_block2(res_out, masks)
        out = self.sam(res_out)

        # high-level feature
        he = self.gmp(x4)
        hesam = torch.add(out, he.squeeze(dim=-1).squeeze(dim=-1))
        logits = self.final_fc(hesam.squeeze())

        if self.save_latent:
            return [logits, hesam.squeeze()]
        else:
            return logits


class HESAM2(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (64, 128, 256, 256),
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.RELU,
        norm=Norm.BATCH,
        dropout=0.0,
        upsample: str = "deconv",
        mode: str = None,
        save_attentionmap: bool = False,
    ) -> None:
        """
        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            features: 4 integers as numbers of features.
                Defaults to ``(32, 64, 128, 256)``,
                - the first five values correspond to the five-level encoder feature sizes.
            last_feature: number of feature corresponds to the feature size after the last upsampling.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        """
        super().__init__()
        print(f"HESAM features: {features}.")
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dimensions]
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dimensions]

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, features[0], features[1], act, norm, dropout)
        self.down_2 = Down(dimensions, features[1], features[2], act, norm, dropout)
        self.down_3 = Down(dimensions, features[2], features[3], act, norm, dropout)

        self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False, use_attention=False)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False, use_attention=False)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False, use_attention=False)

        self.final_conv = Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.residuals = nn.Sequential(
            ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature+1,
                out_channels=features[-1],
                strides=2,
                act=Act.RELU,
                norm=Norm.BATCH,
                ),
            ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                act=Act.RELU,
                norm=Norm.BATCH,
            )
        )
        self.sam = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], 1)
        self.fc1_ = nn.Linear(features[-1], out_channels)
        self.fc3_ = nn.Linear(256+out_channels, 1)
        self.mode = mode
        self.save_attentionmap = save_attentionmap
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, label: torch.Tensor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        u3 = self.upcat_3(x3, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        out = self.final_conv(u1)
        _slice = int((masks.shape[1]+1)/2-1)
        _out = torch.cat([out, masks[:,_slice:_slice+1,...]], dim=1)

        res_out = self.residuals(_out)
        out = self.sam(res_out)

        # high-level feature
        he = self.gmp(x3)
        hesam = torch.add(out, he.squeeze(dim=-1))
        logits = self.final_fc(hesam.squeeze())

        if self.mode == 'parallel':
            out_feat = self.fc1_(hesam.squeeze())
            if self.save_attentionmap:
                coeffi_MB = self.final_fc.weight
                coeffi_feat = self.fc1_.weight
                save_path_MB = check_dir('/homes/yliu/Data/pn_cls_exp/lidc-82-N/0601_1410-hesam2-slice_5-bs_40-lr_0.001-sgd-mean-parallel/attentionmap-MB')
                save_attention(x, res_out, coeffi_MB, label, logits, (320,320), 0, save_path_MB)
                for i in range(coeffi_feat.shape[0]):
                    save_path_feat = check_dir(f'/homes/yliu/Data/pn_cls_exp/lidc-82-N/0601_1410-hesam2-slice_5-bs_40-lr_0.001-sgd-mean-parallel/attentionmap-feature{i}')
                    save_attention(x, res_out, coeffi_feat, features, out_feat, (320,320), i, save_path_feat)
            return logits, out_feat
        elif self.mode == 'non-parallel':
            out_feat = self.fc1_(hesam.squeeze())
            if len(hesam.squeeze().shape) > 1:
                _hesam = torch.cat([hesam.squeeze(), out_feat], dim=1)
            else:
                _hesam = torch.cat([hesam.squeeze().unsqueeze(0), out_feat.unsqueeze(0)], dim=1)
            out = self.fc3_(_hesam)
            # if self.save_attentionmap:
            #     coeffi_MB = self.fc3_.weight
            #     coeffi_feat = self.fc1_.weight
            #     save_path_MB = check_dir('/homes/yliu/Data/pn_cls_exp/lidc-82/0531_1733-hesam-slice_5-bs_40-lr_0.001-sgd-weight-non-parallel/attentionmap-MB')
            #     save_path_feat = check_dir('/homes/yliu/Data/pn_cls_exp/lidc-82/0531_1733-hesam-slice_5-bs_40-lr_0.001-sgd-weight-non-parallel/attentionmap-feature2')
            #     # save_attention(x, res_out, coeffi_MB, label, out, (320,320), 0, save_path_MB)
            #     save_attention(x, res_out, coeffi_feat, features, out_feat, (320,320), 2, save_path_feat)
            return out, out_feat
        else:
            return logits


# MA multi attention (CABM)
class HESAM_CABM(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        out_channels_f: int,
        features: Sequence[int] = (64, 128, 256, 256),
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.RELU,
        norm=Norm.BATCH,
        dropout=0.0,
        upsample: str = "deconv",
        mode: str = None,
        use_attention: bool = False,
        use_cbam: bool = False,
        use_mask: bool = False,
        save_attentionmap_fpath: Optional[str] = None,
    ) -> None:
        """
        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            features: 4 integers as numbers of features.
                Defaults to ``(32, 64, 128, 256)``,
                - the first five values correspond to the five-level encoder feature sizes.
            last_feature: number of feature corresponds to the feature size after the last upsampling.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        """
        super().__init__()
        print(f"HESAM features: {features}.")
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dimensions]
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dimensions]

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, features[0], features[1], act, norm, dropout)
        self.down_2 = Down(dimensions, features[1], features[2], act, norm, dropout)
        self.down_3 = Down(dimensions, features[2], features[3], act, norm, dropout)

        self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False, use_attention=use_attention)

        self.final_conv = Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.res_block1 = ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
                )
        self.res_block2 = ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
            )

        self.sam = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], 1) if mode != 'multi_slice' else nn.Linear(features[-1], out_channels_f)
        self.fc1_ = nn.Linear(features[-1], out_channels)
        self.mode = mode
        self.save_attentionmap_fpath = save_attentionmap_fpath
        self.att_code = None
        self.res_block2.cbam.SpatialGate.sigmoid.register_forward_hook(
            hook=self.get_att_code()
        ) if use_cbam else None
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_att_code(self):
        def hook(model, input, output):
            self.att_code = output
        return hook

    def forward(self, x: torch.Tensor, label: torch.Tensor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        u3 = self.upcat_3(x3, x2, masks)
        u2 = self.upcat_2(u3, x1, masks)
        u1 = self.upcat_1(u2, x0, masks)
        out = self.final_conv(u1)

        res_out = self.res_block1(out, masks)
        res_out = self.res_block2(res_out, masks)
        out = self.sam(res_out)

        # high-level feature
        he = self.gmp(x3)
        hesam = torch.add(out, he.squeeze(dim=-1))

        if self.mode == 'parallel':
            logits = self.final_fc(hesam.squeeze())
            out_feat = self.fc1_(hesam.squeeze())
            if self.save_attentionmap_fpath is not None:
                coeffi_MB = self.final_fc.weight
                coeffi_feat = self.fc1_.weight
                save_path_MB = check_dir(Path(self.save_attentionmap_fpath)/'attentionmap-MB')
                save_attention(x, res_out, self.att_code, coeffi_MB, label, logits, (320,320), 0, save_path_MB)
                for i in range(coeffi_feat.shape[0]):
                    save_path_feat = check_dir(Path(self.save_attentionmap_fpath)/f'attentionmap-feature{i}')
                    save_activation(x, res_out, coeffi_feat, features, out_feat, (320,320), i, save_path_feat)
            return logits, out_feat
        else:
            logits = self.final_fc(hesam.squeeze())
            return logits

class HESAM_CABM_2head(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        out_channels_f: int,
        features: Sequence[int] = (64, 128, 256, 256),
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.RELU,
        norm=Norm.BATCH,
        dropout=0.0,
        upsample: str = "deconv",
        mode: str = None,
        save_attentionmap_fpath: Optional[str] = None,
        use_attention: bool = False,
        use_cbam: bool = False,
        use_mask: bool = False,
        use_aspp: bool = False,
        save_latent=False,
    ) -> None:
        """
        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            features: 4 integers as numbers of features.
                Defaults to ``(32, 64, 128, 256)``,
                - the first five values correspond to the five-level encoder feature sizes.
            last_feature: number of feature corresponds to the feature size after the last upsampling.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        """
        super().__init__()
        print(f"HESAM features: {features}.")
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dimensions]
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dimensions]

        self.save_latent = save_latent

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout, use_aspp=use_aspp)
        self.down_1 = Down(dimensions, features[0], features[1], act, norm, dropout)
        self.down_2 = Down(dimensions, features[1], features[2], act, norm, dropout)
        self.down_3 = Down(dimensions, features[2], features[3], act, norm, dropout)

        self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False, use_attention=use_attention)

        self.final_conv = SimpleASPP(dimensions, last_feature, last_feature) if use_aspp else Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.res_block1_MB = ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
                )
        self.res_block2_MB = ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
            )

        self.res_block1_feat = ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
                )
        self.res_block2_feat = ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
            )


        self.sam_MB = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.sam_feat = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], out_channels) if mode == 'parallel' else nn.Linear(2* features[-1], out_channels)
        self.fc_feat = nn.Linear(features[-1], out_channels_f)
        self.mode = mode
        self.save_attentionmap_fpath = save_attentionmap_fpath
        self.att_code = None
        self.res_block2_MB.cbam.SpatialGate.sigmoid.register_forward_hook(
            hook=self.get_att_code()
        ) if use_cbam else None
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def get_att_code(self):
        def hook(model, input, output):
            self.att_code = output
        return hook

    def forward(self, x: torch.Tensor, label: torch.Tensor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        u3 = self.upcat_3(x3, x2, masks)
        u2 = self.upcat_2(u3, x1, masks)
        u1 = self.upcat_1(u2, x0, masks)
        out = self.final_conv(u1)


        res_out_MB = self.res_block1_MB(out, masks)
        res_out_MB = self.res_block2_MB(res_out_MB, masks)
        res_out_feat = self.res_block1_feat(out, masks)
        res_out_feat  = self.res_block2_feat(res_out_feat, masks)
        out_MB = self.sam_MB(res_out_MB)
        out_feat = self.sam_feat(res_out_feat)

        # high-level feature
        he = self.gmp(x3)

        if self.mode == 'parallel':
            # hesam_MB = torch.add(out_MB, he.squeeze(dim=-1))
            hesam_MB = torch.add(out_MB, he.squeeze().unsqueeze(-1))
            logits = self.final_fc(hesam_MB.squeeze())
            hesam_feat = torch.add(out_feat, he.squeeze().unsqueeze(-1))
            out_feat = self.fc_feat(hesam_feat.squeeze())
            if self.save_attentionmap_fpath is not None:
                coeffi_MB = self.final_fc.weight
                coeffi_feat = self.fc_feat.weight
                save_path_MB = check_dir(Path(self.save_attentionmap_fpath)/'attentionmap-MB')
                # save_attention(x, res_out_MB, self.att_code, coeffi_MB, label, logits, (320,320), 0, save_path_MB)
                for i in range(coeffi_feat.shape[0]):
                    save_path_feat = check_dir(Path(self.save_attentionmap_fpath)/f'attentionmap-feature{i}')
                    save_activation(x, res_out_feat, coeffi_feat, features, out_feat, (320,320), i, save_path_feat)
                    # save_activation(x, res_out_feat, coeffi_feat, features, out_feat, (320,320), i, save_path_feat)
            if self.save_latent:
                return logits, out_feat, [hesam_MB.squeeze(), hesam_feat.squeeze()], \
                (res_out_MB, self.final_fc.weight, res_out_feat, self.fc_feat.weight)
            else:
                return logits, out_feat, \
                (res_out_MB, self.final_fc.weight, res_out_feat, self.fc_feat.weight)
        elif self.mode == 'non-parallel':
            hesam_feat = torch.add(out_feat, he.squeeze(dim=-1))
            final_out_feat = self.fc_feat(hesam_feat.squeeze())
            _hesam_MB = torch.add(out_MB, he.squeeze(dim=-1))
            hesam_MB = torch.cat([_hesam_MB.squeeze(), out_feat.squeeze()], dim=1)
            logits = self.final_fc(hesam_MB.squeeze())
            if self.save_attentionmap_fpath is not None:
                coeffi_MB = self.final_fc.weight
                coeffi_feat = self.fc_feat.weight
                save_path_MB = check_dir(Path(self.save_attentionmap_fpath)/'attentionmap-MB')
                save_attention(x, res_out_MB, self.att_code, coeffi_MB, label, logits, (320,320), 0, save_path_MB)
                for i in range(coeffi_feat.shape[0]):
                    save_path_feat = check_dir(Path(self.save_attentionmap_fpath)/f'attentionmap-feature{i}')
                    save_activation(x, res_out_feat, coeffi_feat, features, final_out_feat, (320,320), i, save_path_feat)
            return logits, final_out_feat, \
                (res_out_MB, self.final_fc.weight, res_out_feat, self.fc_feat.weight)
        elif self.mode == 'multi-direc':
            raise NotImplementedError
            return he, res_out_MB, res_out_feat, self.att_code
        else:
            raise NotImplementedError
            hesam_MB = torch.add(out_MB, he.squeeze(dim=-1))
            logits = self.final_fc(hesam_MB.squeeze())
            return logits
