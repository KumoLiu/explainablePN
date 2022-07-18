import sys
sys.path.append('..')
from pathlib import Path
from typing import ForwardRef, List, Optional, Sequence, Callable, Union
import torch
import torch.nn as nn
import numpy as np

# from monai.networks.nets import DynUNet
from strix.models.cnn import DynUNet
from strix.models.cnn.blocks.dynunet_block import UnetBasicBlock, UnetUpBlock
from strix.models.cnn.layers.anatomical_gate import AnatomicalAttentionGate as AAG
from monai_ex.networks.layers import Act, Norm, Conv, Pool
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.nets.basic_unet import Down
from torch.nn.modules.activation import ReLU
from blocks.basic_block import TwoConv, UpCat, ResidualUnit, SimpleASPP
from blocks.attention_block import Attention_block
from nets.HESAM import MultiChannelLinear
from nets.utils import set_trainable, save_activation


class DynUNet_cls(DynUNet):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        norm_name: str = "instance",
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        last_activation: Optional[str] = None,
        is_prunable: bool = False,
        filters: Optional[Sequence[int]] = None,
        output_bottleneck: bool = False,
    ) -> None:
        super().__init__(spatial_dims, in_channels, out_channels, kernel_size, strides, 
                        upsample_kernel_size, norm_name, deep_supervision, deep_supr_num,
                        res_block, last_activation, is_prunable, filters, output_bottleneck)
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        self.aag_layers = nn.ModuleList(
                [AAG(spatial_dims, roi_chns, roi_chns) for roi_chns in range(len(self.filters[:-1]))]
            )
        self.downsamples_cls = self.get_downsamples()
        self.avgpool = globalavgpool((1,)*spatial_dims)
        self.fc = nn.Linear(self.filters[-1], out_channels)
        self.apply(self.initialize_weights)
        

    def forward(self, x):
        out = self.input_block(x)
        outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            outputs.append(out)
        code = self.bottleneck(out)

        out = code.clone()
        upsample_outs = []
        for upsample, skip in zip(self.upsamples, reversed(outputs)):
            out = upsample(out, skip)
            upsample_outs.append(out)
        seg_out = self.output_block(out)

        #! classification
        # high-level feature
        he = self.gmp(code)

        downsample_outs = self.aag_layers[0](outputs[0], upsample_outs[-1])
        downsample_outs_cls = [downsample_outs]
        for aag_layers, downsample_cls, upsample_out in zip(self.aag_layers[1:], 
                                                        self.downsamples_cls, reversed(upsample_outs[:-1])):
            downsample_outs = downsample_cls(downsample_outs)
            downsample_outs = aag_layers(downsample_outs, upsample_out)
            downsample_outs_cls.append(downsample_outs)

        cls_out = self.avgpool(downsample_outs)
        cls_out = torch.flatten(cls_out, 1)
        final_feat = torch.add(cls_out, he)
        cls_out = self.fc(final_feat)

        return seg_out, cls_out

    @staticmethod
    def initialize_weights(module):
        name = module.__class__.__name__.lower()
        if "conv3d" in name or "conv2d" in name:
            nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif "norm" in name:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)