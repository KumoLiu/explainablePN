import sys
from pathlib import Path
from typing import List, Optional, Sequence, Callable
from utils_cw.utils import check_dir
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets.basic_unet import Down

# from monai.networks.nets import DynUNet
from medlp.models.cnn import DynUNet
from monai_ex.networks.layers import Act, Norm, Conv, Pool
from medlp.models.cnn.layers.anatomical_gate import AnatomicalAttentionGate as AAG
from monai.networks.blocks.dynunet_block import get_conv_layer
from torch.nn.modules.activation import ReLU
from blocks.basic_block import TwoConv, UpCat, ResidualUnit, SimpleASPP
from blocks.attention_block import Attention_block
from nets.HESAM import MultiChannelLinear
from nets.utils import set_trainable, save_activation


class hesam_aag(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        roi_classes: int,
        out_channels: int,
        out_channels_f: int,
        features: Sequence[int] = (64, 128, 256, 256),
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.RELU,
        norm=Norm.BATCH,
        dropout=0.0,
        upsample: str = "deconv",
        save_attentionmap_fpath: Optional[str] = None,
        use_attention: bool = False,
        use_cbam: bool = False,
        use_mask: bool = False,
        use_aspp: bool = False,
        freeze_backbone: bool = False
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
        self.down_1 = TwoConv(dimensions, features[0], features[1], act, norm, dropout)
        self.down_2 = TwoConv(dimensions, features[1], features[2], act, norm, dropout)
        self.down_3 = TwoConv(dimensions, features[2], features[3], act, norm, dropout)

        self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False, use_attention=use_attention)

        self.final_conv = SimpleASPP(dimensions, last_feature, last_feature) if use_aspp else Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.maxpool = Pool["MAX", dimensions](kernel_size=2)
        self.avgpool = globalavgpool((1,)*dimensions)
        
        roi_chns = [roi_classes] + list(features[:-1])
        self.roi_convs = nn.ModuleList(
            [
                get_conv_layer(
                    dimensions,
                    chn,
                    roi_chns[i+1],
                    kernel_size=3,
                    stride=1,
                    act='relu',
                    norm='batch',
                    bias=True,
                    conv_only=False
                ) for i, chn in enumerate(roi_chns[:-1])
            ]
        )
        self.aag_layers = nn.ModuleList(
            [AAG(dimensions, chn, chn) for chn in roi_chns[1:]]
            # [AAG(dimensions, chn, chn, mode='cat') for chn in roi_chns[1:]]
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


        self.sam_feat = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], out_channels)
        self.fc_feat = nn.Linear(features[-1], out_channels_f)
        self.save_attentionmap_fpath = save_attentionmap_fpath
        self.att_code = None

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # if freeze_backbone:
        #     set_trainable(self, True)
            # set_trainable(self.final_fc, False)
        
    def get_att_code(self):
        def hook(model, input, output):
            self.att_code = output
        return hook

    def forward(self, x: torch.Tensor, label: torch.Tensor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        img, roi = x, masks
        x0 = self.conv_0(img)
        y = self.roi_convs[0](roi)
        
        x1 = self.maxpool(x0)
        y = self.maxpool(y)
        x1 = self.aag_layers[0](x1, y)
        x1 = self.down_1(x1)

        y = self.roi_convs[1](y)
        y = self.maxpool(y)
        x2 = self.maxpool(x1)
        x2 = self.aag_layers[1](x2, y)
        x2 = self.down_2(x2)

        y = self.roi_convs[2](y)
        y = self.maxpool(y)
        x3 = self.maxpool(x2)
        x3 = self.aag_layers[2](x3, y)
        x3 = self.down_3(x3)

        u3 = self.upcat_3(x3, x2, masks)
        u2 = self.upcat_2(u3, x1, masks)
        u1 = self.upcat_1(u2, x0, masks)
        out = self.final_conv(u1)

        res_out_feat = self.res_block1_feat(out, masks)
        res_out_feat  = self.res_block2_feat(res_out_feat, masks)
        out_feat = self.sam_feat(res_out_feat)

        # high-level feature
        he = self.gmp(x3)

        hesam_feat = torch.add(out_feat, he.squeeze().unsqueeze(-1))
        out_feat = self.fc_feat(hesam_feat.squeeze())
        
        x_MB = self.avgpool(x3)
        x_MB_flatten = torch.flatten(x_MB, 1)
        logits = self.final_fc(x_MB_flatten)

        if self.save_attentionmap_fpath is not None:
            coeffi_feat = self.fc_feat.weight
            for i in range(coeffi_feat.shape[0]):
                save_path_feat = check_dir(Path(self.save_attentionmap_fpath)/f'attentionmap-feature{i}')
                save_activation(x, res_out_feat, coeffi_feat, features, out_feat, (320,320), i, save_path_feat)

        return logits, out_feat

class hesam_woaag(nn.Module):
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
        save_attentionmap_fpath: Optional[str] = None,
        use_attention: bool = False,
        use_cbam: bool = False,
        use_mask: bool = False,
        use_aspp: bool = False,
        freeze_backbone: bool = False
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

        self.final_conv = SimpleASPP(dimensions, last_feature, last_feature) if use_aspp else Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.maxpool = Pool["MAX", dimensions](kernel_size=2)
        self.avgpool = globalavgpool((1,)*dimensions)
        
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


        self.sam_feat = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], out_channels)
        self.fc_feat = nn.Linear(features[-1], out_channels_f)
        self.save_attentionmap_fpath = save_attentionmap_fpath
        self.att_code = None


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # if freeze_backbone:
        #     set_trainable(self, True)
            # set_trainable(self.final_fc, False)
        
    def get_att_code(self):
        def hook(model, input, output):
            self.att_code = output
        return hook

    def forward(self, x: torch.Tensor, label: torch.Tensor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        img, roi = x, masks
        x0 = self.conv_0(img)
        
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        u3 = self.upcat_3(x3, x2, roi)
        u2 = self.upcat_2(u3, x1, roi)
        u1 = self.upcat_1(u2, x0, roi)
        out = self.final_conv(u1)


        res_out_feat = self.res_block1_feat(out, roi)
        res_out_feat  = self.res_block2_feat(res_out_feat, roi)
        out_feat = self.sam_feat(res_out_feat)

        # high-level feature
        he = self.gmp(x3)

        hesam_feat = torch.add(out_feat, he.squeeze().unsqueeze(-1))
        out_feat = self.fc_feat(hesam_feat.squeeze())
        
        x_MB = self.avgpool(x3)
        x_MB_flatten = torch.flatten(x_MB, 1)
        logits = self.final_fc(x_MB_flatten)

        if self.save_attentionmap_fpath is not None:
            coeffi_feat = self.fc_feat.weight
            for i in range(coeffi_feat.shape[0]):
                save_path_feat = check_dir(Path(self.save_attentionmap_fpath)/f'attentionmap-feature{i}')
                save_activation(x, res_out_feat, coeffi_feat, features, out_feat, (320,320), i, save_path_feat)

        return logits, out_feat


class raw_hesam_aag(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        roi_classes: int,
        out_channels: int,
        out_channels_f: int,
        features: Sequence[int] = (64, 128, 256, 256),
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.RELU,
        norm=Norm.BATCH,
        dropout=0.0,
        upsample: str = "deconv",
        save_attentionmap_fpath: Optional[str] = None,
        use_attention: bool = False,
        use_cbam: bool = False,
        use_mask: bool = False,
        use_aspp: bool = False,
        freeze_backbone: bool = False,
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
        self.down_1 = TwoConv(dimensions, features[0], features[1], act, norm, dropout)
        self.down_2 = TwoConv(dimensions, features[1], features[2], act, norm, dropout)
        self.down_3 = TwoConv(dimensions, features[2], features[3], act, norm, dropout)

        self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False, use_attention=use_attention)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False, use_attention=use_attention)

        self.final_conv = SimpleASPP(dimensions, last_feature, last_feature) if use_aspp else Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.maxpool = Pool["MAX", dimensions](kernel_size=2)
        self.avgpool = globalavgpool((1,)*dimensions)
        
        roi_chns = [roi_classes] + list(features[:-1])
        self.roi_convs = nn.ModuleList(
            [
                get_conv_layer(
                    dimensions,
                    chn,
                    roi_chns[i+1],
                    kernel_size=3,
                    stride=1,
                    act='relu',
                    norm='batch',
                    bias=True,
                    conv_only=False
                ) for i, chn in enumerate(roi_chns[:-1])
            ]
        )
        self.aag_layers = nn.ModuleList(
            # [AAG(dimensions, chn, chn) for chn in roi_chns[1:]]
            [AAG(dimensions, chn, chn, mode='cat') for chn in roi_chns[1:]]
        )

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
        self.final_fc = nn.Linear(features[-1], out_channels)
        self.fc_feat = nn.Linear(features[-1], out_channels_f)
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
        
        # if freeze_backbone:
        #     set_trainable(self, True)
            # set_trainable(self.final_fc, False)
        
    def get_att_code(self):
        def hook(model, input, output):
            self.att_code = output
        return hook

    def forward(self, x: torch.Tensor, label: torch.Tensor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        img, roi = x, masks
        x0 = self.conv_0(img)
        y = self.roi_convs[0](roi)
        
        x1 = self.maxpool(x0)
        y = self.maxpool(y)
        x1 = self.aag_layers[0](x1, y)
        x1 = self.down_1(x1)

        y = self.roi_convs[1](y)
        y = self.maxpool(y)
        x2 = self.maxpool(x1)
        x2 = self.aag_layers[1](x2, y)
        x2 = self.down_2(x2)

        y = self.roi_convs[2](y)
        y = self.maxpool(y)
        x3 = self.maxpool(x2)
        x3 = self.aag_layers[2](x3, y)
        x3 = self.down_3(x3)

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

        hesam_feat = torch.add(out_feat, he.squeeze().unsqueeze(-1))
        out_feat = self.fc_feat(hesam_feat.squeeze())
        
        hesam_MB = torch.add(out_MB, he.squeeze().unsqueeze(-1))
        logits = self.final_fc(hesam_MB.squeeze())

        if self.save_attentionmap_fpath is not None:
            coeffi_feat = self.fc_feat.weight
            for i in range(coeffi_feat.shape[0]):
                save_path_feat = check_dir(Path(self.save_attentionmap_fpath)/f'attentionmap-feature{i}')
                save_activation(x, res_out_feat, coeffi_feat, features, out_feat, (320,320), i, save_path_feat)

        if self.save_latent:
            return logits, out_feat, [hesam_MB.squeeze(), hesam_feat.squeeze()]
        else:
            return logits, out_feat

