import os
from numpy.lib.function_base import append

import torch
import torch.nn as nn
from typing import Callable
import numpy as np
from medlp.models.cnn.nets.resnet import ResNet, BasicBlock, Bottleneck
from medlp.models.cnn.layers.anatomical_gate import AnatomicalAttentionGate as AAG
from medlp.models.cnn.utils import set_trainable
from monai_ex.networks.layers import Act, Norm, Conv, Pool
from blocks.basic_block import UpCat, ResidualUnit
from nets.HESAM import MultiChannelLinear
from monai.networks.blocks.dynunet_block import get_conv_layer


class ResNetAAG(ResNet):
    def __init__(
        self, block, layers, dim=2, in_channels=3, roi_classes=1,
        num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
        replace_stride_with_dilation=None, norm_layer=None, freeze_backbone=False,
        out_channels_f: int = 4,
        sam_size: int = 6,
        use_cbam: bool = False,
        use_mask: bool = False,
    ):
        super().__init__(
            block=block, layers=layers, dim=dim, in_channels=in_channels, num_classes=num_classes,
            zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer
        )
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dim]
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dim]
        self.conv1_img = get_conv_layer(
            dim,
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            act='relu',
            norm='batch',
            bias=True,
            conv_only=False
        )
        if block == BasicBlock:
            roi_chns = [roi_classes, 64, 64, 128, 256]
            features = [64, 64, 128, 256, 512]
        else:
            roi_chns = [roi_classes, 64, 256, 512, 1024]
            features = [64, 64, 256, 512, 1024, 2048]

        self.roi_convs = nn.ModuleList(
            [
                get_conv_layer(
                    dim,
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
            [AAG(dim, chn, chn) for chn in roi_chns[1:]]
        )

        last_feature = 64
        self.upcat_4 = UpCat(dim, features[4], features[3], features[2], act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.upcat_3 = UpCat(dim, features[2], features[2], features[1], act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.upcat_2 = UpCat(dim, features[1], features[1], features[0], act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.upcat_1 = UpCat(dim, features[0], features[0], last_feature, act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.final_conv = Conv["conv", dim](last_feature, last_feature, kernel_size=1)

        self.res_block1_feat = ResidualUnit(
                dimensions=dim,
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
                dimensions=dim,
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
            globalavgpool((sam_size,)*dim),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dim), features[-1])
        )
        self.gmp = globalmaxpool(((1,)*dim))
        self.fc_feat = nn.Linear(features[-1], out_channels_f)


        if freeze_backbone:
            set_trainable(self, False)
            set_trainable(self.fc, True)

    def forward(self, inputs, label, features, masks):
        inp, roi = inputs, masks
        x = self.conv1_img(inp)
        y = self.roi_convs[0](roi)
        x1 = self.maxpool(x)
        y = self.maxpool(y)

        x1 = self.aag_layers[0](x1, y)
        x1 = self.layer1(x1)

        y = self.roi_convs[1](y)
        # y = self.maxpool(y)
        x1 = self.aag_layers[1](x1, y)
        x2 = self.layer2(x1)

        y = self.roi_convs[2](y)
        y = self.maxpool(y)
        x2 = self.aag_layers[2](x2, y)
        x3 = self.layer3(x2)

        y = self.roi_convs[3](y)
        y = self.maxpool(y)
        x3 = self.aag_layers[3](x3, y)
        x4 = self.layer4(x3)

        up_x4 = self.upcat_4(x4, x3)
        up_x3 = self.upcat_3(up_x4, x2)
        up_x2 = self.upcat_2(up_x3, x1)
        up_x1 = self.upcat_1(up_x2, x)
        res_out_feat = self.res_block1_feat(up_x1, masks)
        res_out_feat  = self.res_block2_feat(res_out_feat, masks)
        out_feat = self.sam_feat(res_out_feat)

        
        x_MB = self.avgpool(x4)
        x_MB_flatten = torch.flatten(x_MB, 1)
        x_MB = self.fc(x_MB_flatten)

        hesam = torch.add(out_feat, x_MB_flatten.unsqueeze(-1))
        out_feat = self.fc_feat(hesam.squeeze())
        # out_feat = self.fc_feat(out_feat.squeeze())

        return x_MB, out_feat

class ResNetAAG_v2(ResNet):
    def __init__(
        self, block, layers, dim=2, in_channels=3, roi_classes=1,
        num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
        replace_stride_with_dilation=None, norm_layer=None, freeze_backbone=False,
        out_channels_f: int = 4,
        sam_size: int = 6,
        use_cbam: bool = False,
        use_mask: bool = False,
    ):
        super().__init__(
            block=block, layers=layers, dim=dim, in_channels=in_channels, num_classes=num_classes,
            zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer
        )
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dim]
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dim]
        self.out_channels_f = out_channels_f
        self.conv1_img = get_conv_layer(
            dim,
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            act='relu',
            norm='batch',
            bias=True,
            conv_only=False
        )
        if block == BasicBlock:
            roi_chns = [roi_classes, 64, 64, 128, 256]
            features = [64, 64, 128, 256, 512]
        else:
            roi_chns = [roi_classes, 64, 256, 512, 1024]
            features = [64, 64, 256, 512, 1024, 2048]

        self.roi_convs = nn.ModuleList(
            [
                get_conv_layer(
                    dim,
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
            [AAG(dim, chn, chn) for chn in roi_chns[1:]]
        )

        last_feature = 64
        self.upcat_4 = UpCat(dim, features[4], features[3], features[2], act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.upcat_3 = UpCat(dim, features[2], features[2], features[1], act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.upcat_2 = UpCat(dim, features[1], features[1], features[0], act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.upcat_1 = UpCat(dim, features[0], features[0], last_feature, act=Act.RELU, norm=Norm.BATCH, dropout=0.0, upsample='deconv', halves=False)
        self.final_conv = Conv["conv", dim](last_feature, last_feature, kernel_size=1)

        self.res_block1_feat_layers = nn.ModuleList(
            [ResidualUnit(
                dimensions=dim,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
                ) for i in range(out_channels_f)]
        )
        self.res_block2_feat_layers = nn.ModuleList(
            [ResidualUnit(
                dimensions=dim,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                adn_ordering=["NDA", "ND"],
                act=Act.RELU,
                norm=Norm.BATCH,
                use_cbam=use_cbam,
                use_mask=use_mask,
            ) for i in range(out_channels_f)]
        )

        self.sam_feat_layers = nn.ModuleList(
            [nn.Sequential(
            globalavgpool((sam_size,)*dim),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dim), features[-1])) for i in range(out_channels_f)]
        )
        
        self.gmp = globalmaxpool(((1,)*dim))
        self.fc_feat_layers = nn.ModuleList(
            [nn.Linear(features[-1], 1) for i in range(out_channels_f)]
        )

        if freeze_backbone:
            set_trainable(self, False)
            set_trainable(self.fc, True)

    def forward(self, inputs, label, features, masks):
        inp, roi = inputs, masks
        x = self.conv1_img(inp)
        y = self.roi_convs[0](roi)
        x1 = self.maxpool(x)
        y = self.maxpool(y)

        x1 = self.aag_layers[0](x1, y)
        x1 = self.layer1(x1)

        y = self.roi_convs[1](y)
        # y = self.maxpool(y)
        x1 = self.aag_layers[1](x1, y)
        x2 = self.layer2(x1)

        y = self.roi_convs[2](y)
        y = self.maxpool(y)
        x2 = self.aag_layers[2](x2, y)
        x3 = self.layer3(x2)

        y = self.roi_convs[3](y)
        y = self.maxpool(y)
        x3 = self.aag_layers[3](x3, y)
        x4 = self.layer4(x3)

        up_x4 = self.upcat_4(x4, x3)
        up_x3 = self.upcat_3(up_x4, x2)
        up_x2 = self.upcat_2(up_x3, x1)
        up_x1 = self.upcat_1(up_x2, x)

        out_feat_list = []
        for i in range(self.out_channels_f):
            res_out_feat = self.res_block1_feat_layers[i](up_x1, masks)
            res_out_feat  = self.res_block2_feat_layers[i](res_out_feat, masks)
            out_feat = self.sam_feat_layers[i](res_out_feat)
            out_feat_list.append(out_feat)
        
        x_MB = self.avgpool(x4)
        x_MB_flatten = torch.flatten(x_MB, 1)
        x_MB = self.fc(x_MB_flatten)

        final_out_feat_list = []
        for i in range(self.out_channels_f):
            hesam = torch.add(out_feat_list[i], x_MB_flatten.unsqueeze(-1))
            out_feat = self.fc_feat_layers[i](hesam.squeeze())
            final_out_feat_list.append(out_feat)
        if len(final_out_feat_list[0].shape) == 1:
            out_feat = torch.cat([final_out_feat_list[i] for i in range(self.out_channels_f)])
        else:
            out_feat = torch.cat([final_out_feat_list[i] for i in range(self.out_channels_f)], dim=1)

        return x_MB, out_feat


def resnet18_aag_sam(pretrained_model_path, **kwargs):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        num_classes_ = kwargs['num_classes']
        kwargs['num_classes'] = 1

    model = ResNetAAG(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        in_features_ = model.fc.in_features
        model.fc = nn.Linear(in_features_, num_classes_)

    return model

def resnet34_aag_sam(pretrained_model_path, **kwargs):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        num_classes_ = kwargs['num_classes']
        kwargs['num_classes'] = 1

    model = ResNetAAG(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        in_features_ = model.fc.in_features
        model.fc = nn.Linear(in_features_, num_classes_)

    return model

def resnet34_aag_sam_v2(pretrained_model_path, **kwargs):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        num_classes_ = kwargs['num_classes']
        kwargs['num_classes'] = 1

    model = ResNetAAG_v2(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        in_features_ = model.fc.in_features
        model.fc = nn.Linear(in_features_, num_classes_)

    return model