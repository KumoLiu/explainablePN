import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample

class Attention_block(nn.Module):
    """
    :param F_g: feat_map after upsample
    :param F_l: output from last block or skip
    """
    def __init__(self, spatial_dims, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = Convolution(
            spatial_dims,
            F_l,
            F_int,
            adn_ordering='N',
            norm="BATCH",
            strides=1,
            kernel_size=1,
            padding=0,
        )
        self.W_x = Convolution(
            spatial_dims,
            F_g,
            F_int,
            adn_ordering='N',
            norm="BATCH",
            strides=1,
            kernel_size=1,
            padding=0,
        )
        self.psi = Convolution(
            spatial_dims,
            F_int+1,
            1,
            adn_ordering='NA',
            norm="BATCH",
            act='SIGMOID',
            strides=1,
            kernel_size=1,
            padding=0,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, masks):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        _masks = torch.nn.functional.interpolate(masks, size=g1.shape[2:], mode='nearest')
        pre_relu = torch.cat([g1+x1, _masks], dim=1)
        psi = self.relu(pre_relu)
        psi = self.psi(psi)
        out = x * psi
        return out