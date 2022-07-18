from math import acos
import torch
from torch import nn
from typing import Callable
from monai_ex.networks.layers import Conv, Norm, Pool


class ChannelAttention(nn.Module):
    def __init__(self, dim, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        conv_type: Callable = Conv[Conv.CONV, dim]
        maxpool_type: Callable = Pool[Pool.MAX, dim]
        avgpool_type: Callable = Pool[Pool.AVG, dim]
        self.avg_pool = avgpool_type(1)
        self.max_pool = maxpool_type(1)

        self.fc1   = conv_type(in_planes, in_planes // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2   = conv_type(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        atten_out = x * self.sigmoid(out)
        return atten_out


class SpatialAttention(nn.Module):
    def __init__(self, dim, channel, use_mask=False, kernel_size=7):
        super(SpatialAttention, self).__init__()
        conv_type: Callable = Conv[Conv.CONV, dim]

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = conv_type(2, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = conv_type(channel, channel//2, kernel_size, padding=padding, bias=False)
        self.conv3 = conv_type(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.use_mask = use_mask

    def forward(self, input, masks):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)

        if self.use_mask:
            _slice = int((masks.shape[1]+1)/2-1)
            _masks = torch.nn.functional.interpolate(masks, size=avg_out.shape[2:], mode='nearest')[:,_slice:_slice+1,...]

            x = torch.cat([avg_out, _masks], dim=1)
            x = self.conv1(x)
            x_act = self.sigmoid(x)
            out = input * x_act
            return out
        else:
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            x_act = self.sigmoid(x)
            return input * x_act

class CBAM(nn.Module):
    def __init__(self, dim, in_planes, use_mask=False):
        super().__init__()
        self.ChannelGate = ChannelAttention(dim, in_planes)
        self.SpatialGate = SpatialAttention(dim, in_planes*2, use_mask=use_mask, kernel_size=7)
        
    def forward(self, x, masks):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out, masks)
        return x_out

