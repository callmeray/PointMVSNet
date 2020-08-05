from torch import nn
import torch.nn.functional as F

from .init import init_uniform, init_bn


class FC(nn.Module):
    """Applies a linear transformation to the incoming data
    optionally followed by batch normalization and relu activation

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    """

    def __init__(self, in_channels, out_channels,
                 relu=True, bn=True, bn_momentum=0.1):
        super(FC, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels, bias=(not bn))
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.fc)
        if self.bn is not None:
            init_bn(self.bn)
