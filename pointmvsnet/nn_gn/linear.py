from torch import nn
import torch.nn.functional as F

from .init import init_uniform


class FC(nn.Module):
    """Applies a linear transformation to the incoming dataloader
    optionally followed by batch normalization and relu activation

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (nn.Module, optional): relu activation module

    """

    def __init__(self, in_channels, out_channels,
                 bias=True, gn=True, num_group=8, relu=True):
        super(FC, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.gn = nn.GroupNorm(num_group, out_channels) if gn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.fc(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.fc)
