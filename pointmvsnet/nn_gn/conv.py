from torch import nn
import torch.nn.functional as F

from .init import init_uniform


class Conv1d(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes.
    optionally followed by batch normalization and relu activation

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (nn.Module, optional): relu activation module

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 gn=True, num_group=4, relu=True, **kwargs):
        super(Conv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        self.gn = nn.GroupNorm(num_group, out_channels) if gn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (nn.Module, optional): relu activation module

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 gn=True, num_group=4, relu=True, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.gn = nn.GroupNorm(num_group, out_channels) if gn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 gn=True, num_group=4, relu=True, **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        self.gn = nn.GroupNorm(num_group, out_channels) if gn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size,
                 gn=True, num_group=4, relu=True, **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, **kwargs)
        self.gn = nn.GroupNorm(num_group, out_channels) if gn else None
        self.relu = relu

        self.init_weights()

    def forward(self, x):
        y = self.conv(x)
        if self.gn is not None:
            x = self.gn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self):
        """default initialization"""
        init_uniform(self.conv)
