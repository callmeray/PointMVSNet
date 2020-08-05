import torch
import torch.nn as nn
import torch.nn.functional as F


def CGR(kernel_size, input_channel, output_channel, strides):
    pad = (kernel_size - 1) // 2
    G = max(1, output_channel // 8)
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, strides, pad, bias=False),
        nn.GroupNorm(G, output_channel),
        nn.ReLU(inplace=True)
    )


def DGR(kernel_size, input_channel, output_channel, strides):
    pad = (kernel_size - 1) // 2
    G = max(1, output_channel // 8)
    return nn.Sequential(
        nn.ConvTranspose2d(input_channel, output_channel, kernel_size, strides, pad, bias=False),
        nn.GroupNorm(G, output_channel),
        nn.ReLU(inplace=True)
    )


class UNetDS2GN(nn.Module):
    def __init__(self, base_channels):
        super(UNetDS2GN, self).__init__()

        input_channel = 3
        self.out_channels = base_channels * 4

        # aggregation
        # input x=>
        self.conv1_0 = CGR(3, input_channel, base_channels * 2, 2)
        self.conv2_0 = CGR(3, base_channels * 2, base_channels * 4, 2)
        self.conv3_0 = CGR(3, base_channels * 4, base_channels * 8, 2)
        self.conv4_0 = CGR(3, base_channels * 8, base_channels * 16, 2)

        self.conv0_1 = CGR(3, input_channel, base_channels, 1)
        self.conv0_2 = CGR(3, base_channels, base_channels, 1)

        self.conv1_1 = CGR(3, base_channels * 2, base_channels * 2, 1)
        self.conv1_2 = CGR(3, base_channels * 2, base_channels * 2, 1)
        self.conv2_1 = CGR(3, base_channels * 4, base_channels * 4, 1)
        self.conv2_2 = CGR(3, base_channels * 4, base_channels * 4, 1)
        self.conv3_1 = CGR(3, base_channels * 8, base_channels * 8, 1)
        self.conv3_2 = CGR(3, base_channels * 8, base_channels * 8, 1)
        self.conv4_1 = CGR(3, base_channels * 16, base_channels * 16, 1)
        self.conv4_2 = CGR(3, base_channels * 16, base_channels * 16, 1)
        self.conv5_0 = DGR(3, base_channels * 16, base_channels * 8, 2)

        # conv5_0 + conv3_2
        self.conv5_1 = CGR(3, base_channels * 16, base_channels * 8, 1)
        self.conv5_2 = CGR(3, base_channels * 8, base_channels * 8, 1)
        self.conv6_0 = DGR(3, base_channels * 8, base_channels * 4, 2)

        # conv6_0 + conv2_2
        self.conv6_1 = CGR(3, base_channels * 8, base_channels * 4, 1)
        self.conv6_2 = CGR(3, base_channels * 4, base_channels * 4, 1)
        self.conv7_0 = DGR(3, base_channels * 4, base_channels * 2, 2)

        # conv7_0 + conv1_2
        self.conv7_1 = CGR(3, base_channels * 4, base_channels * 2, 1)
        self.conv7_2 = CGR(3, base_channels * 2, base_channels * 2, 1)
        self.conv8_0 = DGR(3, base_channels * 2, base_channels, 2)

        # conv8_0 + conv0_2
        self.conv8_1 = CGR(3, base_channels * 2, base_channels, 1)
        self.conv8_2 = CGR(3, base_channels * 1, base_channels, 1)

        # 
        self.conv9_0 = CGR(5, base_channels, base_channels * 2, 2)
        self.conv9_1 = CGR(3, base_channels * 2, base_channels * 2, 1)
        self.conv9_2 = CGR(3, base_channels * 2, base_channels * 2, 1)
        self.conv10_0 = CGR(5, base_channels * 2, base_channels * 4, 2)
        self.conv10_1 = CGR(3, base_channels * 4, base_channels * 4, 1)
        self.conv10_2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1, bias=False)

    def forward(self, x):
        """

        Args:
            x: B*N*C*H*W

        Returns:
            B*N*C_out*(H/4)*(W/4)
        """
        batch_size, num_view, img_channel, img_height, img_width = x.shape
        x = x.view(batch_size * num_view, img_channel, img_height, img_width)

        f0_1 = self.conv0_1(x)
        f0_2 = self.conv0_2(f0_1)

        f1_0 = self.conv1_0(x)
        f2_0 = self.conv2_0(f1_0)
        f3_0 = self.conv3_0(f2_0)
        f4_0 = self.conv4_0(f3_0)

        f1_1 = self.conv1_1(f1_0)
        f1_2 = self.conv1_2(f1_1)

        f2_1 = self.conv2_1(f2_0)
        f2_2 = self.conv2_2(f2_1)

        f3_1 = self.conv3_1(f3_0)
        f3_2 = self.conv3_2(f3_1)

        f4_1 = self.conv4_1(f4_0)
        f4_2 = self.conv4_2(f4_1)

        f5_0 = self.conv5_0(f4_2)
        f5_0 = F.pad(f5_0, [0, 1, 0, 1])

        cat5_0 = torch.cat((f5_0, f3_2), dim=1)
        f5_1 = self.conv5_1(cat5_0)
        f5_2 = self.conv5_2(f5_1)
        f6_0 = self.conv6_0(f5_2)
        f6_0 = F.pad(f6_0, [0, 1, 0, 1])

        cat6_0 = torch.cat((f6_0, f2_2), dim=1)
        f6_1 = self.conv6_1(cat6_0)
        f6_2 = self.conv6_2(f6_1)
        f7_0 = self.conv7_0(f6_2)
        f7_0 = F.pad(f7_0, [0, 1, 0, 1])

        cat7_0 = torch.cat((f7_0, f1_2), dim=1)
        f7_1 = self.conv7_1(cat7_0)
        f7_2 = self.conv7_2(f7_1)
        f8_0 = self.conv8_0(f7_2)
        f8_0 = F.pad(f8_0, [0, 1, 0, 1])

        cat8_0 = torch.cat((f8_0, f0_2), dim=1)
        f8_1 = self.conv8_1(cat8_0)
        f8_2 = self.conv8_2(f8_1)
        f9_0 = self.conv9_0(f8_2)
        f9_1 = self.conv9_1(f9_0)
        f9_2 = self.conv9_2(f9_1)
        f10_0 = self.conv10_0(f9_2)
        f10_1 = self.conv10_1(f10_0)
        f10_2 = self.conv10_2(f10_1)

        out_feature = f10_2.contiguous().view(batch_size, num_view, self.out_channels, img_height // 4, img_width // 4)

        return out_feature
