import torch
import torch.nn as nn

from pointmvsnet.nn_gn.conv import Conv2d


class ConvGRUCell(nn.Module):
    """A GRU cell with convolutions"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 num_group=1,
                 activation=torch.tanh):
        super(ConvGRUCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate_conv = Conv2d(in_channels + out_channels, 2 * out_channels, kernel_size,
                                padding=(kernel_size - 1) // 2, gn=False, relu=False)
        self.reset_norm = nn.GroupNorm(num_group, out_channels)
        self.update_norm = nn.GroupNorm(num_group, out_channels)

        self.output_conv = Conv2d(in_channels + out_channels, out_channels, kernel_size,
                                  padding=(kernel_size - 1) // 2, num_group=num_group, relu=False)
        self.activation = activation

    def forward(self, x, h):
        input = torch.cat((x, h), dim=1)
        conv = self.gate_conv(input)
        reset_gate = conv[:, :self.out_channels].contiguous()
        update_gate = conv[:, self.out_channels:].contiguous()

        # normalization
        reset_gate = self.reset_norm(reset_gate)
        update_gate = self.update_norm(update_gate)

        # activation
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        input = torch.cat((x, reset_gate * h), dim=1)
        output_conv = self.output_conv(input)
        y = self.activation(output_conv)

        output = update_gate * h + (1 - update_gate) * y

        return output, output
