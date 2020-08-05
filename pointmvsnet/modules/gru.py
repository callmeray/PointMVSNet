import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super(GRU, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = F.sigmoid(rn)
        uns = F.sigmoid(un)
        return rns, uns

    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h=None):
        N, C, H, W = x.shape
        HC = self.output_channel
        if (h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
        return u * h + (1 - u) * y
