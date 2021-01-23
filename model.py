import torch
import torch.nn as nn
from torch.nn import functional as F


class NERF(nn.Module):
    def __init__(self, in_pos=60, in_dir=24, W=256, D=8):
        super(NERF, self).__init__()
        self.in_pos = in_pos
        self.in_dir = in_dir

        self.density_layers = nn.ModuleList()
        self.color_layers = nn.ModuleList()

        layer_sizes = [in_pos] + [W] * D
        for i in range(len(layer_sizes) - 1):
            from_size = layer_sizes[i] + (in_pos if i == 5 else 0)
            to_size = layer_sizes[i + 1]
            self.density_layers.append(nn.Linear(from_size, to_size))
            self.density_layers.append(nn.ReLU())

        self.color_layers.append(nn.Linear(W + in_dir, W // 2))
        self.color_layers.append(nn.Linear(W // 2, 3))
        self.color_layers.append(nn.Sigmoid())

    def forward(self, x_in):
        x = 1 * x_in[:, : self.in_pos]
        for i, layer in enumerate(self.density_layers):
            if i == 10:
                x = torch.cat([x, x_in[:, : self.in_pos]], dim=1)
            x = layer(x)

        density = x[:, :1] + 0.1
        x = torch.cat([x, x_in[:, self.in_pos :]], dim=1)
        for layer in self.color_layers:
            x = layer(x)

        rgb = (x - 0.5) * 1.2 + 0.5
        return rgb, density
