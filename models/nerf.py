import torch
import torch.nn as nn
from torch.nn import functional as F
import math

L_pos = 10
L_dir = 4

def pos_encoding(x, L, device="cpu"):
    nx = torch.zeros((len(x), L * 2))
    for i in range(L):
        nx[:, 2 * i] = torch.cos(2 ** i * math.pi * x)
        nx[:, 2 * i + 1] = torch.sin(2 ** i * math.pi * x)
    return nx

class NERF(nn.Module):
    def __init__(self, in_pos=60, in_dir=24, W=100, D=5):
        super(NERF, self).__init__()
        self.in_pos = in_pos
        self.in_dir = in_dir

        self.density_layers = nn.ModuleList()
        self.color_layers = nn.ModuleList()

        layer_sizes = [in_pos] + [W] * D
        for i in range(len(layer_sizes) - 1):
            from_size = layer_sizes[i] + (in_pos if i == 2 else 0)
            to_size = layer_sizes[i + 1]
            self.density_layers.append(nn.Linear(from_size, to_size))
            self.density_layers.append(nn.ReLU())

        self.color_layers.append(nn.Linear(W + in_dir, W // 2))
        self.color_layers.append(nn.ReLU())
        self.color_layers.append(nn.Linear(W // 2, 3))
        self.color_layers.append(nn.Sigmoid())

    def forward(self, points, dirs, device="cpu"):
        dirs /= torch.linalg.norm(dirs, dim=-1).reshape((-1, 1))

        x_in = torch.zeros((len(points), 6 * L_pos + 24), device=device)

        for i in range(3):
            f = (i * 2) * L_pos
            t = (i * 2 + 2) * L_pos
            x_in[:, f:t] = pos_encoding(points[:, i], L=L_pos, device=device)
        for i in range(3):
            f = 6 * L_pos + (i * 2) * L_dir
            t = 6 * L_pos + (i * 2 + 2) * L_dir
            x_in[:, f:t] = pos_encoding(dirs[:, i], L=L_dir, device=device)

        x = 1 * x_in[:, : self.in_pos]
        for i, layer in enumerate(self.density_layers):
            if i == 4:
                x = torch.cat([x, x_in[:, : self.in_pos]], dim=1)
            x = layer(x)

        density = x[:, :1] + 0.1
        x = torch.cat([x, x_in[:, self.in_pos :]], dim=1)
        for layer in self.color_layers:
            x = layer(x)

        rgb = (x - 0.5) * 1.2 + 0.5
        return rgb, density
