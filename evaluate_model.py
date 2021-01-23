import torch
import math

L_pos = 10
L_dir = 4


def pos_encoding(x, L, device="cpu"):
    nx = torch.zeros((len(x), L * 2))
    for i in range(L):
        nx[:, 2 * i] = torch.cos(2 ** i * math.pi * x)
        nx[:, 2 * i + 1] = torch.sin(2 ** i * math.pi * x)
    return nx


def evaluate_model(points, dirs, model, device="cpu"):
    dirs /= torch.linalg.norm(dirs, dim=-1).reshape((-1, 1))

    nerf_input = torch.zeros((len(points), 6 * L_pos + 24), device=device)

    for i in range(3):
        f = (i * 2) * L_pos
        t = (i * 2 + 2) * L_pos
        nerf_input[:, f:t] = pos_encoding(points[:, i], L=L_pos, device=device)
    for i in range(3):
        f = 6 * L_pos + (i * 2) * L_dir
        t = 6 * L_pos + (i * 2 + 2) * L_dir
        nerf_input[:, f:t] = pos_encoding(dirs[:, i], L=L_dir, device=device)

    torch.cuda.synchronize()
    rgb, density = model(nerf_input.cuda())

    torch.cuda.synchronize()
    return rgb, density
