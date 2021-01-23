import torch

from evaluate_model import evaluate_model

t_n = 2
t_f = 6
N_c = 64
N_f = 128


def bin_random(num_rays, num_bins, t_n, t_f, device="cpu"):
    t = torch.linspace(t_n, t_f, num_bins + 1, device=device)[:-1].repeat(num_rays, 1)
    t += torch.rand((num_rays, num_bins), device=device) * (t_f - t_n) / num_bins
    return t


def render(rays, N, t, model, device="cpu"):
    origins, dirs = rays

    # Reshaping to make broadcasting work properly
    points = origins.reshape((-1, 1, 3)) + dirs.reshape((-1, 1, 3)) * t.reshape((-1, N, 1))
    view_dirs = torch.zeros_like(points) + dirs.reshape((-1, 1, 3))
    torch.cuda.synchronize()
    c, density = evaluate_model(
        points.reshape((-1, 3)), view_dirs.reshape((-1, 3)), model, device=device
    )
    torch.cuda.synchronize()
    c = c.reshape((-1, N, 3))
    density = density.reshape((-1, N))

    delta = torch.zeros_like(density)
    delta[:, :-1] = t[:, 1:] - t[:, :-1]

    density_delta = density * delta

    presum_dd = torch.zeros_like(density_delta)
    presum_dd[:, 1:] = torch.cumsum(density_delta, dim=1)[:, :-1]

    T = torch.exp(-presum_dd)
    w = T * (1 - torch.exp(-density_delta))
    C = torch.sum(w.reshape((-1, N, 1)) * c, dim=1)

    torch.cuda.synchronize()
    return C, w


def render_rays(rays, coarse_model, model, device="cpu"):
    num_rays = len(rays[0])

    coarse_t = bin_random(num_rays, N_c, t_n, t_f, device=device)
    torch.cuda.synchronize()

    C_c, w = render(rays, N_c, coarse_t, coarse_model, device=device)
    torch.cuda.synchronize()

    return C_c
