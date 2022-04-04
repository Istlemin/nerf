import torch

t_n = 1
t_f = 7
N_c = 200
N_f = 64


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

    #print(points.min(dim=0)[0].min(dim=0))
    #print(points.max(dim=0)[0].max(dim=0))
    
    c, density = model(
        points.reshape((-1, 3)), view_dirs.reshape((-1, 3)), device=device
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

def render_rays(rays, model, device="cpu"):
    num_rays = len(rays[0])
    
    coarse_t = bin_random(num_rays, N_c, t_n, t_f, device=device)
    torch.cuda.synchronize()
    C_c, w = render(rays, N_c, coarse_t, model, device=device)
    return C_c

def render_rays_hierarchal(rays, coarse_model, fine_model, device="cpu"):
    num_rays = len(rays[0])

    coarse_t = bin_random(num_rays, N_c, t_n, t_f, device=device)
    torch.cuda.synchronize()

    # Course render
    C_c, w = render(rays, N_c, coarse_t, coarse_model, device=device)
    torch.cuda.synchronize()

    bin_start = torch.zeros_like(coarse_t)
    bin_end = torch.zeros_like(coarse_t)
    bin_end[:,-1] = t_f
    bin_start[:,0] = t_n
    bin_start[:,1:] = bin_end[:,:-1] = (coarse_t[:,:-1]+coarse_t[:,1:])/2

    prob = (bin_end-bin_start)*w
    prob = torch.maximum(prob,torch.tensor(0.000001,device=device))

    chosen_bin_inds = torch.distributions.Categorical(prob).sample((N_f,)).sort(dim=0)[0].T
    chosen_bin_starts = bin_start[torch.arange(num_rays,device=device).repeat_interleave(N_f), chosen_bin_inds.flatten()]
    chosen_bin_ends = bin_end[torch.arange(num_rays,device=device).repeat_interleave(N_f), chosen_bin_inds.flatten()]
    x = torch.rand(chosen_bin_starts.shape,device=device)
    fine_t = chosen_bin_starts + x*(chosen_bin_ends-chosen_bin_starts)
    fine_t = fine_t.reshape((num_rays,N_f))

    C_f, w = render(rays, N_f, fine_t, fine_model, device=device)

    return C_c, C_f
