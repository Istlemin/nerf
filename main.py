from PIL import Image
import numpy as np
import tqdm

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

from load_dataset import load_dataset
from model import NERF
from render import render_rays


def run_epoch(rays, colors, model, optimizer, train=True, bs=2000):
    origins, dirs = rays
    dataset = TensorDataset(origins, dirs, colors)
    loader = DataLoader(dataset, batch_size=bs, shuffle=train)

    output_C = []

    losses = []

    # scaler = torch.cuda.amp.GradScaler()

    # for i in range(0, len(origins), bs):
    #    b_origins = origins[i : i + bs]
    #    b_dirs = dirs[i : i + bs]
    #    b_colors = colors[i : i + bs]

    for b_origins, b_dirs, b_colors in loader:

        optimizer.zero_grad()

        C = render_rays((b_origins, b_dirs), model, model, device="cuda")
        loss = F.mse_loss(b_colors, C)
        loss.backward()
        losses.append(loss.detach().cpu().numpy())

        if train:
            optimizer.step()
            # scaler.step(optimizer)
        # scaler.update()

        output_C.append(C)

    return torch.cat(output_C, dim=0), np.mean(losses)


def train(rays, colors, model, width, height):
    colors = colors.type(torch.float32) / 255
    origins, dirs = rays

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for i in range(100000):
        C, loss = run_epoch(rays, colors, model, optimizer)
        print(loss, i)

        if i % 10 == -1:
            C, loss = run_epoch(
                (origins[: 200 * 200], dirs[: 200 * 200]),
                colors[: 200 * 200],
                model,
                optimizer,
                train=False,
            )
            out_img = C.detach().cpu().numpy().reshape((height, width, 3))
            Image.fromarray(np.uint8(np.clip((out_img) * 255, 0, 255))).show()


def main():
    (origins, dirs), colors = load_dataset("C:/Users/Fredrik/repos/test_data/nerf_synthetic/lego/")

    model = NERF().cuda()
    train((origins.cuda(), dirs.cuda()), colors.cuda(), model.cuda(), 200, 200)


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main()
