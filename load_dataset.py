import json
import cv2
import numpy as np

# import open3d as o3d
import torch


def get_c2ws_imgs(path, image_size):
    frames = json.load(open(path + "transforms_train.json", "r"))["frames"]
    c2ws = []
    imgs = []
    for f in frames:
        img = cv2.imread(path + f["file_path"][2:] + ".png")[:, :, ::-1]
        img = cv2.resize(img, (image_size, image_size))
        imgs.append(np.float32(img)/255)
        c2ws.append(f["transform_matrix"])

    return c2ws, imgs


def apply_homography(c2w, points):
    homo_points = np.ones((len(points), 4, 1))
    homo_points[:, :3, 0] = points
    return np.matmul(c2w, homo_points)[:, :3, 0]


def c2w_to_rays(c2w, width, height):
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    x = x.flatten()
    y = y.flatten()

    num_rays = width * height
    origins = np.zeros((num_rays, 3))
    origins = apply_homography(c2w, origins)
    dirs = np.array([x, y, -np.ones(num_rays)]).T
    dirs = apply_homography(c2w, dirs) - origins

    return origins, dirs


def load_dataset(path, image_size):
    c2ws, imgs = get_c2ws_imgs(path, image_size)

    all_origins = []
    all_dirs = []
    all_colors = []

    # pcs = []
    for c2w, img in zip(c2ws[:50], imgs):
        origins, dirs = c2w_to_rays(c2w, img.shape[1], img.shape[0])
        colors = img.reshape((-1, 3))

        all_origins.append(origins)
        all_dirs.append(dirs)
        all_colors.append(colors)

        """pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(origins)
        pc1.paint_uniform_color([1, 0, 0])
        pc3 = o3d.geometry.PointCloud()
        pc3.points = o3d.utility.Vector3dVector(dirs + origins)
        pc3.paint_uniform_color([0, 0, 1])
        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(dirs * 2 + origins)
        pc2.paint_uniform_color([0, 1, 0])
        pcs.append(pc1)
        pcs.append(pc2)
        pcs.append(pc3)
        print(origins[0])"""
    # o3d.visualization.draw_geometries(pcs)

    origins = torch.tensor(np.concatenate(all_origins))
    dirs = torch.tensor(np.concatenate(all_dirs))
    colors = torch.tensor(np.concatenate(all_colors))

    return (origins, dirs), colors
