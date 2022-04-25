import os
import megengine as mge
import megengine.functional as F
import numpy as np
import math


__all__ = [
    "ensure_dir",
    "img2mse",
    "mse2psnr",
    "to8b",
    "get_rays",
    "get_rays_np",
    "ndc_rays",
    "sample_pdf",
    "meshgrid",
    "cumprod",
]


def cumprod(x: mge.Tensor, axis: int):
    dim = x.ndim
    axis = axis if axis > 0 else axis + dim
    num_loop = x.shape[axis]
    t_shape = [i + 1 if i < axis else i for i in range(dim)]
    t_shape[axis] = 0
    x = x.transpose(*t_shape)
    assert len(x) == num_loop
    cum_val = F.ones(x[0].shape)
    for i in range(num_loop):
        cum_val *= x[i]
        x[i] = cum_val
    return x.transpose(*t_shape)


def ensure_dir(path: str):
    """create directories if *path* does not exist"""
    if not os.path.isdir(path):
        os.makedirs(path)


def meshgrid(x, y, indexing="xy"):
    """meshgrid wrapper for megengine"""
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    if indexing == "ij":
        mesh_x, mesh_y = mesh_x.T, mesh_y.T
    return mesh_x, mesh_y


# Misc
def img2mse(x, y):
    return F.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * (F.log(x) / math.log(10.0))


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = meshgrid(F.linspace(0, W - 1, W), F.linspace(0, H - 1, H), indexing="xy")
    dirs = F.stack(
        [
            (i - float(K[0][2])) / float(K[0][0]),
            -(j - float(K[1][2])) / float(K[1][1]),
            -F.ones_like(i),
        ], -1
    )
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = (F.expand_dims(dirs, axis=-2) * c2w[:3, :3]).sum(axis=-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = F.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H: int, W: int, K: np.array, c2w: np.array):
    """

    Args:
        H (int): height of image.
        W (int): width of image.
        K (np.array): intrinsic matrix.
        c2w (np.array): camera to world matrix.
    """
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    # K @ dirs is (x, -y0, -1), and K is intrinsics of camera
    dirs = np.stack([(i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]  # noqa
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    get ray o' and d' in normalized device coordinates space.
    check more details in appendix C of nerf paper.

    Args:
        rays_o: rays origin of shape (N, 3).
        rays_d: rays direction of shape (N, 3).
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + F.expand_dims(t, axis=-1) * rays_d

    # Projection
    # according to paper, a_x = - f_cam / (W / 2), a_y = -f_cam / (H / 2)
    a_x, a_y = - float((2.0 * focal) / W), - float((2.0 * focal) / H)
    o_x = a_x * rays_o[..., 0] / rays_o[..., 2]
    o_y = a_y * rays_o[..., 1] / rays_o[..., 2]
    o_z = 1.0 + 2.0 * near / rays_o[..., 2]

    d_x = a_x * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d_y = a_y * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d_z = -2.0 * near / rays_o[..., 2]

    rays_o = F.stack([o_x, o_y, o_z], -1)
    rays_d = F.stack([d_x, d_y, d_z], -1)

    return rays_o, rays_d


def search_sorted(cdf, value):
    # TODO: torch to pure mge
    import torch
    inds = torch.searchsorted(torch.tensor(cdf.numpy()), torch.tensor(value.numpy()), right=True)
    inds = mge.tensor(inds)
    return inds


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / F.sum(weights, -1, keepdims=True)
    cdf = F.cumsum(pdf, -1 + pdf.ndim)
    cdf = F.concat([F.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = F.linspace(0.0, 1.0, N_samples)
        u = F.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = mge.random.uniform(size=list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = mge.Tensor(u)

    # Invert CDF
    inds = search_sorted(cdf, u)
    below = F.maximum(F.zeros_like(inds - 1), inds - 1)
    above = F.minimum((cdf.shape[-1] - 1) * F.ones_like(inds), inds)
    inds_g = F.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = F.gather(F.broadcast_to(F.expand_dims(cdf, axis=1), matched_shape), 2, inds_g)
    bins_g = F.gather(F.broadcast_to(F.expand_dims(bins, axis=1), matched_shape), 2, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = F.where(denom < 1e-5, F.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
