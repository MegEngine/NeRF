#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import imageio
import numpy as np
import megengine as mge
import megengine.functional as F
import time
from collections import defaultdict
from loguru import logger
from tqdm import tqdm

from nerf.utils import get_rays, ndc_rays, to8b, sample_pdf, cumprod

__all__ = [
    "batchify_rays",
    "render_rays",
    "render_path",
    "render",
]


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = defaultdict(list)
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i: i + chunk], **kwargs)
        for k, v in ret.items():
            all_ret[k].append(v)

    return {k: F.concat(all_ret[k], 0) for k in all_ret}


def render(
    H: int,
    W: int,
    K: float,
    chunk: int = 1024 * 32,
    rays=None,
    c2w=None,
    ndc: bool = True,
    near: float = 0.0,
    far: float = 1.0,
    use_viewdirs: bool = False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays

    Args:
      H (int): Height of image in pixels.
      W (int): Width of image in pixels.
      K: intrinsics of camera
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / F.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = viewdirs.reshape(-1, 3)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    near, far = near * F.ones_like(rays_d[..., :1]), far * F.ones_like(rays_d[..., :1])
    rays = F.concat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = F.concat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k, v in all_ret.items():
        all_ret[k] = v.reshape(list(sh[:-1]) + list(v.shape[1:]))

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: v for k, v in all_ret.items() if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs, disps = [], []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        logger.info(f"{i} {time.time() - t}")
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            logger.info(f"rgb shape: {rgb.shape}, disp shape: {disp.shape}")

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.

    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.

    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.

    """
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - F.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = F.concat([dists, F.full(dists[..., :1].shape, 1e10)], -1)  # [N_rays, N_samples]

    dists = dists * F.norm(F.expand_dims(rays_d, axis=-2), axis=-1)

    rgb = F.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = mge.random.normal(size=raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = mge.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = (
        alpha * cumprod(
            F.concat([F.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1
        )[:, :-1]
    )
    rgb_map = F.sum(F.expand_dims(weights, axis=-1) * rgb, -2)  # [N_rays, 3]

    depth_map = F.sum(weights * z_vals, -1)
    disp_map = 1.0 / F.maximum(
        1e-10 * F.ones_like(depth_map), depth_map / F.sum(weights, -1)
    )
    acc_map = F.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    pytest=False,
):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = ray_batch[..., 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Generate sample bins
    bin_vals = F.linspace(0.0, 1.0, N_samples)
    if not lindisp:
        z_vals = near + (far - near) * bin_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - bin_vals) + 1.0 / far * (bin_vals))
    z_vals = F.broadcast_to(z_vals, [N_rays, N_samples])

    if perturb > 0.0:
        # get intervals between samples
        mids = (z_vals[..., :-1] + z_vals[..., 1:]) / 2.0
        upper = F.concat([mids, z_vals[..., -1:]], -1)
        lower = F.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = mge.random.uniform(size=z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = mge.Tensor(np.random.rand(*list(z_vals.shape)))

        z_vals = lower + (upper - lower) * t_rand

    pts = F.expand_dims(rays_o, axis=-2) + F.expand_dims(rays_d, axis=-2) * F.expand_dims(z_vals, axis=-1)  # noqa
    # shape of pts: [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            N_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        )
        z_samples = z_samples.detach()

        # note that sort in megengine is different from torch
        z_vals, _ = F.sort(F.concat([z_vals, z_samples], -1,), descending=False)
        pts = F.expand_dims(rays_o, -2) + F.expand_dims(rays_d, -2) * F.expand_dims(z_vals, -1)
        # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = F.std(z_samples, axis=-1)  # [N_rays]

    return ret
