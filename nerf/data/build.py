#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from loguru import logger

from .load_llff import load_llff_data
from .load_deepvoxels import load_dv_data
from .load_blender import load_blender_data
from .load_LINEMOD import load_LINEMOD_data


def build_loader(dataset_type, args):
    K = None
    if dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        logger.info(f"Loaded llff {images.shape} {render_poses.shape} {hwf} {args.datadir}")
        num_images = images.shape[0]
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            logger.info(f"Auto LLFF holdout, {args.llffhold}")
            i_test = np.arange(num_images)[:: args.llffhold]

        i_val = i_test
        i_train = np.array(
            [i for i in range(num_images) if (i not in i_test and i not in i_val)]
        )
        i_split = (i_train, i_val, i_test)

        logger.info("DEFINING BOUNDS")
        if args.no_ndc:
            near, far = np.ndarray.min(bds) * 0.9, np.ndarray.max(bds) * 1.0
        else:
            near, far = 0.0, 1.0
        logger.info(f"NEAR: {near} FAR: {far}")

    elif dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip
        )
        logger.info(f"Loaded blender {images.shape} {render_poses.shape} {hwf} {args.datadir}")
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]
        near, far = 2.0, 6.0

    elif dataset_type == "LINEMOD":
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip
        )
        logger.info(f"Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}")
        logger.info(f"[CHECK HERE] near: {near}, far: {far}.")

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif dataset_type == "deepvoxels":
        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip
        )
        logger.info(f"Loaded deepvoxels {images.shape} {render_poses.shape} {hwf} {args.datadir}")
        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near, far = hemi_R - 1.0, hemi_R + 1.0

    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    # cast height and wigth to right types
    height, width, focal = hwf
    height, width = int(height), int(width)
    hwf = [height, width, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * width],
            [0, focal, 0.5 * height],
            [0, 0, 1]
        ])

    return images, poses, render_poses, hwf, i_split, near, far, K
