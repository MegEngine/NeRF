#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime
import numpy as np
import os
import imageio
import time
from loguru import logger

import megengine as mge
import megengine.functional as F
import megengine.distributed as dist
from megengine.autodiff import GradManager

from nerf.data import build_loader
from nerf.models.build import create_nerf
from nerf.utils import (
    ensure_dir, get_rays, get_rays_np, to8b, img2mse, mse2psnr, render_path, render, meshgrid
)


def rendering_only(args, images, i_test, render_poses, hwf, K, render_kwargs_test, start_iter):
    logger.info("RENDER ONLY")
    # render_test switches to test poses, Default is smoother render_poses path
    images = images[i_test] if args.render_test else None

    testsavedir = os.path.join(
        args.basedir,
        args.expname,
        "renderonly_{}_{:06d}".format("test" if args.render_test else "path", start_iter),
    )
    ensure_dir(testsavedir)
    logger.info(f"test poses shape: {render_poses.shape}")

    rgbs, _ = render_path(
        render_poses,
        hwf,
        K,
        args.chunk,
        render_kwargs_test,
        savedir=testsavedir,
        render_factor=args.render_factor,
    )
    logger.info(f"Done rendering {testsavedir}")
    imageio.mimwrite(os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8)


class Trainer:
    def __init__(self, args):
        # init function only defines some basic attr, other attrs like model, optimizer
        # are built in `before_train` methods.
        self.args = args
        self.start_iter, self.max_iter = (0, 200000)
        self.rank = 0
        self.amp_training = False

    def train(self):
        self.before_train()
        try:
            self.train_in_iter()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        args = self.args

        logger.info(f"Full args:\n{args}")
        # model related init
        images, poses, render_poses, hwf, i_split, near, far, K = build_loader(args.dataset_type, args)  # noqa

        i_train, i_val, i_test = i_split
        info_string = f"Train views are {i_train}\nTest views are {i_test}\nVal views are {i_val}"
        if args.render_test:
            render_poses = np.array(poses[i_test])

        self.i_split = i_split
        self.hwf = hwf
        self.K = K
        self.render_poses = mge.Tensor(render_poses)

        # create dir to save all the results
        self.save_dir = os.path.join(args.basedir, args.expname)
        ensure_dir(self.save_dir)
        logger.add(os.path.join(self.save_dir, "log.txt"))

        # save args.txt config.txt
        with open(os.path.join(self.save_dir, "args.txt"), "w") as f:
            for arg in sorted(vars(args)):
                f.write(f"{arg} = {getattr(args, arg)}\n")

        if args.config is not None:
            with open(os.path.join(self.save_dir, "config.txt"), "w") as f:
                f.write(open(args.config, "r").read())

        # Create nerf model
        render_kwargs_train, render_kwargs_test, optimizer, gm = self.build_nerf()
        bds_dict = {"near": near, "far": far}
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.optimizer = optimizer
        self.grad_manager = gm

        # Short circuit if only rendering out from trained model
        if args.render_only:
            rendering_only(
                args, images, i_test, self.render_poses, self.hwf, self.K,
                self.render_kwargs_test, self.start_iter,
            )
            return

        # Prepare raybatch tensor if batching random rays
        use_batching = not args.no_batching
        assert use_batching
        if use_batching:
            # For random ray batching
            logger.info("get rays")
            rays = np.stack(
                [get_rays_np(self.hwf[0], self.hwf[1], self.K, p) for p in poses[:, :3, :4]], 0
            )  # [N, ro+rd, H, W, 3]
            logger.info("get rays done, start concats")
            rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            logger.info("shuffle rays")
            np.random.shuffle(rays_rgb)
            logger.info("shuffle rasy done")

            images = mge.Tensor(images)
            rays_rgb = mge.Tensor(rays_rgb)
            i_batch = 0

        self.poses = mge.Tensor(poses)
        self.images = images
        self.rays_rgb = rays_rgb
        self.i_batch = i_batch

        logger.info("Begin training\n" + info_string)

    def build_nerf(self):
        args = self.args

        model, model_fine, network_query_fn = create_nerf(args)
        params = list(model.parameters())
        logger.info(f"Model:\n{model}")
        if model_fine is not None:
            logger.info(f"Model Fine:\n{model_fine}")
            params += list(model_fine.parameters())

        gm = GradManager()
        world_size = dist.get_world_size()
        callbacks = [dist.make_allreduce_cb("MEAN", dist.WORLD)] if world_size > 1 else None  # noqa
        gm.attach(params, callbacks=callbacks)

        optimizer = mge.optimizer.Adam(params=params, lr=args.lr, betas=(0.9, 0.999))
        self.resume_ckpt(model, model_fine, optimizer)

        render_kwargs_train = {
            "network_query_fn": network_query_fn,
            "perturb": args.perturb,
            "N_importance": args.N_importance,
            "network_fine": model_fine,
            "N_samples": args.N_samples,
            "network_fn": model,
            "use_viewdirs": args.use_viewdirs,
            "white_bkgd": args.white_bkgd,
            "raw_noise_std": args.raw_noise_std,
        }

        # NDC only good for LLFF-style forward facing data
        if args.dataset_type != "llff" or args.no_ndc:
            logger.info("Not ndc!")
            render_kwargs_train["ndc"] = False
            render_kwargs_train["lindisp"] = args.lindisp

        render_kwargs_test = {k: v for k, v in render_kwargs_train.items()}
        render_kwargs_test["perturb"] = False
        render_kwargs_test["raw_noise_std"] = 0.0

        return render_kwargs_train, render_kwargs_test, optimizer, gm

    def after_train(self):
        logger.info("Training of experiment is done.")

    def train_in_iter(self):
        H, W, _ = self.hwf

        for i in range(self.start_iter, self.max_iter):
            self.global_step = i + 1

            iter_start_time = time.time()
            batch_rays, target_s = self.sample_rays()

            #  Core optimization loop
            with self.grad_manager:
                rgb, disp, acc, extras = render(
                    H, W, self.K, chunk=self.args.chunk, rays=batch_rays,
                    retraw=True, **self.render_kwargs_train,
                )

                loss = img2mse(rgb, target_s)
                psnr = mse2psnr(loss.detach())
                if "rgb0" in extras:
                    loss += img2mse(extras["rgb0"], target_s)

                self.grad_manager.backward(loss)
                self.optimizer.step().clear_grad()

            lr = self.update_lr()
            iter_time = time.time() - iter_start_time

            self.save_ckpt()
            self.save_test()

            #  log training info
            if self.global_step % self.args.log_interval == 0:
                eta_seconds = (self.max_iter - self.global_step) * iter_time
                logger.info(
                    f"iter: {self.global_step}/{self.max_iter}, "
                    f"loss: {loss.item():.4f}, "
                    f"PSNR: {psnr.item():.3f}, "
                    f"lr: {lr:.4e}, "
                    f"iter time: {iter_time:.3f}s, "
                    f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
                )

    def update_lr(self, decay_rate=0.1):
        decay_steps = self.args.lrate_decay * 1000
        new_lr = self.args.lr * (decay_rate ** (self.global_step / decay_steps))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        return new_lr

    def save_ckpt(self):
        if self.rank == 0 and self.global_step % self.args.i_weights == 0:
            path = os.path.join(self.save_dir, f"{self.global_step:06d}.tar")
            ckpt_state = {
                "global_step": self.global_step,
                "network_fn_state_dict": self.render_kwargs_train["network_fn"].state_dict(),
                "network_fine_state_dict": self.render_kwargs_train["network_fine"].state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            mge.save(ckpt_state, path)
            logger.info(f"Save checkpoint at {path}")

    def save_test(self):
        if self.global_step % self.args.i_video == 0:
            # Turn on testing mode
            rgbs, disps = render_path(
                self.render_poses, self.hwf, self.K, self.args.chunk, self.render_kwargs_test
            )
            logger.info(f"Done, saving {rgbs.shape} {disps.shape}")
            moviebase = os.path.join(
                self.save_dir, "{}_spiral_{:06d}_".format(self.args.expname, self.global_step)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8
            )

        if self.global_step % self.args.i_testset == 0:
            i_test = self.i_split[-1]
            test_save_dir = os.path.join(self.save_dir, f"testset_{self.global_step:06d}")
            ensure_dir(test_save_dir)
            logger.info(f"test poses shape: {self.poses[i_test].shape}")
            render_path(
                mge.Tensor(self.poses[i_test]),
                self.hwf,
                self.K,
                self.args.chunk,
                self.render_kwargs_test,
                savedir=test_save_dir,
            )
            logger.info("Saved test set")

    def resume_ckpt(self, model, model_fine, optimizer):
        if self.args.ft_path is not None and self.args.ft_path != "None":
            ckpts = [self.args.ft_path]
        else:
            ckpts = [
                os.path.join(self.save_dir, f)
                for f in sorted(os.listdir(self.save_dir))
                if f.endswith("tar")
            ]

        if ckpts and not self.args.no_reload:
            logger.info(f"Found ckpts: {ckpts}")
            ckpt_to_load = ckpts[-1]
            logger.info(f"Reloading from {ckpt_to_load}")
            ckpt = mge.load(ckpt_to_load)

            self.start_iter = ckpt["global_step"]
            model.load_state_dict(ckpt["network_fn_state_dict"])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt["network_fine_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        return model, model_fine, optimizer

    def sample_rays(self):
        # Sample random ray batch
        N_rand = self.args.N_rand
        use_batching = not self.args.no_batching
        rays_rgb = self.rays_rgb
        i_train = self.i_split[0]
        images = self.images
        H, W, _ = self.hwf

        if use_batching:
            # Random over all images
            batch = rays_rgb[self.i_batch: self.i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = batch.transpose(1, 0, 2)
            batch_rays, target_s = batch[:2], batch[2]

            self.i_batch += N_rand
            if self.i_batch >= rays_rgb.shape[0]:
                logger.info("Shuffle data after an epoch!")
                rand_idx = mge.Tensor(np.random.permutation(rays_rgb.shape[0]))
                rays_rgb = rays_rgb[rand_idx]
                self.i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = mge.Tensor(target)
            pose = self.poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, self.K, pose)  # (H, W, 3), (H, W, 3)

                if self.global_step < self.args.precrop_iters:
                    dH = int(H // 2 * self.args.precrop_frac)
                    dW = int(W // 2 * self.args.precrop_frac)
                    coords = F.stack(
                        meshgrid(
                            F.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            F.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                            indexing="ij"
                        ),
                        -1,
                    )
                    if self.global_step == 1:
                        logger.info(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.args.precrop_iters}"  # noqa
                        )
                else:
                    coords = F.stack(
                        meshgrid(F.linspace(0, H - 1, H), F.linspace(0, W - 1, W), indexing="ij"),
                        -1,
                    )  # (H, W, 2)

                coords = coords.reshape(-1, 2)  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False
                )  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = F.stack([rays_o, rays_d], 0)
                target_s = target[
                    select_coords[:, 0], select_coords[:, 1]
                ]  # (N_rand, 3)

        return batch_rays, target_s
