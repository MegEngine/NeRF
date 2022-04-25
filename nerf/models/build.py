#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine.functional as F

from .nerf import NeRF
from .embed import get_embedder

__all__ = ["create_nerf"]


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return F.concat([fn(inputs[i: i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = inputs.reshape([-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = F.broadcast_to(F.expand_dims(viewdirs, axis=1), inputs.shape)
        input_dirs_flat = input_dirs.reshape([-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = F.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = outputs_flat.reshape(list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        depth=args.netdepth,
        width=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    )

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            depth=args.netdepth_fine,
            width=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        )

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(
            inputs, viewdirs, network_fn, embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn, netchunk=args.netchunk,
        )

    return model, model_fine, network_query_fn
