#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import megengine.module as nn
import megengine.functional as F


class NeRF(nn.Module):
    """NeRF module"""
    def __init__(
        self, depth=8, width=256, input_ch=3, input_ch_views=3,
        output_ch=4, skips=[4], use_viewdirs=False,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = [nn.Linear(input_ch, width)] + [
            nn.Linear(width, width) if i not in self.skips else nn.Linear(width + input_ch, width)
            for i in range(depth - 1)
        ]

        if use_viewdirs:
            self.alpha_linear = nn.Linear(width, 1)
            self.feature_linear = nn.Linear(width, width)
            self.views_linears = nn.Linear(input_ch_views + width, width // 2)
            self.rgb_linear = nn.Linear(width // 2, 3)
        else:
            self.output_linear = nn.Linear(width, output_ch)

    def forward(self, x):
        input_pts, input_views = F.split(x, [self.input_ch], axis=-1)
        h = input_pts

        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = F.concat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = F.concat([feature, input_views], -1)
            h = F.relu(self.views_linears(h))
            rgb = self.rgb_linear(h)
            outputs = F.concat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
