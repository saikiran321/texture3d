"""OpenCV-based depth preprocessing before TSDF integration."""

from __future__ import annotations

import cv2
import numpy as np

from rgbd_recon.config import DepthFilterConfig


def filter_depth(depth_m: np.ndarray, cfg: DepthFilterConfig) -> np.ndarray:
    """Clean a metric depth map (float32, meters). Invalid pixels become 0.

    Steps:
      1. Range clipping: drop implausibly near/far readings.
      2. Bilateral filtering: denoise while preserving depth edges.
      3. Discontinuity masking: drop "flying pixels" at object boundaries,
         which otherwise smear into the TSDF volume.
    """
    if not cfg.enabled:
        return depth_m

    depth = depth_m.copy()
    valid = (depth > cfg.min_depth_m) & (depth < cfg.max_depth_m)
    depth[~valid] = 0.0

    if cfg.bilateral_d > 0:
        # Bilateral on meters; sigma_color given in mm for intuitive tuning.
        smoothed = cv2.bilateralFilter(
            depth,
            d=cfg.bilateral_d,
            sigmaColor=cfg.bilateral_sigma_color / 1000.0,
            sigmaSpace=cfg.bilateral_sigma_space,
        )
        depth = np.where(depth > 0, smoothed, 0.0).astype(np.float32)

    if cfg.edge_threshold_m > 0:
        k = max(3, cfg.edge_kernel | 1)  # odd kernel
        d_min = cv2.erode(depth, np.ones((k, k), np.uint8))
        d_max = cv2.dilate(depth, np.ones((k, k), np.uint8))
        jump = (d_max - d_min) > cfg.edge_threshold_m
        depth[jump & (depth > 0)] = 0.0

    return depth
