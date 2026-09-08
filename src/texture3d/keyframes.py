"""Keyframe selection.

From every window of `rgbd_frame_gap` frames, take the least blurry one
(Crete metric); drop it if its pose is too close to the previously accepted
keyframe.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from texture3d.blur import estimate_blurriness
from texture3d.camera import RGBDFrame
from texture3d.config import KeyframeConfig


def poses_close(t1: np.ndarray, t2: np.ndarray, cfg: KeyframeConfig) -> bool:
    """True when translation or relative rotation is under the threshold."""
    if np.linalg.norm(t1[:3, 3] - t2[:3, 3]) < cfg.closest_pose_translation:
        return True
    relative = t1[:3, :3].T @ t2[:3, :3]
    angles = Rotation.from_matrix(relative).as_euler("xyz")
    return bool(np.linalg.norm(angles) < cfg.closest_pose_rotation_angle)


def select_keyframes(frames: Sequence[RGBDFrame], cfg: KeyframeConfig) -> List[RGBDFrame]:
    selected: List[RGBDFrame] = []
    n = len(frames)
    gap = max(1, cfg.rgbd_frame_gap)
    for start in range(0, n, gap):
        window = frames[start : min(start + gap, n)]
        if len(window) == 1:
            best = window[0]
        else:
            best = min(window, key=lambda f: estimate_blurriness(f.load_rgb()))
        if selected and poses_close(selected[-1].pose_c2w, best.pose_c2w, cfg):
            continue
        selected.append(best)
    return selected
