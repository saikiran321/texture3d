"""Image blurriness estimation.

Crete et al., "The blur effect: perception and estimation with a new
no-reference perceptual blur metric" (SPIE 2007). The image is blurred with
a 1x9 box kernel horizontally and vertically; the loss of neighboring-pixel
variation between the original and the blurred image measures how sharp the
original was. Returns a value in [0, 1] where higher means blurrier.
"""

from __future__ import annotations

import cv2
import numpy as np


def _abs_diff_vertical(img: np.ndarray) -> np.ndarray:
    """|I(y, x) - I(y, x-1)| with a zero first column."""
    out = np.zeros_like(img)
    out[:, 1:] = np.abs(img[:, 1:] - img[:, :-1])
    return out


def _abs_diff_horizontal(img: np.ndarray) -> np.ndarray:
    """|I(y, x) - I(y-1, x)| with a zero first row."""
    out = np.zeros_like(img)
    out[1:, :] = np.abs(img[1:, :] - img[:-1, :])
    return out


def estimate_blurriness(rgb: np.ndarray) -> float:
    """Blurriness of an HxWx3 RGB uint8 image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    kernel = np.full((1, 9), 1.0 / 9, dtype=np.float32)
    b_ver = cv2.filter2D(gray, -1, kernel)  # blur along rows
    b_hor = cv2.filter2D(gray, -1, kernel.T)  # blur along columns

    d_f_ver = _abs_diff_vertical(gray)
    d_f_hor = _abs_diff_horizontal(gray)
    d_b_ver = _abs_diff_vertical(b_ver)
    d_b_hor = _abs_diff_horizontal(b_hor)

    v_ver = np.maximum(d_f_ver - d_b_ver, 0.0)
    v_hor = np.maximum(d_f_hor - d_b_hor, 0.0)

    # Sums exclude the first row and column.
    s_f_ver = float(d_f_ver[1:, 1:].sum())
    s_f_hor = float(d_f_hor[1:, 1:].sum())
    s_v_ver = float(v_ver[1:, 1:].sum())
    s_v_hor = float(v_hor[1:, 1:].sum())

    b_f_ver = (s_f_ver - s_v_ver) / s_f_ver if s_f_ver > 0 else 0.0
    b_f_hor = (s_f_hor - s_v_hor) / s_f_hor if s_f_hor > 0 else 0.0
    return max(b_f_ver, b_f_hor)
