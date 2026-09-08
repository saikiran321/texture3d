"""The image arithmetic the stages share."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from texture3d.mapping.types import View


def gray_normalized(view: View) -> np.ndarray:
    """Grey in 0...1: ``0.114 B + 0.587 G + 0.299 R`` over normalised bytes."""
    bgr = view.bgr.astype(np.float32) * np.float32(1.0 / 255.0)
    return bgr[..., 0] * np.float32(0.114) + bgr[..., 1] * np.float32(0.587) + bgr[..., 2] * np.float32(0.299)


def gradient_magnitude_blurred(gray: np.ndarray) -> np.ndarray:
    """Sobel 3x3 gradient magnitude at scale 1/8, then a 15x15 Gaussian blur."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REFLECT_101)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.GaussianBlur(magnitude, (15, 15), 0, borderType=cv2.BORDER_REFLECT_101)


def invalid_region_mask(view: View) -> Optional[np.ndarray]:
    """Black regions reached by flood fill from eight border seeds.

    Those regions are the padding left by lens undistortion and must not
    texture anything. None when no seed is black.
    """
    gray = cv2.cvtColor(view.bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    seeds = [(0, 0), (0, h // 2), (0, h - 1), (w // 2, h - 1), (w - 1, h - 1), (w - 1, h // 2), (w - 1, 0), (w // 2, 0)]
    if not any(gray[y, x] == 0 for x, y in seeds):
        return None
    filled = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    for x, y in seeds:
        if gray[y, x] == 0 and filled[y + 1, x + 1] == 0:
            cv2.floodFill(gray.copy(), filled, (x, y), 255, 0, 3, flags)
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[filled[1:-1, 1:-1] != 0] = 0
    return mask


def sample_linear(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample between the four pixels around ``floor(pt)``.

    Integer coordinates are pixel centres; pixels outside the image
    contribute nothing. ``image`` is (H, W, 3) of any numeric type; the
    result is float32.
    """
    h, w = image.shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    gx = np.floor(pts[:, 0]).astype(np.int64)
    gy = np.floor(pts[:, 1]).astype(np.int64)
    dx = pts[:, 0] - gx.astype(np.float32)
    dy = pts[:, 1] - gy.astype(np.float32)
    result = np.zeros((len(pts), 3), dtype=np.float32)
    for i, wy in ((0, 1 - dy), (1, dy)):
        y = gy + i
        y_ok = (y >= 0) & (y < h)
        for j, wx in ((0, 1 - dx), (1, dx)):
            x = gx + j
            ok = y_ok & (x >= 0) & (x < w)
            weight = (wx * wy)[ok]
            result[ok] += image[y[ok], x[ok]].astype(np.float32) * weight[:, None]
    return result


def gaussian_blur_bgr8(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def sharpen(image: np.ndarray, sigma: float, weight: float) -> np.ndarray:
    """Unsharp mask: ``img (1 + w) - blur(img) w``."""
    blurred = gaussian_blur_bgr8(image, sigma)
    return cv2.addWeighted(image, 1.0 + weight, blurred, -weight, 0.0)
