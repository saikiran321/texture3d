"""Camera and posed-frame primitives.

The library is dataset-agnostic: callers hand `bake()` a sequence of
`RGBDFrame`s, each holding (or lazily pointing at) an RGB image, a metric
depth image, a camera-to-world pose, and pinhole intrinsics. Convenience
loaders for on-disk layouts live in `texture3d.io`.

Conventions:
- camera looks down +z, x right, y down (OpenCV pinhole);
- poses are 4x4 camera-to-world;
- RGB images are HxWx3 uint8 in RGB channel order;
- depth images are HxW float32 meters (0 = invalid).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

ImageSource = Union[np.ndarray, str, Path]


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, k: np.ndarray, width: int, height: int) -> PinholeCamera:
        k = np.asarray(k, dtype=np.float64)
        return cls(
            width=int(width), height=int(height),
            fx=float(k[0, 0]), fy=float(k[1, 1]),
            cx=float(k[0, 2]), cy=float(k[1, 2]),
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) camera-space points. Returns ((N, 2) pixels, (N,) z)."""
        z = points_cam[:, 2]
        zsafe = np.where(z != 0, z, 1.0)
        u = points_cam[:, 0] / zsafe * self.fx + self.cx
        v = points_cam[:, 1] / zsafe * self.fy + self.cy
        return np.stack([u, v], axis=-1), z


@dataclass
class RGBDFrame:
    """One posed RGB-D frame. Images may be in-memory arrays or file paths.

    `depth` semantics: float arrays are metric meters already; integer arrays
    and integer image files are multiplied by `depth_scale` (meters per unit,
    e.g. 1e-3 for millimeter PNGs). Zeros mean invalid.
    """

    pose_c2w: np.ndarray  # (4, 4)
    camera: PinholeCamera
    rgb: ImageSource
    depth: ImageSource
    depth_scale: float = 1.0e-3
    depth_camera: Optional[PinholeCamera] = None
    name: str = ""

    def __post_init__(self):
        self.pose_c2w = np.asarray(self.pose_c2w, dtype=np.float64)
        if self.pose_c2w.shape != (4, 4):
            raise ValueError(f"pose_c2w must be 4x4, got {self.pose_c2w.shape}")
        if self.depth_camera is None:
            self.depth_camera = self.camera

    @property
    def pose_w2c(self) -> np.ndarray:
        return np.linalg.inv(self.pose_c2w)

    def load_rgb(self) -> np.ndarray:
        """HxWx3 uint8, RGB order."""
        if isinstance(self.rgb, np.ndarray):
            return self.rgb
        bgr = cv2.imread(str(self.rgb), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Failed to read RGB image {self.rgb}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_depth_m(self) -> np.ndarray:
        """HxW float32 meters, 0 = invalid."""
        if isinstance(self.depth, np.ndarray):
            arr = self.depth
        else:
            arr = cv2.imread(str(self.depth), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise OSError(f"Failed to read depth image {self.depth}")
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32)
        return arr.astype(np.float32) * self.depth_scale
