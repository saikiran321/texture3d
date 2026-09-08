"""Dataset access for the per-frame RGBD layout produced by create_rgbd_dataset.py.

Expected structure:

    <dataset_dir>/
        dataset.json            # index with frames, intrinsics, depth scale
        intrinsics.txt          # shared 3x3 color intrinsics
        <frame_id>/
            rgb.jpg
            depth.png           # uint16, metric = value * depth_scale
            pose.txt            # 4x4 camera-to-world
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class Frame:
    """One RGBD frame with pose. Images are loaded lazily via the loaders below."""

    frame_id: str
    index: int
    rgb_path: Path
    depth_path: Path
    pose_c2w: np.ndarray  # 4x4 camera-to-world
    depth_scale_to_m: float

    @property
    def pose_w2c(self) -> np.ndarray:
        return np.linalg.inv(self.pose_c2w)

    def load_rgb(self) -> np.ndarray:
        """RGB image as HxWx3 uint8 (RGB order)."""
        bgr = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read RGB image {self.rgb_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_depth_raw(self) -> np.ndarray:
        """Depth image as HxW uint16 in sensor units."""
        depth = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Failed to read depth image {self.depth_path}")
        if depth.dtype != np.uint16:
            raise ValueError(f"Expected uint16 depth, got {depth.dtype} in {self.depth_path}")
        return depth

    def load_depth_m(self) -> np.ndarray:
        """Depth image as HxW float32 in meters."""
        return self.load_depth_raw().astype(np.float32) * self.depth_scale_to_m


class RGBDSequence:
    """Reads a subset directory (e.g. apt0_seq0) produced by create_rgbd_dataset.py."""

    def __init__(self, dataset_dir: Path):
        self.root = Path(dataset_dir)
        index_path = self.root / "dataset.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No dataset.json in {self.root}")
        self.index = json.loads(index_path.read_text())

        size = self.index["image_size"]
        k = np.asarray(self.index["intrinsics"]["color_matrix_3x3"], dtype=np.float64)
        self.intrinsics = CameraIntrinsics(
            width=int(size["width"]),
            height=int(size["height"]),
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
        )
        kd = np.asarray(
            self.index["intrinsics"].get("depth_matrix_3x3", k), dtype=np.float64
        )
        # Depth images share the color resolution in this layout.
        self.depth_intrinsics = CameraIntrinsics(
            width=int(size["width"]),
            height=int(size["height"]),
            fx=float(kd[0, 0]),
            fy=float(kd[1, 1]),
            cx=float(kd[0, 2]),
            cy=float(kd[1, 2]),
        )
        self.depth_scale_to_m = float(self.index["depth"]["scale_to_meters"])
        self.name = self.index.get("name", self.root.name)

        self._frames = [self._build_frame(i, rec) for i, rec in enumerate(self.index["frames"])]

    def _build_frame(self, index: int, record: dict) -> Frame:
        pose = np.loadtxt(self.root / record["pose"]).reshape(4, 4)
        return Frame(
            frame_id=record["frame_id"],
            index=index,
            rgb_path=self.root / record["rgb"],
            depth_path=self.root / record["depth"],
            pose_c2w=pose,
            depth_scale_to_m=self.depth_scale_to_m,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, i: int) -> Frame:
        return self._frames[i]

    def frames(self, stride: int = 1, max_frames: Optional[int] = None) -> Iterator[Frame]:
        selected = self._frames[:: max(1, stride)]
        if max_frames:
            selected = selected[:max_frames]
        yield from selected

    def trajectory_c2w(self) -> np.ndarray:
        """(N, 4, 4) stack of camera-to-world poses."""
        return np.stack([f.pose_c2w for f in self._frames])
