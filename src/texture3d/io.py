"""Dataset loaders.

`load_dataset` reads the per-frame RGB-D layout produced by
create_rgbd_dataset.py (dataset.json index + <frame_id>/{rgb.jpg, depth.png,
pose.txt}), returning plain `RGBDFrame`s. Other layouts can be adapted by
constructing `RGBDFrame`s directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from texture3d.camera import PinholeCamera, RGBDFrame


def load_dataset(dataset_dir: str | Path) -> List[RGBDFrame]:
    root = Path(dataset_dir)
    index_path = root / "dataset.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No dataset.json in {root}")
    index = json.loads(index_path.read_text())

    size = index["image_size"]
    color = PinholeCamera.from_matrix(
        np.asarray(index["intrinsics"]["color_matrix_3x3"]), size["width"], size["height"]
    )
    depth_k = index["intrinsics"].get("depth_matrix_3x3")
    depth_cam = (
        PinholeCamera.from_matrix(np.asarray(depth_k), size["width"], size["height"])
        if depth_k is not None
        else color
    )
    depth_scale = float(index["depth"]["scale_to_meters"])

    frames = []
    for rec in index["frames"]:
        frames.append(
            RGBDFrame(
                pose_c2w=np.loadtxt(root / rec["pose"]).reshape(4, 4),
                camera=color,
                rgb=root / rec["rgb"],
                depth=root / rec["depth"],
                depth_scale=depth_scale,
                depth_camera=depth_cam,
                name=rec.get("frame_id", ""),
            )
        )
    return frames
