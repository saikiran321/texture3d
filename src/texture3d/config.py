"""Bake configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class KeyframeConfig:
    """Keyframe selection: the least blurry frame of every window, skipping
    near-duplicate poses."""

    # 1 keyframe per window of this many frames (the least blurry). 1 = use all.
    rgbd_frame_gap: int = 5
    # Candidates whose pose is close to the previous keyframe are skipped.
    closest_pose_translation: float = 0.05  # meters
    closest_pose_rotation_angle: float = 0.09  # radians


@dataclass
class MappingConfig:
    """The one-keyframe-per-face mapping (`texture3d.mapping`)."""

    # Views whose mean face colour falls below this Gaussian likelihood against
    # the other views' colours are dropped for that face. 0 disables the test.
    outlier_threshold: float = 0.06
    # Data cost scale relative to the Potts smoothness cost. Lower makes
    # larger, fewer patches.
    data_smoothness_ratio: float = 0.1
    global_seam_leveling: bool = True
    local_seam_leveling: bool = True
    # Width in pixels of the border strip the local leveling blends.
    local_leveling_strip_width: int = 20
    # Unsharp-mask weight applied to the finished atlas. 0 disables it.
    sharpness_weight: float = 0.5
    # The atlas is the smallest power of two that packs the patches, up to this.
    max_texture_size: int = 8192
    # Colour of atlas texels no patch covers (RGB).
    empty_color: tuple = (255, 255, 255)
    # Views scored at once; each holds a face map and a depth map while scored.
    scoring_workers: int = 4
    # Give a face no keyframe rasterised (smaller than a pixel in every view)
    # the view of a textured neighbour instead of leaving it the empty colour.
    fill_unseen: bool = True
    # Patch count up to which the atlas is packed with MaxRects (quadratic in
    # the patch count); beyond it a shelf packer runs. 0 = always MaxRects.
    exact_packing_limit: int = 4000


@dataclass
class BakeConfig:
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
