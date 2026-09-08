"""Configuration for the RGBD reconstruction pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DepthFilterConfig:
    """OpenCV-based depth preprocessing options."""

    enabled: bool = True
    # Bilateral filter smooths depth while keeping geometric edges.
    bilateral_d: int = 5
    bilateral_sigma_color: float = 30.0  # in depth units (mm)
    bilateral_sigma_space: float = 5.0
    # Discard depth outside this metric range (meters).
    min_depth_m: float = 0.3
    max_depth_m: float = 4.0
    # Invalidate pixels on strong depth discontinuities (flying pixels).
    edge_kernel: int = 3
    edge_threshold_m: float = 0.15


@dataclass
class FusionConfig:
    """TSDF volume parameters."""

    voxel_length_m: float = 0.01
    sdf_trunc_m: float = 0.04
    depth_trunc_m: float = 4.0
    color_type: str = "rgb8"


@dataclass
class MeshConfig:
    """Mesh post-processing options."""

    simplify_target_triangles: int = 0  # 0 disables simplification
    remove_small_clusters: bool = True
    min_cluster_triangles: int = 500
    compute_normals: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    dataset_dir: Path = Path(".")
    output_dir: Path = Path("output")
    frame_stride: int = 1  # use every Nth frame
    max_frames: int = 0  # 0 means all frames
    save_point_cloud: bool = True
    point_cloud_voxel_m: float = 0.01  # downsample size for exported cloud, 0 = raw
    save_trajectory: bool = True
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)

    def to_json(self) -> str:
        data = asdict(self)
        data["dataset_dir"] = str(self.dataset_dir)
        data["output_dir"] = str(self.output_dir)
        return json.dumps(data, indent=2)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n")
