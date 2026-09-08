"""TSDF fusion of posed RGBD frames using Open3D."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from rgbd_recon.config import FusionConfig
from rgbd_recon.dataset import CameraIntrinsics, Frame


class TSDFFusion:
    """Wraps Open3D's ScalableTSDFVolume for incremental integration."""

    def __init__(self, intrinsics: CameraIntrinsics, cfg: FusionConfig):
        self.cfg = cfg
        self.intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=cfg.voxel_length_m,
            sdf_trunc=cfg.sdf_trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.num_integrated = 0

    def integrate(self, frame: Frame, rgb: np.ndarray, depth_m: np.ndarray) -> None:
        """Integrate one frame. `rgb` is HxWx3 uint8, `depth_m` float32 meters."""
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth_m)),
            depth_scale=1.0,  # depth already metric
            depth_trunc=self.cfg.depth_trunc_m,
            convert_rgb_to_intensity=False,
        )
        # Open3D expects world-to-camera extrinsics.
        self.volume.integrate(rgbd, self.intrinsics_o3d, frame.pose_w2c)
        self.num_integrated += 1

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        return self.volume.extract_triangle_mesh()

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        return self.volume.extract_point_cloud()
