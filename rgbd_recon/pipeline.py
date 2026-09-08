"""End-to-end reconstruction pipeline: load -> preprocess -> fuse -> export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from rgbd_recon.config import PipelineConfig
from rgbd_recon.dataset import RGBDSequence
from rgbd_recon.fusion import TSDFFusion
from rgbd_recon.preprocessing import filter_depth


class ReconstructionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.sequence = RGBDSequence(cfg.dataset_dir)

    def run(self) -> dict:
        cfg = self.cfg
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg.save(out / "config.json")

        fusion = TSDFFusion(self.sequence.intrinsics, cfg.fusion)
        max_frames = cfg.max_frames or None
        frames = list(self.sequence.frames(stride=cfg.frame_stride, max_frames=max_frames))

        print(
            f"[{self.sequence.name}] integrating {len(frames)} frames "
            f"(stride={cfg.frame_stride}, voxel={cfg.fusion.voxel_length_m*100:.1f}cm)"
        )
        t0 = time.time()
        for i, frame in enumerate(frames):
            rgb = frame.load_rgb()
            depth_m = filter_depth(frame.load_depth_m(), cfg.depth_filter)
            fusion.integrate(frame, rgb, depth_m)
            if (i + 1) % 100 == 0 or (i + 1) == len(frames):
                rate = (i + 1) / (time.time() - t0)
                print(f"  integrated {i + 1}/{len(frames)} ({rate:.1f} fps)")
        integrate_s = time.time() - t0

        mesh = fusion.extract_mesh()
        mesh = self._postprocess_mesh(mesh)
        mesh_path = out / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"  mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris -> {mesh_path}")

        pcd_path = None
        if cfg.save_point_cloud:
            pcd = fusion.extract_point_cloud()
            if cfg.point_cloud_voxel_m > 0:
                pcd = pcd.voxel_down_sample(cfg.point_cloud_voxel_m)
            pcd_path = out / "point_cloud.ply"
            o3d.io.write_point_cloud(str(pcd_path), pcd)
            print(f"  point cloud: {len(pcd.points)} points -> {pcd_path}")

        traj_path = None
        if cfg.save_trajectory:
            traj_path = out / "trajectory_tum.txt"
            self._save_trajectory_tum(frames, traj_path)

        summary = {
            "sequence": self.sequence.name,
            "dataset_dir": str(cfg.dataset_dir),
            "num_frames_integrated": fusion.num_integrated,
            "frame_stride": cfg.frame_stride,
            "integrate_seconds": round(integrate_s, 1),
            "voxel_length_m": cfg.fusion.voxel_length_m,
            "mesh": {
                "path": str(mesh_path),
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.triangles),
            },
            "point_cloud": str(pcd_path) if pcd_path else None,
            "trajectory": str(traj_path) if traj_path else None,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def _postprocess_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        mcfg = self.cfg.mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        if mcfg.remove_small_clusters and len(mesh.triangles) > 0:
            cluster_ids, cluster_sizes, _ = map(
                np.asarray, mesh.cluster_connected_triangles()
            )
            small = cluster_sizes[cluster_ids] < mcfg.min_cluster_triangles
            mesh.remove_triangles_by_mask(small)
            mesh.remove_unreferenced_vertices()

        if mcfg.simplify_target_triangles > 0 and len(mesh.triangles) > mcfg.simplify_target_triangles:
            mesh = mesh.simplify_quadric_decimation(mcfg.simplify_target_triangles)

        if mcfg.compute_normals:
            mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def _save_trajectory_tum(frames, path: Path) -> None:
        """TUM format: timestamp tx ty tz qx qy qz qw (camera-to-world)."""
        lines = []
        for frame in frames:
            pose = frame.pose_c2w
            t = pose[:3, 3]
            q = _rotation_to_quaternion(pose[:3, :3])
            lines.append(
                f"{frame.index} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
            )
        path.write_text("\n".join(lines) + "\n")
        print(f"  trajectory ({len(lines)} poses) -> {path}")


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (qx, qy, qz, qw)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q = np.empty(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
        qx, qy, qz, qw = q
    return np.array([qx, qy, qz, qw])
