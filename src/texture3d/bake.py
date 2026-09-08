"""End-to-end texture baking: mesh + posed RGB-D frames -> textured trimesh.

  keyframes -> views -> texture mapping (score, label, patch, level, pack)
  -> textured trimesh

The mapping (`texture3d.mapping`) textures each face from one keyframe, cuts
patches of same-keyframe faces out of that keyframe, levels the seams and
packs the patches into one atlas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import cv2
import numpy as np
import trimesh

from texture3d import mapping
from texture3d.camera import RGBDFrame
from texture3d.config import BakeConfig
from texture3d.keyframes import select_keyframes

MeshLike = Union[trimesh.Trimesh, str, Path]


@dataclass
class BakeResult:
    mesh: trimesh.Trimesh  # textured mesh (TextureVisuals with baked image)
    texture: np.ndarray  # (S, S, 3) uint8 RGB
    stats: dict

    def export(self, path: Union[str, Path]) -> Path:
        """Write the textured mesh (format from extension: .obj, .glb, ...)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.mesh.export(path)
        return path


def _load_mesh(mesh: MeshLike) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    # process=False preserves vertex/face order and never merges vertices.
    loaded = trimesh.load(mesh, process=False, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"Not a triangle mesh: {mesh}")
    return loaded


def _material(texture: np.ndarray):
    from PIL import Image

    # Ka/Kd white + Ks black: viewers multiply the texture by Kd, and
    # trimesh's default 0.4 grey would darken everything.
    return trimesh.visual.material.SimpleMaterial(
        image=Image.fromarray(texture),
        name="texture3d",
        ambient=[255, 255, 255, 255],
        diffuse=[255, 255, 255, 255],
        specular=[0, 0, 0, 255],
    )


def _views(keyframes: List[RGBDFrame]) -> List[mapping.View]:
    """The mapping's views: BGR pixels and a camera with R world-to-camera and
    C the centre in world coordinates, from each frame's camera-to-world pose."""
    views = []
    for frame in keyframes:
        w2c = frame.pose_w2c
        cam = frame.camera
        camera = mapping.Camera(
            fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, R=w2c[:3, :3].copy(), C=frame.pose_c2w[:3, 3].copy()
        )
        views.append(mapping.View(bgr=cv2.cvtColor(frame.load_rgb(), cv2.COLOR_RGB2BGR), camera=camera))
    return views


def _output_mesh(mesh: trimesh.Trimesh, result: mapping.Result, texture: np.ndarray) -> trimesh.Trimesh:
    """The textured result as a trimesh with per-vertex UVs. A vertex has one
    texture coordinate per patch it belongs to (its projection into that
    patch's view), so vertices are split along patch borders."""
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    size = result.atlas_size
    corner_vertex = faces.ravel()
    corner_patch = np.repeat(result.face_patch, 3)
    key = corner_vertex * (result.patch_count + 1) + corner_patch
    unique_key, new_index = np.unique(key, return_inverse=True)
    new_vertices = vertices[unique_key // (result.patch_count + 1)]
    # Pixel centres to OBJ UVs: u right, v up from the bottom-left.
    tc = result.face_texcoords.reshape(-1, 2).astype(np.float64)
    uv = np.empty((len(unique_key), 2), dtype=np.float64)
    uv[new_index] = np.stack([(tc[:, 0] + 0.5) / size, 1.0 - (tc[:, 1] + 0.5) / size], axis=-1)
    visual = trimesh.visual.TextureVisuals(uv=uv, material=_material(texture))
    return trimesh.Trimesh(vertices=new_vertices, faces=new_index.reshape(-1, 3), visual=visual, process=False)


def bake(
    mesh: MeshLike,
    frames: Iterable[RGBDFrame],
    config: Optional[BakeConfig] = None,
    verbose: bool = True,
) -> BakeResult:
    """Texture `mesh` from posed RGB-D `frames`: every face from one keyframe,
    seams levelled. Any UVs the mesh carries are ignored, since the mapping's
    UVs follow the labelling. Returns a `BakeResult` whose `.mesh` is a
    textured `trimesh.Trimesh` ready to `.export()`."""
    cfg = config or BakeConfig()
    t0 = time.time()
    mesh = _load_mesh(mesh)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    log(f"mesh: {len(vertices)} verts, {len(faces)} faces")
    frames = list(frames)
    keyframes = select_keyframes(frames, cfg.keyframes)
    log(f"keyframes: {len(keyframes)} / {len(frames)} frames")
    views = _views(keyframes)
    m = cfg.mapping
    empty = m.empty_color
    tuning = mapping.Tuning(
        outlier_threshold=m.outlier_threshold,
        data_smoothness_ratio=m.data_smoothness_ratio,
        global_seam_leveling=m.global_seam_leveling,
        local_seam_leveling=m.local_seam_leveling,
        empty_color_bgr=(int(empty[2]), int(empty[1]), int(empty[0])),
        sharpness_weight=m.sharpness_weight,
        max_texture_size=m.max_texture_size,
        local_leveling_strip_width=m.local_leveling_strip_width,
        scoring_workers=m.scoring_workers,
        exact_packing_limit=m.exact_packing_limit,
        fill_unseen=m.fill_unseen,
    )
    log("mapping: scoring views, labelling faces, levelling seams, packing...")
    result = mapping.run(mapping.Mesh(positions=vertices, faces=faces), views, tuning, log)
    texture = cv2.cvtColor(result.atlas_bgr, cv2.COLOR_BGR2RGB)
    out_mesh = _output_mesh(mesh, result, texture)
    textured = int((result.labels != mapping.types.NO_ID).sum())
    stats = {
        "num_frames": len(frames),
        "num_keyframes": len(keyframes),
        "num_patches": result.patch_count,
        "texture_size": result.atlas_size,
        "num_faces": int(len(faces)),
        "textured_faces": textured,
        "textured_fraction": round(textured / max(1, len(faces)), 4),
        "output_vertices": int(len(out_mesh.vertices)),
        "stage_seconds": result.timings,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    return BakeResult(mesh=out_mesh, texture=texture, stats=stats)
