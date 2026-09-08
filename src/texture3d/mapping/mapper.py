"""The whole pipeline, in stage order."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from texture3d.mapping import atlas as atlas_stage
from texture3d.mapping import labeling, leveling, patches, scoring, topology
from texture3d.mapping.types import EmptyMesh, Mesh, NoViews, NoVisibleFace, Result, View


@dataclass
class Tuning:
    """Tunable arguments of the mapping pipeline."""

    #: Views whose mean face colour falls below this Gaussian likelihood
    #: against the other views' colours are dropped for that face. 0 disables.
    outlier_threshold: float = 0.06
    #: Data cost scale relative to the Potts smoothness cost. Lower makes
    #: larger, fewer patches.
    data_smoothness_ratio: float = 0.1
    global_seam_leveling: bool = True
    local_seam_leveling: bool = True
    #: 0 packs into a power-of-two atlas.
    texture_size_multiple: int = 0
    #: Colour of atlas texels no patch covers, BGR.
    empty_color_bgr: Tuple[int, int, int] = (39, 127, 255)
    #: Unsharp-mask weight applied to the finished atlas. 0 disables it.
    sharpness_weight: float = 0.5
    max_texture_size: int = 8192
    #: Width in pixels of the border strip the local leveling blends.
    local_leveling_strip_width: int = 20
    #: Views scored at once; a memory budget, each holds its face and depth maps.
    scoring_workers: int = 4
    #: Give a face no view rasterised (sub-pixel in every keyframe) the view of
    #: a textured neighbour instead of the empty colour. See ``labeling.fill_unseen``.
    fill_unseen: bool = True
    #: Patch count up to which the atlas is packed with MaxRects; beyond it a
    #: shelf packer runs, since MaxRects grows quadratically with the patch
    #: count. 0 = always MaxRects.
    exact_packing_limit: int = 4000


def run(mesh: Mesh, views: List[View], tuning: Optional[Tuning] = None, log: Optional[Callable[[str], None]] = None) -> Result:
    tuning = tuning or Tuning()
    if mesh.face_count == 0 or mesh.vertex_count == 0:
        raise EmptyMesh("the mesh has no faces")
    if not views:
        raise NoViews("no views to texture from")
    timings = {}
    start = time.time()
    phase = start

    def lap(name: str) -> None:
        nonlocal phase
        now = time.time()
        timings[name] = round(now - phase, 2)
        phase = now

    topo = topology.build(mesh)
    lap("topology")

    face_views = scoring.score(mesh, topo, views, tuning.outlier_threshold, tuning.scoring_workers, log)
    if not face_views.visible.any():
        raise NoVisibleFace("no view sees any face")
    lap("scoring")
    if log:
        log(f"  {int(face_views.visible.sum())}/{mesh.face_count} faces seen, {len(face_views.view)} face-view pairs")

    outcome = labeling.assign(mesh, topo, face_views, tuning.data_smoothness_ratio, log, views, tuning.fill_unseen)
    lap("labeling")
    if log:
        log(f"  labelled in {outcome.iterations} sweeps, energy {outcome.energy:.0f}")

    patch_set = patches.build(mesh, topo, face_views, outcome, views)
    seams = patches.create_seam_vertices(mesh, patch_set)
    lap("patches")
    if log:
        log(f"  {patch_set.patch_count} patches, {seams.count} seam vertices")

    images = [view.bgr.copy() for view in views]
    if patch_set.patch_count > 1:
        if tuning.global_seam_leveling:
            leveling.global_leveling(mesh, topo, patch_set, seams, views, images, log)
        lap("global_leveling")
        if tuning.local_seam_leveling:
            leveling.local_leveling(mesh, patch_set, seams, views, images, tuning.local_leveling_strip_width, log)
        lap("local_leveling")

    patches.merge_contained(patch_set)
    out = atlas_stage.generate(
        patch_set,
        images,
        views,
        texture_size_multiple=tuning.texture_size_multiple,
        max_texture_size=tuning.max_texture_size,
        empty_color_bgr=tuning.empty_color_bgr,
        sharpness_weight=tuning.sharpness_weight,
        exact_packing_limit=tuning.exact_packing_limit,
    )
    lap("atlas")
    timings["total"] = round(time.time() - start, 2)
    return Result(
        labels=outcome.labels,
        face_patch=patch_set.face_patch,
        patch_count=patch_set.patch_count,
        atlas_bgr=out.pixels,
        atlas_size=out.size,
        face_texcoords=out.face_texcoords,
        patch_rects=out.patch_rects,
        timings=timings,
    )
