"""Stage 2: which views see which faces, and how well.

Per view the mesh is rasterised into a face map with a depth test, so a pixel
belongs to the nearest face. A face's quality in that view is the blurred
Sobel gradient magnitude summed over its pixels, scaled by the squared cosine
between the view direction and the smoothed face normal: large, sharp,
head-on projections score high. The mean colour of those pixels (YCbCr over
BGR) feeds the outlier test that drops views disagreeing with the majority,
which is what removes an object passing in front of the subject in one frame.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from texture3d.mapping import imageops, raster
from texture3d.mapping.topology import Topology
from texture3d.mapping.types import NO_ID, FaceViews, Mesh, View, rgb2ycbcr


@dataclass
class ViewStats:
    quality: np.ndarray  # (F,) float32
    area: np.ndarray  # (F,) int64
    color: np.ndarray  # (F, 3) float32, YCbCr of the mean BGR


def face_map(mesh: Mesh, view: View, max_candidates: int = 1 << 24) -> Tuple[np.ndarray, np.ndarray]:
    """The view's face map and depth map: per pixel the nearest face whose
    three vertices project in front of the camera and at least three pixels
    inside the image, or NO_ID."""
    width, height = view.width, view.height
    cam = view.camera.to_camera(mesh.positions)
    z = cam[:, 2]
    in_front = z > 0
    safe = cam.copy()
    safe[~in_front, 2] = 1.0
    px = view.camera.to_image(safe).astype(np.float32)
    inside_vertex = (
        in_front & (px[:, 0] >= 3) & (px[:, 1] >= 3) & (px[:, 0] <= width - 4) & (px[:, 1] <= height - 4)
    )
    candidates = np.flatnonzero(inside_vertex[mesh.faces].all(axis=1))
    faces = mesh.faces[candidates]
    tri = px[faces]
    zf = z[faces].astype(np.float32)

    fmap = np.full(width * height, NO_ID, dtype=np.int64)
    dmap = np.zeros(width * height, dtype=np.float32)
    for ti, x, y, bary in raster.triangles_bary(tri, width, height, cull=True, max_candidates=max_candidates):
        zt = zf[ti]
        pbary = raster.perspective_correct(bary, zt)
        depth = (pbary * zt).sum(axis=1).astype(np.float32)
        pix = y * width + x
        # The nearest face per pixel within the chunk; on equal depth the
        # earlier face wins.
        order = np.lexsort((ti, depth, pix))
        pix_sorted = pix[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = pix_sorted[1:] != pix_sorted[:-1]
        sel = order[first]
        p = pix[sel]
        d = depth[sel]
        current = dmap[p]
        replace = (current == 0) | (current > d)
        dmap[p[replace]] = d[replace]
        fmap[p[replace]] = candidates[ti[sel[replace]]]
    return fmap.reshape(height, width), dmap.reshape(height, width)


def score_view(mesh: Mesh, topology: Topology, view: View, max_candidates: int = 1 << 24) -> ViewStats:
    face_count = mesh.face_count
    gradient = imageops.gradient_magnitude_blurred(imageops.gray_normalized(view)).ravel()
    mask = imageops.invalid_region_mask(view)
    fmap, _ = face_map(mesh, view, max_candidates)
    fmap = fmap.ravel()
    if mask is not None:
        # A face that touched a masked pixel is dropped so none of its
        # remaining pixels contribute.
        bad = np.unique(fmap[(mask.ravel() == 0) & (fmap != NO_ID)])
        if len(bad):
            fmap[np.isin(fmap, bad)] = NO_ID
    valid = fmap != NO_ID
    f = fmap[valid]
    area = np.bincount(f, minlength=face_count)
    quality = np.bincount(f, weights=gradient[valid], minlength=face_count).astype(np.float32)
    bgr = view.bgr.reshape(-1, 3)[valid].astype(np.float64)
    color_sum = np.stack([np.bincount(f, weights=bgr[:, c], minlength=face_count) for c in range(3)], axis=-1)

    seen = area > 0
    faces = mesh.faces[seen]
    p = mesh.positions
    centre = (p[faces[:, 0]] + p[faces[:, 1]] + p[faces[:, 2]]) / np.float32(3)
    cam_dir = view.camera.C.astype(np.float32) - centre
    n = topology.face_normals[seen]
    dot = (cam_dir * n).sum(axis=1)
    norms = np.sqrt((cam_dir * cam_dir).sum(axis=1) * (n * n).sum(axis=1))
    # A zero-area face has no normal; give it the grazing minimum rather than
    # a NaN that would poison the labelling costs.
    cosine = np.clip(np.where(norms > 0, dot / np.where(norms > 0, norms, 1), 0), -1, 1)
    cosine = np.maximum(np.float32(0.001), np.minimum(np.float32(1), cosine))
    quality[seen] *= cosine * cosine
    color = np.zeros((face_count, 3), dtype=np.float32)
    color[seen] = rgb2ycbcr((color_sum[seen] / area[seen][:, None]).astype(np.float32))
    return ViewStats(quality=quality, area=area, color=color)


def score(
    mesh: Mesh,
    topology: Topology,
    views: List[View],
    outlier_threshold: float = 0.06,
    workers: int = 4,
    log: Optional[Callable[[str], None]] = None,
) -> FaceViews:
    """Every view scored, merged per face in view order, then filtered."""
    face_count = mesh.face_count
    stats: List[Optional[ViewStats]] = [None] * len(views)

    def run(v: int) -> int:
        stats[v] = score_view(mesh, topology, views[v])
        return v

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(run, range(len(views))):
            done += 1
            if log and (done % 10 == 0 or done == len(views)):
                log(f"  scored {done}/{len(views)} views")

    face_ids, view_ids, qualities, colors = [], [], [], []
    for v, s in enumerate(stats):
        seen = np.flatnonzero(s.area > 0)
        face_ids.append(seen)
        view_ids.append(np.full(len(seen), v, dtype=np.int64))
        qualities.append(s.quality[seen])
        colors.append(s.color[seen])
    face_ids = np.concatenate(face_ids)
    view_ids = np.concatenate(view_ids)
    order = np.lexsort((view_ids, face_ids))
    offsets = np.zeros(face_count + 1, dtype=np.int64)
    np.cumsum(np.bincount(face_ids, minlength=face_count), out=offsets[1:])
    face_views = FaceViews(
        offsets=offsets,
        view=view_ids[order],
        quality=np.concatenate(qualities)[order],
        color=np.concatenate(colors)[order],
    )
    if outlier_threshold > 0:
        face_views = remove_outliers(face_views, outlier_threshold)
    return face_views


def remove_outliers(face_views: FaceViews, threshold: float) -> FaceViews:
    """Drop views whose mean face colour is an outlier of the others.

    Fit a Gaussian to the views' mean colours, drop views whose likelihood
    is below the threshold, refit, up to ten times. A face whose test
    abstains (too few views, or a degenerate covariance) keeps every view.
    """
    th = 6e-2 if threshold <= 0 else float(threshold)
    min_covariance = 1e-3
    max_iterations = 10
    min_inliers = 4
    eps = np.finfo(np.float64).eps

    counts = face_views.counts
    keep = np.ones(len(face_views.view), dtype=bool)
    for n in np.unique(counts[counts > min_inliers]):
        n = int(n)
        faces_n = np.flatnonzero(counts == n)
        slots = face_views.offsets[faces_n][:, None] + np.arange(n)[None, :]
        colors = face_views.color[slots].astype(np.float64)  # (Fn, n, 3)
        inliers = np.ones((len(faces_n), n), dtype=bool)
        active = np.ones(len(faces_n), dtype=bool)
        remove = np.zeros(len(faces_n), dtype=bool)
        for _ in range(max_iterations):
            a = np.flatnonzero(active)
            if len(a) == 0:
                break
            c = colors[a]
            w = inliers[a][..., None].astype(np.float64)
            num = inliers[a].sum(axis=1)
            mean = (c * w).sum(axis=1) / num[:, None]
            d = (c - mean[:, None, :]) * w
            cov = np.einsum("fni,fnj->fij", d, d) / (num - 1)[:, None, None]
            degenerate = np.abs(cov).reshape(len(a), -1).max(axis=1) < min_covariance
            active[a[degenerate]] = False
            remove[a[degenerate]] = True
            rest = ~degenerate
            a, cov, mean = a[rest], cov[rest], mean[rest]
            if len(a) == 0:
                continue
            # Treat the covariance as singular when the smallest eigenvalue
            # falls below 3 eps of the largest.
            eig = np.linalg.eigvalsh(cov)
            singular = eig[:, 0] <= 3 * eps * eig[:, 2]
            active[a[singular]] = False
            good = ~singular
            a, cov, mean = a[good], cov[good], mean[good]
            if len(a) == 0:
                continue
            inv = np.linalg.inv(cov)
            dd = colors[a] - mean[:, None, :]
            q = np.einsum("fni,fij,fnj->fn", dd, inv, dd)
            new_inliers = np.exp(-0.5 * q) > th
            num_new = new_inliers.sum(axis=1)
            changed = (new_inliers != inliers[a]).any(axis=1)
            inliers[a] = new_inliers
            all_in = num_new == n
            too_few = ~all_in & (num_new < min_inliers)
            stable = ~all_in & ~too_few & ~changed
            active[a[all_in | too_few | stable]] = False
            remove[a[stable]] = True
        remove[active] = True
        drop = remove[:, None] & ~inliers
        keep[slots[drop]] = False

    if keep.all():
        return face_views
    face_of_slot = np.repeat(np.arange(len(counts)), counts)[keep]
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(np.bincount(face_of_slot, minlength=len(counts)), out=offsets[1:])
    return FaceViews(
        offsets=offsets, view=face_views.view[keep], quality=face_views.quality[keep], color=face_views.color[keep]
    )
