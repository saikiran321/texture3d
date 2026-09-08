"""Stage 5: hide the seams between patches by editing the view images the
patches are cut from, in place, so the atlas assembled afterwards already
agrees across patch borders.

Global leveling solves one colour offset per (vertex, patch): along every seam
the two patches' colours should agree after the offset, and adjacent vertices
in a patch should have similar offsets. The offsets are interpolated over each
patch and added in YCbCr.

Local leveling then takes a strip along each patch's border, sets the border
pixels to the mean of the two sides, and solves a Poisson problem in the strip
so the gradients of the original are kept while the border values are met.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from texture3d.mapping import imageops, raster
from texture3d.mapping.patches import PatchSet, SeamData
from texture3d.mapping.topology import Topology
from texture3d.mapping.types import Mesh, View, clamp_byte, rgb2ycbcr, round2int, ycbcr2rgb

MASK_EMPTY = 0
MASK_BORDER = 128
MASK_INTERIOR = 255
TIKHONOV_LAMBDA = np.float32(0.1)


def conjugate_gradient(A: sparse.csr_matrix, b: np.ndarray, tolerance: float, max_iterations: int) -> np.ndarray:
    """Jacobi-preconditioned conjugate gradient from a zero start, stopping
    when the residual falls below ``tolerance`` relative to the right-hand
    side or after ``max_iterations``."""
    b = np.asarray(b, dtype=np.float64)
    if not np.any(b):
        return np.zeros_like(b)
    diag = A.diagonal()
    inv_diag = np.where(diag != 0, 1.0 / np.where(diag != 0, diag, 1.0), 1.0)
    M = sparse.diags(inv_diag)
    try:
        x, _ = cg(A, b, rtol=tolerance, atol=0.0, maxiter=max_iterations, M=M)
    except TypeError:  # scipy before 1.12 spelt the relative tolerance ``tol``
        x, _ = cg(A, b, tol=tolerance, atol=0.0, maxiter=max_iterations, M=M)
    return x


def _assign_last(target: np.ndarray, x: np.ndarray, y: np.ndarray, values: np.ndarray) -> None:
    """``target[y, x] = values`` in sequence, later entries overwriting earlier
    ones, as a sequential rasteriser would."""
    if len(x) == 0:
        return
    pix = y * target.shape[1] + x
    _, first_from_end = np.unique(pix[::-1], return_index=True)
    sel = len(pix) - 1 - first_from_end
    target[y[sel], x[sel]] = values[sel]


def _lookup_table(keys: np.ndarray):
    """A vectorised ``dict`` over integer keys: returns a function mapping
    query keys to their position in ``keys`` or -1."""
    order = np.argsort(keys)
    sorted_keys = keys[order]

    def find(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.int64)
        pos = np.searchsorted(sorted_keys, query)
        pos_c = np.minimum(pos, max(len(sorted_keys) - 1, 0))
        found = (len(sorted_keys) > 0) & (sorted_keys[pos_c] == query)
        return np.where(found, order[pos_c], -1)

    return find


# ── Global ──────────────────────────────────────────────────────────────────


def _vertex_colors(patch_set: PatchSet, seams: SeamData, views: List[View], images: List[np.ndarray]) -> np.ndarray:
    """Per seam-vertex entry, the accumulated colour along leaving seam edges.

    Samples along each seam edge leaving the vertex in the entry's view,
    weighted towards the vertex, then the edges weighted by length.
    """
    entry_count = len(seams.entry_patch)
    colors = np.zeros((entry_count, 3), dtype=np.float32)
    if entry_count == 0 or len(seams.edge_other) == 0:
        return colors
    num_patches = patch_set.patch_count
    entry_key = seams.entry_sv * (num_patches + 1) + seams.entry_patch
    find_entry = _lookup_table(entry_key)
    q_entry = np.repeat(np.arange(entry_count), np.diff(seams.edge_offsets))
    q_patch = seams.entry_patch[q_entry]
    other = find_entry(seams.edge_other * (num_patches + 1) + q_patch)
    valid = other >= 0
    p0 = seams.entry_proj[q_entry]
    p1 = seams.entry_proj[np.maximum(other, 0)]
    d = p1 - p0
    length = np.sqrt((d * d).sum(axis=1)).astype(np.float32)
    valid &= length > 0
    q = np.flatnonzero(valid)
    if len(q) == 0:
        return colors
    samples = round2int(np.maximum(length[q], 1) * 2) - 1
    rep = np.repeat(q, samples)
    local = np.arange(int(samples.sum())) - np.repeat(np.cumsum(samples) - samples, samples)
    frac = (local.astype(np.float32) / samples[np.searchsorted(q, rep)].astype(np.float32)).astype(np.float32)
    pos = p0[rep] + d[rep] * frac[:, None]
    weight = np.float32(1) - frac
    labels = np.array([p.label for p in patch_set.patches[:num_patches]], dtype=np.int64)
    view_of_sample = labels[q_patch[rep]]
    sampled = np.zeros((len(rep), 3), dtype=np.float32)
    for v in np.unique(view_of_sample):
        m = view_of_sample == v
        sampled[m] = rgb2ycbcr(imageops.sample_linear(images[int(v)], pos[m]))
    edge_sum = np.stack([np.bincount(rep, weights=sampled[:, c] * weight, minlength=len(q_entry)) for c in range(3)], -1)
    edge_weight = np.bincount(rep, weights=weight, minlength=len(q_entry))
    edge_color = np.where(edge_weight[:, None] != 0, edge_sum / np.where(edge_weight[:, None] != 0, edge_weight[:, None], 1), 0)
    entry_sum = np.stack(
        [np.bincount(q_entry[q], weights=edge_color[q, c] * length[q], minlength=entry_count) for c in range(3)], -1
    )
    entry_weight = np.bincount(q_entry[q], weights=length[q], minlength=entry_count)
    nonzero = entry_weight != 0
    colors[nonzero] = (entry_sum[nonzero] / entry_weight[nonzero][:, None]).astype(np.float32)
    return colors


def dilate_mean(adjust: np.ndarray) -> np.ndarray:
    """An interior zero pixel takes the mean of its non-zero 8-neighbours."""
    h, w = adjust.shape[:2]
    out = adjust.copy()
    if w < 3 or h < 3:
        return out
    nonzero = np.any(adjust != 0, axis=2)
    weight = nonzero.astype(np.float32)
    padded = np.pad(adjust, ((1, 1), (1, 1), (0, 0)))
    padded_w = np.pad(weight, 1)
    total = np.zeros_like(adjust)
    count = np.zeros((h, w), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            total += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            count += padded_w[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    fill = ~nonzero & (count > 0)
    fill[0, :] = fill[-1, :] = False
    fill[:, 0] = fill[:, -1] = False
    out[fill] = total[fill] / count[fill][:, None]
    return out


def global_leveling(
    mesh: Mesh,
    topology: Topology,
    patch_set: PatchSet,
    seams: SeamData,
    views: List[View],
    images: List[np.ndarray],
    log: Optional[Callable[[str], None]] = None,
) -> None:
    if seams.count == 0:
        return
    num_patches = patch_set.patch_count
    vertex_count = mesh.vertex_count
    faces = mesh.faces
    face_patch = patch_set.face_patch

    # Which patch each vertex is in (the last face containing it decides),
    # or which seam vertex it is.
    flat = faces.ravel()
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    last = np.searchsorted(flat_sorted, np.arange(vertex_count), side="right") - 1
    first = np.searchsorted(flat_sorted, np.arange(vertex_count), side="left")
    has_face = first <= last
    vertex_patch = np.full(vertex_count, num_patches, dtype=np.int64)
    vertex_patch[has_face] = face_patch[order[last[has_face]] // 3]
    is_seam = np.zeros(vertex_count, dtype=bool)
    is_seam[seams.vertex] = True

    # One unknown per (vertex, patch), in vertex order.
    entry_sv = seams.entry_sv
    plain = np.flatnonzero(~is_seam & (vertex_patch < num_patches))
    row_vertex = np.concatenate([plain, seams.vertex[entry_sv]])
    row_patch = np.concatenate([vertex_patch[plain], seams.entry_patch])
    perm = np.argsort(row_vertex, kind="stable")
    row_vertex, row_patch = row_vertex[perm], row_patch[perm]
    rows_x = len(row_vertex)
    find_row = _lookup_table(row_vertex * (num_patches + 1) + row_patch)
    inverse_perm = np.empty(rows_x, dtype=np.int64)
    inverse_perm[perm] = np.arange(rows_x)
    entry_row = inverse_perm[len(plain) + np.arange(len(entry_sv))]

    # Tikhonov rows: adjacent vertices in the same patch should agree.
    a = faces
    b = np.roll(faces, -1, axis=1)
    lo = np.minimum(a, b).ravel()
    hi = np.maximum(a, b).ravel()
    vertex_edges = np.unique(lo * vertex_count + hi)
    ev = vertex_edges // vertex_count
    ew = vertex_edges % vertex_count
    rows_per_vertex = np.bincount(row_vertex, minlength=vertex_count)
    row_start = np.concatenate([[0], np.cumsum(rows_per_vertex)])
    rep = np.repeat(np.arange(len(ev)), rows_per_vertex[ev])
    local = np.arange(len(rep)) - np.repeat(np.cumsum(rows_per_vertex[ev]) - rows_per_vertex[ev], rows_per_vertex[ev])
    col_v = row_start[ev[rep]] + local
    patch_v = row_patch[col_v]
    col_w = find_row(ew[rep] * (num_patches + 1) + patch_v)
    ok = col_w >= 0
    col_v, col_w = col_v[ok], col_w[ok]
    rows_gamma = len(col_v)
    gamma = sparse.coo_matrix(
        (
            np.concatenate([np.full(rows_gamma, TIKHONOV_LAMBDA), np.full(rows_gamma, -TIKHONOV_LAMBDA)]).astype(np.float64),
            (np.concatenate([np.arange(rows_gamma), np.arange(rows_gamma)]), np.concatenate([col_v, col_w])),
        ),
        shape=(rows_gamma, rows_x),
    ).tocsr()

    # Seam rows: the two patches' colours at a seam vertex should agree.
    vertex_colors = _vertex_colors(patch_set, seams, views, images)
    entry_counts = np.diff(seams.entry_offsets)
    a_rows, a_cols, a_vals, coeff_b = [], [], [], []
    row_base = 0
    for c in np.unique(entry_counts[entry_counts >= 2]):
        c = int(c)
        svs = np.flatnonzero(entry_counts == c)
        entries = seams.entry_offsets[svs][:, None] + np.arange(c)[None, :]
        by_patch = np.argsort(seams.entry_patch[entries], axis=1, kind="stable")
        entries = np.take_along_axis(entries, by_patch, axis=1)
        i_idx, j_idx = np.triu_indices(c, 1)
        e_i = entries[:, i_idx].ravel()
        e_j = entries[:, j_idx].ravel()
        n = len(e_i)
        rows = row_base + np.arange(n)
        a_rows.append(np.concatenate([rows, rows]))
        a_cols.append(np.concatenate([entry_row[e_i], entry_row[e_j]]))
        a_vals.append(np.concatenate([np.ones(n), -np.ones(n)]))
        coeff_b.append(vertex_colors[e_j] - vertex_colors[e_i])
        row_base += n
    rows_a = row_base
    if rows_a == 0:
        return
    A = sparse.coo_matrix(
        (np.concatenate(a_vals), (np.concatenate(a_rows), np.concatenate(a_cols))), shape=(rows_a, rows_x)
    ).tocsr()
    coeff_b = np.concatenate(coeff_b).astype(np.float64)

    lhs = (A.T @ A + gamma.T @ gamma).tocsr()
    adjustments = np.zeros((rows_x, 3), dtype=np.float32)
    for channel in range(3):
        rhs = A.T @ coeff_b[:, channel]
        x = conjugate_gradient(lhs, rhs, tolerance=1e-4, max_iterations=1000)
        adjustments[:, channel] = (x - x.sum() / max(1, rows_x)).astype(np.float32)
    if log:
        log(f"  global leveling: {rows_x} unknowns, {rows_a} seam rows, {rows_gamma} smoothness rows")

    # Interpolate the offsets over every patch and add them in YCbCr.
    tex = patch_set.face_texcoords
    for p in range(num_patches):
        patch = patch_set.patches[p]
        rect = patch.rect
        if rect.width <= 0 or rect.height <= 0:
            continue
        view = views[patch.label]
        image = images[patch.label]
        corner_rows = find_row(faces[patch.faces].ravel() * (num_patches + 1) + p)
        corner = np.where(corner_rows[:, None] >= 0, adjustments[np.maximum(corner_rows, 0)], 0).reshape(-1, 3, 3)
        adjust = np.zeros((rect.height, rect.width, 3), dtype=np.float32)
        for ti, x, y, bary in raster.triangles_bary(tex[patch.faces], rect.width, rect.height, cull=True):
            values = (corner[ti] * bary[:, :, None]).sum(axis=1)
            _assign_last(adjust, x, y, values)
        adjust = dilate_mean(adjust)
        ys, xs = np.nonzero(np.any(adjust != 0, axis=2))
        iy = ys + rect.y
        ix = xs + rect.x
        inside = (ix >= 0) & (iy >= 0) & (ix < view.width) & (iy < view.height)
        ys, xs, iy, ix = ys[inside], xs[inside], iy[inside], ix[inside]
        if len(ys) == 0:
            continue
        pixel = image[iy, ix].astype(np.float32)
        adjusted = ycbcr2rgb(rgb2ycbcr(pixel) + adjust[ys, xs])
        image[iy, ix] = clamp_byte(round2int(adjusted))


# ── Local ───────────────────────────────────────────────────────────────────


def _shift(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """``mask`` read at ``(r + dr, c + dc)``, empty outside the image."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    rs, re = max(0, -dr), min(h, h - dr)
    cs, ce = max(0, -dc), min(w, w - dc)
    if re > rs and ce > cs:
        out[rs:re, cs:ce] = mask[rs + dr : re + dr, cs + dc : ce + dc]
    return out


def process_mask(mask: np.ndarray, strip_width: int) -> None:
    """Close gaps between the drawn border and the rasterised interior.

    Keep only a strip ``strip_width`` deep inside the patch border as the
    region to blend, its outer ring marked border.
    """
    h, w = mask.shape
    # Dilate border pixels (interior ones) into their four neighbours.
    if h > 2 and w > 2:
        border = np.zeros_like(mask, dtype=bool)
        border[1:-1, 1:-1] = mask[1:-1, 1:-1] == MASK_BORDER
        touched = _shift(border, 1, 0) | _shift(border, -1, 0) | _shift(border, 0, 1) | _shift(border, 0, -1)
        mask[touched & (mask != MASK_BORDER)] = MASK_INTERIOR
    # Erode interior pixels caught between a border and emptiness.
    interior = mask == MASK_INTERIOR
    kill = np.zeros_like(interior)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
        vl = _shift(mask, -dr, -dc)
        vr = _shift(mask, dr, dc)
        kill |= ((vl == MASK_BORDER) & (vr == MASK_EMPTY)) | ((vr == MASK_BORDER) & (vl == MASK_EMPTY))
    mask[interior & kill] = MASK_EMPTY
    # Interior pixels with an empty 4-neighbour become border.
    interior = mask == MASK_INTERIOR
    empty4 = (
        (_shift(mask, -1, 0) == MASK_EMPTY)
        | (_shift(mask, 0, -1) == MASK_EMPTY)
        | (_shift(mask, 1, 0) == MASK_EMPTY)
        | (_shift(mask, 0, 1) == MASK_EMPTY)
    )
    mask[interior & empty4] = MASK_BORDER
    # Erode ``strip_width`` rings inward from the patch's outer edge; what is
    # left is the core that stays as it is, its boundary ring the fixed border.
    kernel = np.ones((3, 3), dtype=np.uint8)
    valid = (mask != MASK_EMPTY).astype(np.uint8)
    core = valid
    for _ in range(strip_width):
        core = cv2.erode(core, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ring = core.astype(bool) & ~cv2.erode(core, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    mask[core != 0] = MASK_EMPTY
    mask[ring] = MASK_BORDER


def poisson_blend(source: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> None:
    """Poisson blend: inside the mask the result has the source's Laplacian,
    on the border it keeps the current values. Solved as the SPD system in
    the interior unknowns."""
    h, w = mask.shape
    interior = mask == MASK_INTERIOR
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    ids = np.flatnonzero(interior.ravel())
    n = len(ids)
    if n == 0:
        return
    index = np.full(h * w, -1, dtype=np.int64)
    index[ids] = np.arange(n)
    src = source.reshape(-1, 3).astype(np.float64)
    d = dst.reshape(-1, 3)
    m = mask.ravel()
    rhs = 4 * src[ids]
    rows = [np.arange(n)]
    cols = [np.arange(n)]
    vals = [np.full(n, 4.0)]
    for off in (-w, -1, 1, w):
        nb = ids + off
        rhs -= src[nb]
        col = index[nb]
        unknown = col >= 0
        known = ~unknown & (m[nb] != MASK_EMPTY)
        rhs[known] += d[nb[known]].astype(np.float64)
        rows.append(np.flatnonzero(unknown))
        cols.append(col[unknown])
        vals.append(np.full(int(unknown.sum()), -1.0))
    A = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    for channel in range(3):
        x = conjugate_gradient(A, rhs[:, channel], tolerance=1e-6, max_iterations=4000)
        d[ids, channel] = x.astype(np.float32)


def local_leveling(
    mesh: Mesh,
    patch_set: PatchSet,
    seams: SeamData,
    views: List[View],
    images: List[np.ndarray],
    strip_width: int = 20,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Patch by patch, in patch order: each patch reads the other views as
    levelled so far and writes only its own view's image."""
    if seams.count == 0:
        return
    num_patches = patch_set.patch_count
    entry_sv = seams.entry_sv
    entry_counts = np.diff(seams.entry_offsets)
    labels = np.array([p.label for p in patch_set.patches[:num_patches]], dtype=np.int64)
    by_patch = np.argsort(seams.entry_patch, kind="stable")
    patch_counts = np.bincount(seams.entry_patch, minlength=num_patches)
    patch_start = np.concatenate([[0], np.cumsum(patch_counts)])
    for p in range(num_patches):
        entries = by_patch[patch_start[p] : patch_start[p + 1]]
        _level_patch(p, entries, mesh, patch_set, seams, entry_sv, entry_counts, labels, views, images, strip_width)
        if log and ((p + 1) % 500 == 0 or p + 1 == num_patches):
            log(f"  levelled {p + 1}/{num_patches} patches")


def _level_patch(
    p: int,
    entries: np.ndarray,
    mesh: Mesh,
    patch_set: PatchSet,
    seams: SeamData,
    entry_sv: np.ndarray,
    entry_counts: np.ndarray,
    labels: np.ndarray,
    views: List[View],
    images: List[np.ndarray],
    strip_width: int,
) -> None:
    patch = patch_set.patches[p]
    rect = patch.rect
    w, h = rect.width, rect.height
    if w <= 0 or h <= 0:
        return
    view_index = patch.label
    view = views[view_index]
    target = images[view_index]
    inv255 = np.float32(1.0 / 255.0)

    # The patch as floats, and an untouched copy.
    ys = rect.y + np.arange(h)
    xs = rect.x + np.arange(w)
    ok_y = (ys >= 0) & (ys < view.height)
    ok_x = (xs >= 0) & (xs < view.width)
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[np.ix_(ok_y, ok_x)] = target[np.ix_(ys[ok_y], xs[ok_x])].astype(np.float32) * inv255
    image_org = image.copy()

    # Coverage mask.
    mask = np.zeros((h, w), dtype=np.uint8)
    _, mx, my = raster.triangles_fixed(patch_set.face_texcoords[patch.faces], w, h)
    mask[my, mx] = MASK_INTERIOR

    # Seams with neighbouring patches, drawn as the mean of both sides, then
    # each seam vertex as the mean over every patch meeting there. Collected
    # first, drawn in one pass so later strokes overwrite earlier ones.
    offset = np.array([rect.x, rect.y], dtype=np.float32)
    line_p0, line_p0_adj, line_p1, line_p1_adj, line_view, line_seq = [], [], [], [], [], []
    vertex_px, vertex_py, vertex_color, vertex_seq = [], [], [], []
    seq = 0
    proj = seams.entry_proj
    entry_patch = seams.entry_patch
    for e0 in entries:
        sv0 = int(entry_sv[e0])
        if entry_counts[sv0] < 2:
            continue
        p0 = proj[e0] - offset
        for q in seams.edges_of(int(e0)):
            sv1 = int(seams.edge_other[q])
            e0_adj = seams.lookup.get((sv1, p))
            if e0_adj is None:
                continue
            p0_adj = proj[e0_adj] - offset
            for e1 in seams.entries_of(sv0):
                patch1 = int(entry_patch[e1])
                if patch1 == p:
                    continue
                edges1 = seams.edges_of(e1)
                if not np.any(seams.edge_other[edges1.start : edges1.stop] == sv1):
                    continue
                e1_adj = seams.lookup.get((sv1, patch1))
                if e1_adj is None:
                    continue
                line_p0.append(p0)
                line_p0_adj.append(p0_adj)
                line_p1.append(proj[e1])
                line_p1_adj.append(proj[e1_adj])
                line_view.append(int(labels[patch1]))
                line_seq.append(seq)
                seq += 1
                break
        # The vertex itself: the mean over every patch meeting there.
        acc = np.zeros(3, dtype=np.float32)
        count = 0
        for e in seams.entries_of(sv0):
            vi = int(labels[entry_patch[e]])
            acc += imageops.sample_linear(images[vi], proj[e][None, :])[0] * inv255
            count += 1
        px, py = int(round2int(p0[0])), int(round2int(p0[1]))
        if 0 <= px < w and 0 <= py < h:
            vertex_px.append(px)
            vertex_py.append(py)
            vertex_color.append(acc / count)
            vertex_seq.append(seq)
            seq += 1

    draw_x, draw_y, draw_c, draw_seq = [], [], [], []
    if line_p0:
        P0 = np.array(line_p0, dtype=np.float32)
        P0A = np.array(line_p0_adj, dtype=np.float32)
        P1 = np.array(line_p1, dtype=np.float32)
        P1A = np.array(line_p1_adj, dtype=np.float32)
        views_of_line = np.array(line_view)
        li, lx, ly = raster.lines(P0, P0A)
        inside = (lx >= 0) & (ly >= 0) & (lx < w) & (ly < h)
        li, lx, ly = li[inside], lx[inside], ly[inside]
        if len(li):
            p0_dir = P0A - P0
            p1_dir = P1A - P1
            length = np.sqrt((p0_dir * p0_dir).sum(axis=1)).astype(np.float32)
            pt = np.stack([lx, ly], axis=-1).astype(np.float32)
            frac = (np.sqrt(((pt - P0[li]) ** 2).sum(axis=1)) / length[li]).astype(np.float32)
            color0 = imageops.sample_linear(image_org, P0[li] + p0_dir[li] * frac[:, None])
            pos1 = P1[li] + p1_dir[li] * frac[:, None]
            color1 = np.zeros_like(color0)
            vol = views_of_line[li]
            for v in np.unique(vol):
                m = vol == v
                color1[m] = imageops.sample_linear(images[int(v)], pos1[m]) * inv255
            draw_x.append(lx)
            draw_y.append(ly)
            draw_c.append((color0 + color1) * np.float32(0.5))
            draw_seq.append(np.array(line_seq)[li])
    if vertex_px:
        draw_x.append(np.array(vertex_px))
        draw_y.append(np.array(vertex_py))
        draw_c.append(np.array(vertex_color, dtype=np.float32))
        draw_seq.append(np.array(vertex_seq))
    if draw_x:
        dx = np.concatenate(draw_x)
        dy = np.concatenate(draw_y)
        dc = np.concatenate(draw_c)
        ds = np.concatenate(draw_seq)
        order = np.argsort(ds, kind="stable")
        _assign_last(image, dx[order], dy[order], dc[order])
        mask[dy, dx] = MASK_BORDER

    process_mask(mask, strip_width)
    poisson_blend(image_org, image, mask)

    ys_, xs_ = np.nonzero(mask != MASK_EMPTY)
    iy = ys_ + rect.y
    ix = xs_ + rect.x
    inside = (ix >= 0) & (iy >= 0) & (ix < view.width) & (iy < view.height)
    target[iy[inside], ix[inside]] = clamp_byte(round2int(image[ys_[inside], xs_[inside]] * 255))
