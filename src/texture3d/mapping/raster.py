"""Rasterisers: barycentric for view face maps and colour-adjustment maps,
fixed-point for the local leveling mask, and Bresenham lines for the seams."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from texture3d.mapping.types import round2int


def edge(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Signed parallelogram area ``(x2 - x0) × (x1 - x0)``, float32."""
    a = x2 - x0
    b = x1 - x0
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def perspective_correct(bary: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Perspective-correct barycentrics for (N, 3) bary and (N, 3) z."""
    pb = np.stack(
        [bary[:, 0] * z[:, 1] * z[:, 2], bary[:, 1] * z[:, 0] * z[:, 2], bary[:, 2] * z[:, 0] * z[:, 1]], axis=-1
    )
    return pb / pb.sum(axis=1, keepdims=True)


def triangles_bary(
    tri: np.ndarray, width: int, height: int, cull: bool = True, max_candidates: int = 1 << 24
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Barycentric rasteriser over (N, 3, 2) float32 triangles.

    Yields chunks of ``(triangle index, x, y, bary)`` for every pixel whose
    centre has non-negative barycentric coordinates, within the image,
    culling triangles whose signed area is not positive. Within a chunk the
    pixels are in triangle order, then row-major inside each triangle.
    """
    tri = np.asarray(tri, dtype=np.float32)
    n = len(tri)
    if n == 0:
        return
    box_min = tri.min(axis=1)
    box_max = tri.max(axis=1)
    outside = (box_max[:, 0] < 0) | (box_min[:, 0] > width - 1) | (box_max[:, 1] < 0) | (box_min[:, 1] > height - 1)
    area = edge(tri[:, 0], tri[:, 1], tri[:, 2])
    keep = ~outside
    if cull:
        keep &= area > 0
    else:
        keep &= area != 0
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return
    min_x = np.clip(np.floor(box_min[idx, 0]).astype(np.int64), 0, width - 1)
    min_y = np.clip(np.floor(box_min[idx, 1]).astype(np.int64), 0, height - 1)
    max_x = np.clip(np.ceil(box_max[idx, 0]).astype(np.int64), 0, width - 1)
    max_y = np.clip(np.ceil(box_max[idx, 1]).astype(np.int64), 0, height - 1)
    bw = max_x - min_x + 1
    bh = max_y - min_y + 1
    counts = bw * bh
    inv_area = (np.float32(1) / area[idx]).astype(np.float32)

    start = 0
    total = len(idx)
    while start < total:
        # Take triangles until the candidate budget is reached.
        cum = np.cumsum(counts[start:])
        take = int(np.searchsorted(cum, max_candidates, side="right"))
        take = max(1, min(take, total - start))
        sel = slice(start, start + take)
        c = counts[sel]
        rep = np.repeat(np.arange(start, start + take), c)
        offsets = np.repeat(np.cumsum(c) - c, c)
        local = np.arange(int(c.sum())) - offsets
        x = min_x[rep] + local % bw[rep]
        y = min_y[rep] + local // bw[rep]
        p = np.stack([x, y], axis=-1).astype(np.float32)
        t = tri[idx[rep]]
        ia = inv_area[rep]
        b1 = edge(t[:, 1], t[:, 2], p) * ia
        b2 = edge(t[:, 2], t[:, 0], p) * ia
        b3 = edge(t[:, 0], t[:, 1], p) * ia
        inside = (b1 >= 0) & (b2 >= 0) & (b3 >= 0)
        yield idx[rep[inside]], x[inside], y[inside], np.stack([b1[inside], b2[inside], b3[inside]], axis=-1)
        start += take


def triangles_fixed(tri: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """28.4 fixed-point half-space rasteriser with a top-left fill
    convention, over (N, 3, 2) triangles. Returns ``(triangle index, x, y)``
    for the covered pixels inside ``width`` x ``height``."""
    tri = np.asarray(tri, dtype=np.float32)
    n = len(tri)
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    X = round2int(16 * tri[..., 0])
    Y = round2int(16 * tri[..., 1])
    X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
    Y1, Y2, Y3 = Y[:, 0], Y[:, 1], Y[:, 2]
    DX12, DX23, DX31 = X1 - X2, X2 - X3, X3 - X1
    DY12, DY23, DY31 = Y1 - Y2, Y2 - Y3, Y3 - Y1
    min_x = (X.min(axis=1) + 0xF) >> 4
    max_x = (X.max(axis=1) + 0xF) >> 4
    min_y = (Y.min(axis=1) + 0xF) >> 4
    max_y = (Y.max(axis=1) + 0xF) >> 4
    C1 = DY12 * X1 - DX12 * Y1 + ((DY12 < 0) | ((DY12 == 0) & (DX12 > 0)))
    C2 = DY23 * X2 - DX23 * Y2 + ((DY23 < 0) | ((DY23 == 0) & (DX23 > 0)))
    C3 = DY31 * X3 - DX31 * Y3 + ((DY31 < 0) | ((DY31 == 0) & (DX31 > 0)))
    # Only pixels inside the image can be reported, so clip the boxes first.
    lo_x = np.clip(min_x, 0, width)
    hi_x = np.clip(max_x, 0, width)
    lo_y = np.clip(min_y, 0, height)
    hi_y = np.clip(max_y, 0, height)
    bw = np.maximum(hi_x - lo_x, 0)
    bh = np.maximum(hi_y - lo_y, 0)
    counts = bw * bh
    rep = np.repeat(np.arange(n), counts)
    if len(rep) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    local = np.arange(int(counts.sum())) - offsets
    x = lo_x[rep] + local % bw[rep]
    y = lo_y[rep] + local // bw[rep]
    x16 = x << 4
    y16 = y << 4
    inside = (
        (C1[rep] + DX12[rep] * y16 - DY12[rep] * x16 > 0)
        & (C2[rep] + DX23[rep] * y16 - DY23[rep] * x16 > 0)
        & (C3[rep] + DX31[rep] * y16 - DY31[rep] * x16 > 0)
    )
    return rep[inside], x[inside], y[inside]


def lines(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bresenham from the rounded start to the rounded end, end exclusive,
    over (N, 2) endpoints. Returns ``(line index, x, y)`` with each line's
    pixels in drawing order, lines in input order."""
    p1 = np.asarray(p1, dtype=np.float32).copy()
    p2 = np.asarray(p2, dtype=np.float32).copy()
    n = len(p1)
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    x1, y1 = p1[:, 0].copy(), p1[:, 1].copy()
    x2, y2 = p2[:, 0].copy(), p2[:, 1].copy()
    steep = np.abs(y2 - y1) > np.abs(x2 - x1)
    x1s, y1s = np.where(steep, y1, x1), np.where(steep, x1, y1)
    x2s, y2s = np.where(steep, y2, x2), np.where(steep, x2, y2)
    flip = x1s > x2s
    xa, xb = np.where(flip, x2s, x1s), np.where(flip, x1s, x2s)
    ya, yb = np.where(flip, y2s, y1s), np.where(flip, y1s, y2s)
    dx = (xb - xa).astype(np.float32)
    dy = np.abs(yb - ya).astype(np.float32)
    error = dx / np.float32(2)
    y = round2int(ya)
    ystep = np.where(ya < yb, 1, -1)
    x = round2int(xa)
    max_x = round2int(xb)
    steps = np.maximum(max_x - x, 0)
    total = int(steps.sum())
    out_line = np.empty(total, dtype=np.int64)
    out_x = np.empty(total, dtype=np.int64)
    out_y = np.empty(total, dtype=np.int64)
    # Each line's pixels are contiguous in the output; step k of line i goes to
    # slot start[i] + k.
    start = np.cumsum(steps) - steps
    active = steps > 0
    k = 0
    while active.any():
        ids = np.flatnonzero(active)
        slot = start[ids] + k
        px, py = x[ids], y[ids]
        out_line[slot] = ids
        out_x[slot] = np.where(steep[ids], py, px)
        out_y[slot] = np.where(steep[ids], px, py)
        error[ids] -= dy[ids]
        inc = error[ids] < 0
        y[ids] += np.where(inc, ystep[ids], 0)
        error[ids] += np.where(inc, dx[ids], np.float32(0))
        x[ids] += 1
        k += 1
        active = steps > k
    return out_line, out_x, out_y
