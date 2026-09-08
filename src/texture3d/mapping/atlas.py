"""Stage 6: pack the patch rectangles into one square texture, copy the
(levelled) pixels, write the per-corner texture coordinates, sharpen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from texture3d.mapping import imageops
from texture3d.mapping.patches import PatchSet
from texture3d.mapping.types import NO_ID, PatchTooLarge, Rect, View


def compute_texture_size(rects: Sequence[Rect], multiple: int) -> int:
    """The square that would hold the rectangles at a 0.9 fill, at least the
    largest side, rounded up to a power of two or a multiple."""
    area = sum(r.area for r in rects)
    max_side = max([max(r.width, r.height) for r in rects] + [0])
    scaled = int(np.ceil(np.float32(area) * np.float32(1.0 / 0.9)))
    side = max(int(np.ceil(np.sqrt(np.float32(scaled)))), max_side)
    if multiple > 0:
        return ((side + multiple - 1) // multiple) * multiple
    size = 1
    while size < side:
        size *= 2
    return size


class MaxRectsBinPack:
    """MaxRects bin packing with the bottom-left rule, rotation allowed.

    Every step places, of all remaining rectangles, the one whose best
    position is lowest (then leftmost).
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.free = np.array([[0, 0, width, height]], dtype=np.int64)

    def insert(self, sizes: np.ndarray) -> Tuple[List[Tuple[Rect, int]], np.ndarray]:
        """Places as many of the (N, 2) ``sizes`` as fit. Returns the placed
        rectangles with their input index, and the indices left unplaced."""
        remaining = np.arange(len(sizes))
        placed: List[Tuple[Rect, int]] = []
        big = np.int64(1) << np.int64(62)
        while len(remaining):
            w = sizes[remaining, 0][:, None]
            h = sizes[remaining, 1][:, None]
            fx, fy, fw, fh = (self.free[:, i][None, :] for i in range(4))
            fr = np.arange(len(self.free), dtype=np.int64)[None, :]
            # Candidate keys order (top, x, free rect, orientation).
            fit0 = (fw >= w) & (fh >= h)
            fit1 = (fw >= h) & (fh >= w)
            key0 = np.where(fit0, ((fy + h) << 39) | (fx << 25) | (fr << 1), big)
            key1 = np.where(fit1, ((fy + w) << 39) | (fx << 25) | (fr << 1) | 1, big)
            best = np.minimum(key0.min(axis=1), key1.min(axis=1))
            feasible = best < big
            if not feasible.any():
                break
            top = best >> 39
            x = (best >> 25) & ((1 << 14) - 1)
            top = np.where(feasible, top, big)
            x = np.where(feasible, x, big)
            u = int(np.lexsort((x, top))[0])
            k = int(best[u])
            free_index = (k >> 1) & ((1 << 24) - 1)
            rotated = bool(k & 1)
            fxu, fyu = int(self.free[free_index, 0]), int(self.free[free_index, 1])
            pw, ph = int(sizes[remaining[u], 0]), int(sizes[remaining[u], 1])
            node = Rect(fxu, fyu, ph if rotated else pw, pw if rotated else ph)
            self._place(node)
            placed.append((node, int(remaining[u])))
            remaining = np.delete(remaining, u)
        return placed, remaining

    def _place(self, node: Rect) -> None:
        fx, fy, fw, fh = (self.free[:, i] for i in range(4))
        ux, uy, uw, uh = node.x, node.y, node.width, node.height
        hit = ~((ux >= fx + fw) | (ux + uw <= fx) | (uy >= fy + fh) | (uy + uh <= fy))
        pieces = []
        for i in np.flatnonzero(hit):
            x, y, w, h = self.free[i]
            if ux < x + w and ux + uw > x:
                if y < uy < y + h:
                    pieces.append((x, y, w, uy - y))
                if uy + uh < y + h:
                    pieces.append((x, uy + uh, w, y + h - (uy + uh)))
            if uy < y + h and uy + uh > y:
                if x < ux < x + w:
                    pieces.append((x, y, ux - x, h))
                if ux + uw < x + w:
                    pieces.append((ux + uw, y, x + w - (ux + uw), h))
        kept = self.free[~hit]
        if pieces:
            kept = np.concatenate([kept, np.array(pieces, dtype=np.int64).reshape(-1, 4)])
        self.free = kept
        self._prune()

    def _prune(self) -> None:
        """Drop free rectangles contained in another; of two identical ones
        the earlier goes."""
        n = len(self.free)
        if n < 2:
            return
        x, y, w, h = (self.free[:, i] for i in range(4))
        contained = (
            (x[:, None] >= x[None, :])
            & (y[:, None] >= y[None, :])
            & (x[:, None] + w[:, None] <= x[None, :] + w[None, :])
            & (y[:, None] + h[:, None] <= y[None, :] + h[None, :])
        )
        identical = contained & contained.T
        np.fill_diagonal(contained, False)
        earlier = np.arange(n)[None, :] < np.arange(n)[:, None]
        contained &= ~(identical & earlier)
        self.free = self.free[~contained.any(axis=1)]


def shelf_pack(sizes: np.ndarray, width: int, height: int) -> Tuple[List[Tuple[Rect, int]], np.ndarray]:
    """Next-fit decreasing-height shelf packing, rotation allowed: every
    rectangle is laid so it is wider than tall, sorted by height, and placed
    left to right on shelves stacked from the top. O(N log N), for the tens of
    thousands of tiny patches a dense mesh seen by many views produces, where
    MaxRects is quadratic. Returns the placed rectangles with their input
    index, and the indices left unplaced."""
    n = len(sizes)
    if n == 0:
        return [], np.zeros(0, dtype=np.int64)
    w = sizes[:, 0].astype(np.int64)
    h = sizes[:, 1].astype(np.int64)
    rotated = h > w
    pw = np.where(rotated, h, w)
    ph = np.where(rotated, w, h)
    order = np.lexsort((np.arange(n), -pw, -ph))
    placed: List[Tuple[Rect, int]] = []
    unplaced: List[int] = []
    y = 0
    i = 0
    while i < n:
        shelf_h = int(ph[order[i]])
        if y + shelf_h > height:
            unplaced.extend(order[i:].tolist())
            break
        # Take rectangles while they fit on this shelf's width; those wider
        # than the texture can never be placed.
        widths = pw[order[i:]]
        x_end = np.cumsum(widths)
        fits = x_end <= width
        take = int(np.argmin(fits)) if not fits.all() else len(widths)
        if take == 0:
            unplaced.append(int(order[i]))
            i += 1
            continue
        xs = x_end[:take] - widths[:take]
        for k in range(take):
            idx = int(order[i + k])
            placed.append((Rect(int(xs[k]), y, int(pw[idx]), int(ph[idx])), idx))
        y += shelf_h
        i += take
    return placed, np.array(unplaced, dtype=np.int64)


def pack_rects(sizes: np.ndarray, size: int, exact_limit: int) -> Tuple[List[Tuple[Rect, int]], np.ndarray]:
    """MaxRects up to ``exact_limit`` rectangles, the shelf packer beyond it
    (0 = always MaxRects)."""
    if exact_limit <= 0 or len(sizes) <= exact_limit:
        return MaxRectsBinPack(size, size).insert(sizes)
    return shelf_pack(sizes, size, size)


@dataclass
class AtlasOutput:
    pixels: np.ndarray  # (S, S, 3) BGR8
    size: int
    face_texcoords: np.ndarray  # (F, 3, 2)
    #: Where each textured patch landed in the atlas.
    patch_rects: List[Rect] = field(default_factory=list)


def generate(
    patch_set: PatchSet,
    images: List[np.ndarray],
    views: List[View],
    texture_size_multiple: int = 0,
    max_texture_size: int = 8192,
    empty_color_bgr: Tuple[int, int, int] = (39, 127, 255),
    sharpness_weight: float = 0.5,
    exact_packing_limit: int = 4000,
) -> AtlasOutput:
    """Packs, copies, and writes the final texture coordinates. Raises
    ``PatchTooLarge`` when a patch does not fit the largest texture allowed.
    Up to ``exact_packing_limit`` patches are packed with MaxRects; more
    take the shelf packer (see ``shelf_pack``)."""
    patches = patch_set.patches
    for patch in patches:
        if max_texture_size > 0 and (patch.rect.width > max_texture_size or patch.rect.height > max_texture_size):
            raise PatchTooLarge(f"patch {patch.rect.width}x{patch.rect.height} exceeds {max_texture_size}")
    sizes = np.array([[p.rect.width, p.rect.height] for p in patches], dtype=np.int64)
    texture_size = compute_texture_size([p.rect for p in patches], texture_size_multiple)
    if max_texture_size > 0:
        texture_size = min(texture_size, max_texture_size)
    while True:
        placed, remaining = pack_rects(sizes, texture_size, exact_packing_limit)
        if len(remaining) == 0 or texture_size == max_texture_size:
            if len(remaining):
                r = patches[int(remaining[0])].rect
                raise PatchTooLarge(f"patch {r.width}x{r.height} does not fit a {texture_size} texture")
            break
        texture_size *= 2
        if max_texture_size > 0:
            texture_size = min(texture_size, max_texture_size)

    atlas = np.empty((texture_size, texture_size, 3), dtype=np.uint8)
    atlas[:] = np.array(empty_color_bgr, dtype=np.uint8)
    texcoords = patch_set.face_texcoords.copy()
    patch_rects: List[Rect] = []
    for rect, p in placed:
        patch = patches[p]
        rotated = rect.width != patch.rect.width
        if patch.label != NO_ID:
            patch_rects.append(rect)
            view = views[patch.label]
            source = images[patch.label]
            r0 = max(0, -patch.rect.y)
            r1 = min(patch.rect.height, view.height - patch.rect.y)
            c0 = max(0, -patch.rect.x)
            c1 = min(patch.rect.width, view.width - patch.rect.x)
            if r1 > r0 and c1 > c0:
                block = source[patch.rect.y + r0 : patch.rect.y + r1, patch.rect.x + c0 : patch.rect.x + c1]
                if rotated:
                    # A rotated placement is the transpose of the patch.
                    atlas[rect.y + c0 : rect.y + c1, rect.x + r0 : rect.x + r1] = block.transpose(1, 0, 2)
                else:
                    atlas[rect.y + r0 : rect.y + r1, rect.x + c0 : rect.x + c1] = block
        t = texcoords[patch.faces]
        if rotated:
            texcoords[patch.faces] = np.stack([t[..., 1] + rect.x, t[..., 0] + rect.y], axis=-1)
        else:
            texcoords[patch.faces] = t + np.array([rect.x, rect.y], dtype=np.float32)

    if sharpness_weight > 0:
        atlas = imageops.sharpen(atlas, 1.5, sharpness_weight)
    return AtlasOutput(pixels=atlas, size=texture_size, face_texcoords=texcoords, patch_rects=patch_rects)
