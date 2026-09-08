"""Shared types and colour helpers of the texture mapping.

One photograph per face, then seams levelled. Stages, each its own module:

1. ``topology``  incident faces, face adjacency, smoothed face normals
2. ``scoring``   per view, rasterise the mesh and score each visible face by the
                 image gradient it covers; reject views whose colour disagrees
                 with the majority
3. ``labeling``  one view per face, an MRF solved by loopy belief propagation
                 with a Potts smoothness term
4. ``patches``   connected same-view faces become patches with a rectangle in
                 their view; seam vertices between them
5. ``leveling``  global (least-squares colour offsets per vertex and patch) and
                 local (Poisson blend of a border strip)
6. ``atlas``     pack the patch rectangles into one texture, write the
                 per-corner texture coordinates, sharpen

Conventions: pixels are BGR, integer pixel coordinates are pixel centres,
the camera looks down +Z with +Y down, and internal arithmetic is float32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

#: An absent face, view, patch or label.
NO_ID = -1


@dataclass
class Mesh:
    positions: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int64, counter-clockwise seen from outside

    def __post_init__(self):
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: ``K`` in pixels of the image it is paired with, ``R``
    rotating world to camera, ``C`` the centre in world coordinates."""

    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # (3, 3) float64
    C: np.ndarray  # (3,) float64

    def to_camera(self, world: np.ndarray) -> np.ndarray:
        """(N, 3) world points to camera space, in double."""
        return (np.asarray(world, dtype=np.float64) - self.C) @ self.R.T

    def to_image(self, cam: np.ndarray) -> np.ndarray:
        """(N, 3) camera-space points to (N, 2) pixel coordinates, in double."""
        z = cam[:, 2]
        return np.stack([self.cx + self.fx * (cam[:, 0] / z), self.cy + self.fy * (cam[:, 1] / z)], axis=-1)

    def project(self, world: np.ndarray) -> np.ndarray:
        """World points to float32 pixel coordinates."""
        return self.to_image(self.to_camera(world)).astype(np.float32)


@dataclass
class View:
    """One keyframe: BGR8 pixels and the camera that took it."""

    bgr: np.ndarray  # (H, W, 3) uint8
    camera: Camera

    @property
    def width(self) -> int:
        return self.bgr.shape[1]

    @property
    def height(self) -> int:
        return self.bgr.shape[0]


@dataclass
class FaceViews:
    """Per face, the views that see it, in view order (CSR)."""

    offsets: np.ndarray  # (F + 1,) int64
    view: np.ndarray  # (N,) int64
    quality: np.ndarray  # (N,) float32
    color: np.ndarray  # (N, 3) float32, YCbCr taken over the BGR channel order

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def visible(self) -> np.ndarray:
        return self.counts > 0


@dataclass
class Rect:
    """An integer pixel rectangle (x, y, width, height)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_contained_in(self, other: Rect) -> bool:
        return (
            self.x >= other.x
            and self.y >= other.y
            and self.x + self.width <= other.x + other.width
            and self.y + self.height <= other.y + other.height
        )


@dataclass
class Result:
    #: Per face: the view it is textured from, or NO_ID.
    labels: np.ndarray
    #: Per face: its patch index; ``patch_count`` is the untextured patch.
    face_patch: np.ndarray
    patch_count: int
    #: Square atlas, BGR8.
    atlas_bgr: np.ndarray
    atlas_size: int
    #: (F, 3, 2) atlas pixel coordinates (integer = pixel centre, row 0 top).
    face_texcoords: np.ndarray
    #: Where each textured patch landed in the atlas.
    patch_rects: List[Rect] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class MappingError(Exception):
    pass


class EmptyMesh(MappingError):
    pass


class NoViews(MappingError):
    pass


class NoVisibleFace(MappingError):
    pass


class PatchTooLarge(MappingError):
    pass


# Colour helpers shared by the stages.

_RGB2YCBCR = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=np.float32
)
_YCBCR_OFFSET = np.array([0.0, 128.0, 128.0], dtype=np.float32)


def rgb2ycbcr(v: np.ndarray) -> np.ndarray:
    """RGB-to-YCbCr on the last axis. Views hand this BGR, so the channels
    stay in BGR order through the conversion."""
    v = np.asarray(v, dtype=np.float32)
    return v @ _RGB2YCBCR.T + _YCBCR_OFFSET


def ycbcr2rgb(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    y = v[..., 0]
    cb = v[..., 1] - 128.0
    cr = v[..., 2] - 128.0
    return np.stack([y + cr * 1.402, y + cb * -0.34414 + cr * -0.71414, y + cb * 1.772], axis=-1)


def round2int(x) -> np.ndarray:
    """Round half up: ``floor(x + 0.5)``."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


def clamp_byte(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)
