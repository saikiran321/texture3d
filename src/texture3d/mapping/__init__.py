"""One-keyframe-per-face texture mapping with seam leveling.

See ``texture3d.mapping.types`` for the stage list and conventions.
"""

from texture3d.mapping.mapper import Tuning, run
from texture3d.mapping.types import (
    Camera,
    EmptyMesh,
    MappingError,
    Mesh,
    NoViews,
    NoVisibleFace,
    PatchTooLarge,
    Result,
    View,
)

__all__ = [
    "run",
    "Tuning",
    "Mesh",
    "Camera",
    "View",
    "Result",
    "MappingError",
    "EmptyMesh",
    "NoViews",
    "NoVisibleFace",
    "PatchTooLarge",
]
