"""texture3d: texture a triangle mesh from posed RGB-D frames.

Each face is textured from one keyframe. Patches of same-keyframe faces are
cut from that image, seams between patches are levelled, and the patches
are packed into a single atlas (see `texture3d.mapping`). Built on trimesh.

    import texture3d

    frames = texture3d.load_dataset("apt0_seq0")   # or build RGBDFrames yourself
    result = texture3d.bake("mesh.ply", frames)
    result.export("textured.obj")                # or .glb
"""

from texture3d.bake import BakeResult, bake
from texture3d.camera import PinholeCamera, RGBDFrame
from texture3d.config import BakeConfig, KeyframeConfig, MappingConfig
from texture3d.io import load_dataset

__version__ = "0.2.0"

__all__ = [
    "bake",
    "BakeResult",
    "BakeConfig",
    "KeyframeConfig",
    "MappingConfig",
    "PinholeCamera",
    "RGBDFrame",
    "load_dataset",
    "__version__",
]
