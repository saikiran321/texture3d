"""Command-line interface.

    texture3d DATASET_DIR --mesh mesh.ply [--out DIR] [--format obj|glb] ...

DATASET_DIR must follow the dataset.json layout understood by
texture3d.io.load_dataset. When --mesh is omitted, the repo convention
reconstructions/<dataset>/mesh.ply is tried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from texture3d.bake import bake
from texture3d.config import BakeConfig
from texture3d.io import load_dataset


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="texture3d", description=__doc__)
    p.add_argument("dataset", help="dataset directory (dataset.json layout)")
    p.add_argument("--mesh", type=Path, default=None, help="mesh to texture (.ply/.obj/.stl)")
    p.add_argument("--out", type=Path, default=None, help="output dir (default textured/<dataset>)")
    p.add_argument("--format", choices=["obj", "glb"], default="obj")
    # Keyframes
    p.add_argument("--gap", type=int, default=5, help="1 keyframe per N frames (1 = all)")
    # Mapping
    p.add_argument("--outlier", type=float, default=0.06,
                   help="Gaussian likelihood below which a view is an outlier for a face (0 = off)")
    p.add_argument("--data-ratio", type=float, default=0.1,
                   help="data cost relative to the smoothness cost; lower = larger patches")
    p.add_argument("--no-global-leveling", action="store_true", help="skip the global seam leveling")
    p.add_argument("--no-local-leveling", action="store_true", help="skip the local seam leveling")
    p.add_argument("--strip-width", type=int, default=20, help="local leveling strip width in px")
    p.add_argument("--sharpness", type=float, default=0.5, help="unsharp-mask weight on the atlas (0 = off)")
    p.add_argument("--max-texture", type=int, default=8192, help="largest atlas edge in px")
    p.add_argument("--no-fill-unseen", action="store_true",
                   help="leave sub-pixel faces no view rasterised in the empty colour")
    args = p.parse_args(argv)

    dataset_dir = Path(args.dataset)
    mesh_path = args.mesh or Path("reconstructions") / dataset_dir.name / "mesh.ply"
    if not mesh_path.exists():
        raise SystemExit(f"Mesh not found: {mesh_path} (pass --mesh)")
    out_dir = args.out or Path("textured") / dataset_dir.name

    cfg = BakeConfig()
    cfg.keyframes.rgbd_frame_gap = args.gap
    cfg.mapping.outlier_threshold = args.outlier
    cfg.mapping.data_smoothness_ratio = args.data_ratio
    cfg.mapping.global_seam_leveling = not args.no_global_leveling
    cfg.mapping.local_seam_leveling = not args.no_local_leveling
    cfg.mapping.local_leveling_strip_width = args.strip_width
    cfg.mapping.sharpness_weight = args.sharpness
    cfg.mapping.max_texture_size = args.max_texture
    cfg.mapping.fill_unseen = not args.no_fill_unseen

    frames = load_dataset(dataset_dir)
    result = bake(mesh_path, frames, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = result.export(out_dir / f"textured_mesh.{args.format}")
    (out_dir / "config.json").write_text(cfg.to_json() + "\n")
    result.stats["output"] = str(out_path)
    result.stats["mesh_input"] = str(mesh_path)
    result.stats["dataset_dir"] = str(dataset_dir)
    (out_dir / "summary.json").write_text(json.dumps(result.stats, indent=2) + "\n")
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
