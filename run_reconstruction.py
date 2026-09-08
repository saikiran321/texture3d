#!/usr/bin/env python3
"""CLI for RGBD reconstruction of apt0 subsets.

Examples:
    # Reconstruct one subset
    python run_reconstruction.py apt0_seq0

    # All three subsets, every 2nd frame, 1cm voxels
    python run_reconstruction.py apt0_seq0 apt0_seq1 apt0_seq2 --stride 2

    # Quick preview: coarse voxels, subset of frames
    python run_reconstruction.py apt0_seq0 --stride 5 --voxel 0.02 --max-frames 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rgbd_recon.config import PipelineConfig
from rgbd_recon.pipeline import ReconstructionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datasets", nargs="+", help="Dataset directories (e.g. apt0_seq0)")
    parser.add_argument("--output-root", type=Path, default=Path("reconstructions"))
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = all)")
    parser.add_argument("--voxel", type=float, default=0.01, help="TSDF voxel size in meters")
    parser.add_argument("--sdf-trunc", type=float, default=0.04, help="SDF truncation in meters")
    parser.add_argument("--depth-trunc", type=float, default=4.0, help="Max depth in meters")
    parser.add_argument("--no-depth-filter", action="store_true", help="Skip OpenCV depth cleanup")
    parser.add_argument("--no-point-cloud", action="store_true")
    parser.add_argument(
        "--simplify", type=int, default=0, help="Target triangle count (0 = no simplification)"
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        dataset_dir = Path(dataset)
        if not dataset_dir.exists():
            raise SystemExit(f"Dataset not found: {dataset_dir}")

        cfg = PipelineConfig(
            dataset_dir=dataset_dir,
            output_dir=args.output_root / dataset_dir.name,
            frame_stride=args.stride,
            max_frames=args.max_frames,
            save_point_cloud=not args.no_point_cloud,
        )
        cfg.fusion.voxel_length_m = args.voxel
        cfg.fusion.sdf_trunc_m = args.sdf_trunc
        cfg.fusion.depth_trunc_m = args.depth_trunc
        cfg.depth_filter.enabled = not args.no_depth_filter
        cfg.depth_filter.max_depth_m = args.depth_trunc
        cfg.mesh.simplify_target_triangles = args.simplify

        summary = ReconstructionPipeline(cfg).run()
        print(f"Done: {summary['mesh']['path']}\n")


if __name__ == "__main__":
    main()
