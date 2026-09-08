#!/usr/bin/env python3
"""Convert BundleFusion-style apt0 RGBD sequence into continuous SLAM subsets."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


INFO_RE = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*=\s*(.*?)\s*$")


def parse_info(path: Path) -> dict:
    info: dict = {}
    for line in path.read_text().splitlines():
        m = INFO_RE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        parts = raw.split()
        if len(parts) == 1:
            token = parts[0]
            if re.fullmatch(r"-?\d+", token):
                info[key] = int(token)
            elif re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", token):
                info[key] = float(token)
            else:
                info[key] = token
        else:
            info[key] = [float(x) for x in parts]
    return info


def mat4_from_16(vals: list[float]) -> np.ndarray:
    if len(vals) != 16:
        raise ValueError(f"Expected 16 values for 4x4 matrix, got {len(vals)}")
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def parse_pose(path: Path) -> np.ndarray:
    vals = [float(x) for x in path.read_text().split()]
    pose = mat4_from_16(vals)
    if not np.isfinite(pose).all():
        raise ValueError(f"Non-finite pose in {path}")
    return pose


def K_from_4x4(mat: np.ndarray) -> np.ndarray:
    return mat[:3, :3].copy()


def convert_subset(
    src: Path,
    dst: Path,
    *,
    subset_name: str,
    start: int,
    count: int,
    info: dict,
    color_K: np.ndarray,
    depth_K: np.ndarray,
    depth_scale: float,
) -> dict:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Shared camera files at subset root.
    np.savetxt(dst / "intrinsics.txt", color_K, fmt="%.9f")
    np.savetxt(dst / "depth_intrinsics.txt", depth_K, fmt="%.9f")
    shutil.copy2(src / "info.txt", dst / "info.txt")

    frames = []
    errors = []

    for local_idx in range(count):
        src_idx = start + local_idx
        frame_id = f"{local_idx:06d}"
        color_src = src / f"frame-{src_idx:06d}.color.jpg"
        depth_src = src / f"frame-{src_idx:06d}.depth.png"
        pose_src = src / f"frame-{src_idx:06d}.pose.txt"

        try:
            if not color_src.exists() or not depth_src.exists() or not pose_src.exists():
                raise FileNotFoundError(f"Missing files for source frame {src_idx:06d}")

            pose = parse_pose(pose_src)
            with Image.open(color_src) as rgb_img:
                width, height = rgb_img.size
            with Image.open(depth_src) as depth_img:
                depth_arr = np.array(depth_img)
                valid = depth_arr > 0
                depth_min = int(depth_arr[valid].min()) if valid.any() else 0
                depth_max = int(depth_arr[valid].max()) if valid.any() else 0
                valid_ratio = float(valid.mean())

            frame_dir = dst / frame_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(color_src, frame_dir / "rgb.jpg")
            shutil.copy2(depth_src, frame_dir / "depth.png")
            np.savetxt(frame_dir / "pose.txt", pose, fmt="%.9f")

            meta = {
                "frame_id": frame_id,
                "local_index": local_idx,
                "source_frame_index": src_idx,
                "source_frame_name": f"frame-{src_idx:06d}",
                "subset": subset_name,
                "dataset": "apt0",
                "sensor": info.get("m_sensorName", "StructureSensor"),
                "files": {
                    "rgb": "rgb.jpg",
                    "depth": "depth.png",
                    "pose": "pose.txt",
                },
                "image_size": {"width": width, "height": height},
                "intrinsics": {
                    "matrix_3x3": color_K.tolist(),
                    "fx": float(color_K[0, 0]),
                    "fy": float(color_K[1, 1]),
                    "cx": float(color_K[0, 2]),
                    "cy": float(color_K[1, 2]),
                    "shared_file": "../intrinsics.txt",
                },
                "depth_intrinsics": {
                    "matrix_3x3": depth_K.tolist(),
                    "fx": float(depth_K[0, 0]),
                    "fy": float(depth_K[1, 1]),
                    "cx": float(depth_K[0, 2]),
                    "cy": float(depth_K[1, 2]),
                    "shared_file": "../depth_intrinsics.txt",
                },
                "pose": {
                    "matrix_4x4": pose.tolist(),
                    "convention": (
                        "BundleFusion/StructureSensor camera-to-world pose (4x4). "
                        "Maps camera coordinates into the sequence world frame."
                    ),
                },
                "depth": {
                    "unit": "millimeter_via_depthShift",
                    "depth_shift": float(info.get("m_depthShift", 1000)),
                    "scale_to_meters": depth_scale,
                    "dtype": "uint16",
                    "valid_min": depth_min,
                    "valid_max": depth_max,
                    "valid_ratio": valid_ratio,
                    "note": "metric_depth_m = depth_png / depth_shift",
                },
            }
            (frame_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

            frames.append(
                {
                    "frame_id": frame_id,
                    "local_index": local_idx,
                    "source_frame_index": src_idx,
                    "rgb": f"{frame_id}/rgb.jpg",
                    "depth": f"{frame_id}/depth.png",
                    "pose": f"{frame_id}/pose.txt",
                    "meta": f"{frame_id}/meta.json",
                    "width": width,
                    "height": height,
                    "fx": float(color_K[0, 0]),
                    "fy": float(color_K[1, 1]),
                    "cx": float(color_K[0, 2]),
                    "cy": float(color_K[1, 2]),
                    "depth_scale_to_meters": depth_scale,
                    "translation": pose[:3, 3].tolist(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"local_index": local_idx, "source_frame_index": src_idx, "error": str(exc)})

        done = local_idx + 1
        if done % 200 == 0 or done == count:
            print(f"  [{subset_name}] {done}/{count}")

    dataset = {
        "name": subset_name,
        "source": str(src),
        "source_range": {"start": start, "end_exclusive": start + count, "count": count},
        "num_frames": len(frames),
        "num_errors": len(errors),
        "contiguous": True,
        "frame_order": "increasing source_frame_index",
        "sensor": info.get("m_sensorName", "StructureSensor"),
        "image_size": {
            "width": int(info.get("m_colorWidth", 640)),
            "height": int(info.get("m_colorHeight", 480)),
        },
        "intrinsics": {
            "color_matrix_3x3": color_K.tolist(),
            "depth_matrix_3x3": depth_K.tolist(),
            "files": {
                "color": "intrinsics.txt",
                "depth": "depth_intrinsics.txt",
                "info": "info.txt",
            },
        },
        "depth": {
            "depth_shift": float(info.get("m_depthShift", 1000)),
            "scale_to_meters": depth_scale,
            "unit_note": "metric_depth_m = depth_png / depth_shift",
        },
        "pose_convention": "camera-to-world 4x4",
        "notes": [
            "Frames are a contiguous temporal window for RGBD reconstruction / SLAM.",
            "Local frame ids are renumbered 000000..N-1; source_frame_index keeps original apt0 ids.",
            "Intrinsics are constant across the subset and stored at the subset root.",
        ],
        "frames": frames,
        "errors": errors,
    }
    (dst / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("/Users/sai/Downloads/apt0"))
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/Users/sai/Desktop/work/projects/3d-texturing/rgbd_dataset"),
    )
    parser.add_argument("--frames-per-subset", type=int, default=1000)
    parser.add_argument(
        "--ranges",
        type=str,
        default="",
        help="Optional comma-separated start indices, e.g. 0,3780,7560",
    )
    args = parser.parse_args()

    info = parse_info(args.src / "info.txt")
    color_K = K_from_4x4(mat4_from_16(info["m_calibrationColorIntrinsic"]))
    depth_K = K_from_4x4(mat4_from_16(info["m_calibrationDepthIntrinsic"]))
    depth_shift = float(info.get("m_depthShift", 1000))
    depth_scale = 1.0 / depth_shift
    total_frames = int(info.get("m_frames.size", 0))
    if total_frames <= 0:
        total_frames = len(list(args.src.glob("frame-*.color.jpg")))

    n = args.frames_per_subset
    if args.ranges.strip():
        starts = [int(x) for x in args.ranges.split(",")]
    else:
        # Three continuous windows: start / middle / end of the trajectory.
        starts = [0, max(0, (total_frames - n) // 2), max(0, total_frames - n)]

    if len(starts) != 3:
        raise ValueError(f"Expected 3 subset starts, got {starts}")
    for s in starts:
        if s < 0 or s + n > total_frames:
            raise ValueError(f"Range [{s}, {s+n}) out of bounds for {total_frames} frames")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)

    summaries = []
    for i, start in enumerate(starts):
        subset_name = f"apt0_seq{i}"
        print(f"Building {subset_name}: source frames [{start}, {start+n})")
        summary = convert_subset(
            args.src,
            args.dst / subset_name,
            subset_name=subset_name,
            start=start,
            count=n,
            info=info,
            color_K=color_K,
            depth_K=depth_K,
            depth_scale=depth_scale,
        )
        summaries.append(
            {
                "subset": subset_name,
                "source_start": start,
                "source_end_exclusive": start + n,
                "num_frames": summary["num_frames"],
                "num_errors": summary["num_errors"],
                "dataset_json": f"{subset_name}/dataset.json",
            }
        )

    root_index = {
        "name": "apt0_rgbd_slam_splits",
        "source": str(args.src),
        "source_total_frames": total_frames,
        "frames_per_subset": n,
        "num_subsets": len(summaries),
        "selection": "contiguous windows at start/middle/end for SLAM continuity",
        "sensor": info.get("m_sensorName", "StructureSensor"),
        "depth_scale_to_meters": depth_scale,
        "intrinsics": {
            "color_matrix_3x3": color_K.tolist(),
            "depth_matrix_3x3": depth_K.tolist(),
        },
        "subsets": summaries,
    }
    (args.dst / "dataset.json").write_text(json.dumps(root_index, indent=2) + "\n")
    print(f"Wrote {len(summaries)} subsets to {args.dst}")
    for s in summaries:
        print(
            f"  {s['subset']}: frames {s['source_start']}..{s['source_end_exclusive']-1} "
            f"({s['num_frames']} ok, {s['num_errors']} err)"
        )


if __name__ == "__main__":
    main()
