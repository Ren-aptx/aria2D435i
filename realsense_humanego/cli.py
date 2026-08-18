"""Command-line entry point for offline HumanEgo export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .exporter import ExportConfig, export_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realsense-humanego",
        description="Convert a native D435i RGB-D capture to HumanEgo base files",
    )
    parser.add_argument("--session", type=Path, required=True,
                        help="session root containing raw/")
    parser.add_argument("--trajectory", type=Path,
                        help="SLAM mode only: override raw/trajectory_rgb.txt")
    camera = parser.add_mutually_exclusive_group()
    camera.add_argument(
        "--fixed-camera", dest="fixed_camera", action="store_true",
        help="use one constant camera-to-world transform (default)",
    )
    camera.add_argument(
        "--slam-camera", dest="fixed_camera", action="store_false",
        help="use and interpolate an ORB-SLAM3 trajectory",
    )
    parser.set_defaults(fixed_camera=True)
    parser.add_argument(
        "--fixed-c2w", type=Path, metavar="JSON",
        help="fixed mode: JSON 4x4 c2w matrix or an object containing a c2w field",
    )
    parser.add_argument("--max-pose-gap", type=float, default=0.20, metavar="SECONDS",
                        help="do not interpolate across a larger tracking gap (default: 0.20)")
    parser.add_argument("--hands", choices=["auto", "none", "mediapipe", "landmarks"],
                        default="auto", help="hand observation source (default: auto)")
    parser.add_argument("--landmarks", type=Path,
                        help="directory of per-source-frame MediaPipe landmark JSON files")
    parser.add_argument(
        "--grasp-close-ratio", type=float, default=0.85,
        help="close when 2D thumb-index/palm ratio falls below this (default: 0.85)",
    )
    parser.add_argument(
        "--grasp-open-ratio", type=float, default=1.00,
        help="re-open when the ratio rises above this (default: 1.00)",
    )
    parser.add_argument("--grasp-window", type=int, default=5)
    parser.add_argument("--depth-patch-radius", type=int, default=3,
                        help="3 gives the recommended 7x7 neighborhood")
    parser.add_argument("--min-depth-pixels", type=int, default=4)
    parser.add_argument("--min-depth-joints", type=int, default=8)
    parser.add_argument("--min-palm-depth-joints", type=int, default=3)
    parser.add_argument("--hand-confidence", type=float, default=0.3)
    parser.add_argument("--hand-min-segment", type=int, default=6)
    parser.add_argument("--hand-interp-gap", type=int, default=10)
    parser.add_argument("--finish-frames", type=int, default=60)
    parser.add_argument(
        "--manip-start", type=int,
        help="fixed mode: first manipulation frame (inclusive, exported-frame index)",
    )
    parser.add_argument(
        "--manip-end", type=int,
        help="fixed mode: last manipulation frame (inclusive, exported-frame index)",
    )
    parser.add_argument(
        "--finished-start", type=int,
        help="fixed mode: first FINISHED frame; defaults to the final --finish-frames",
    )
    return parser


def _load_fixed_c2w(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    matrix = document.get("c2w") if isinstance(document, dict) else document
    return np.asarray(matrix, dtype=np.float64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExportConfig(
        session=args.session,
        trajectory=args.trajectory,
        fixed_camera=args.fixed_camera,
        fixed_c2w=_load_fixed_c2w(args.fixed_c2w),
        max_pose_gap_s=args.max_pose_gap,
        hand_mode=args.hands,
        landmarks_dir=args.landmarks,
        close_ratio=args.grasp_close_ratio,
        open_ratio=args.grasp_open_ratio,
        grasp_smooth_window=args.grasp_window,
        depth_patch_radius=args.depth_patch_radius,
        min_valid_depth_pixels=args.min_depth_pixels,
        min_depth_joints=args.min_depth_joints,
        min_palm_depth_joints=args.min_palm_depth_joints,
        hand_confidence_threshold=args.hand_confidence,
        hand_min_segment_frames=args.hand_min_segment,
        hand_interp_max_gap=args.hand_interp_gap,
        finish_frames=args.finish_frames,
        manip_start=args.manip_start,
        manip_end=args.manip_end,
        finished_start=args.finished_start,
    )
    manifest = export_session(config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
