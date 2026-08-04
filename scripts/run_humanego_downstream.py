#!/usr/bin/env python3
"""Resume HumanEgo after this bridge has exported the compatible base files.

This deliberately bypasses HumanEgo's VRS/MPS initialization. It does not patch or
overwrite a possibly dirty HumanEgo checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os


STAGES = (
    "dinosam", "kptsselector", "cotracker", "camtriangulator",
    "lama", "visualkpts", "datasetgen",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run HumanEgo semantic/object preprocessing on exported D435i base data"
    )
    result.add_argument("--humanego", type=Path, default=Path("/home/tenda/HumanEgo"))
    result.add_argument("--session", type=Path, required=True)
    result.add_argument("--task", required=True)
    result.add_argument("--cfg", type=Path,
                        default=Path("cfg/preprocess/base/Preprocess.yaml"))
    result.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    result.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    result.add_argument(
        "--object-pose", choices=["rgbd", "triangulation"], default="rgbd",
        help="RGB-D fusion (default) or original CoTracker triangulation",
    )
    result.add_argument("--video", action="store_true", help="export HumanEgo diagnostics")
    result.add_argument("--gif", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    humanego = args.humanego.resolve()
    session = args.session.resolve()
    os.chdir(humanego)
    cfg = args.cfg if args.cfg.is_absolute() else humanego / args.cfg
    if not humanego.is_dir():
        raise FileNotFoundError(humanego)
    all_data = session / "preprocess" / "all_data"
    frame_dirs = sorted(
        path for path in all_data.iterdir()
        if path.is_dir() and path.name.isdigit()
    ) if all_data.is_dir() else []
    if not frame_dirs:
        raise FileNotFoundError(
            f"no exported frames in {all_data}; run realsense-humanego first"
        )
    required = ("aria_cam_rgb.json", "aria_hands.json", "aria_slam.json", "aria_phases.json")
    missing = [name for name in required if not (frame_dirs[0] / name).is_file()]
    if missing:
        raise FileNotFoundError(f"first frame is missing compatibility files: {missing}")

    bridge_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(bridge_root))
    from realsense_humanego.object_pose import generate_rgbd_object_poses

    sys.path.insert(0, str(humanego))
    from preprocess.CoTrackerOffline import reset_cotracker_offline
    from preprocess.Preprocess import Preprocess
    from preprocess.VisualKpts import reset_visualkpts
    from utils.utils_io import load_cfg_dynamic_task

    # Construct only the downstream part of the orchestrator. Calling its constructor
    # would initialize Project Aria providers and require sample.vrs/MPS files.
    engine = Preprocess.__new__(Preprocess)
    engine.mps_path = str(session)
    engine.cfg_path = str(cfg)
    engine.task = args.task
    engine.export_video = args.video
    engine.export_gif = args.gif
    engine.backend = "aria"
    engine.sensor_backend = "aria"  # exported filenames intentionally use legacy names
    engine.cfg = load_cfg_dynamic_task(str(cfg), str(session), args.task)
    engine.num_total_frames = len(frame_dirs)
    engine.start_idx = 0
    engine.end_idx = len(frame_dirs) - 1

    reset_cotracker_offline()
    reset_visualkpts()
    engine.preprocess_indices()

    start, end = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if start > end:
        raise ValueError("--from-stage must not come after --to-stage")
    methods = {
        "dinosam": engine.preprocess_dinosam,
        "kptsselector": engine.preprocess_kptsselector,
        "cotracker": engine.preprocess_cotracker,
        "camtriangulator": engine.preprocess_camtriangulator,
        "lama": engine.preprocess_lama,
        "visualkpts": engine.preprocess_visualkpts,
        "datasetgen": engine.preprocess_datasetgen,
    }
    rgbd_pose_done = False
    for stage in STAGES[start:end + 1]:
        if args.object_pose == "rgbd" and stage in {
            "kptsselector", "cotracker", "camtriangulator"
        }:
            if not rgbd_pose_done:
                print("[RealSense bridge] HumanEgo stage: rgbd_object_pose")
                generate_rgbd_object_poses(
                    session, [Path(path) for path in engine.object_centric_image_list]
                )
                rgbd_pose_done = True
            continue
        print(f"[RealSense bridge] HumanEgo stage: {stage}")
        methods[stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
