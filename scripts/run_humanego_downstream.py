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

import cv2
import numpy as np


STAGES = (
    "dinosam", "kptsselector", "cotracker", "camtriangulator",
    "lama", "visualkpts", "datasetgen",
)

_RESULT_PRODUCERS = {
    "kptsselector_results.json": "kptsselector",
    "cotracker_results.json": "cotracker",
    "camtriangulator_results.json": "camtriangulator",
}


def clear_invalidated_results(session: Path, from_stage: str) -> list[Path]:
    """Remove cached results invalidated by a resumed upstream stage."""
    start = STAGES.index(from_stage)
    removed = []
    preprocess = Path(session) / "preprocess"
    for filename, producer in _RESULT_PRODUCERS.items():
        if start <= STAGES.index(producer):
            path = preprocess / filename
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


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
    videos = result.add_mutually_exclusive_group()
    videos.add_argument(
        "--video", dest="video", action="store_true",
        help="export aria_vis.mp4 and visualkpts_vis.mp4 (default)",
    )
    videos.add_argument(
        "--no-video", dest="video", action="store_false",
        help="skip the two final diagnostic videos",
    )
    result.set_defaults(video=True)
    result.add_argument("--gif", action="store_true")
    result.add_argument(
        "--keep-stage-cache", action="store_true",
        help="keep cached stage JSON even when rerunning an upstream stage",
    )
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    humanego = args.humanego.resolve()
    session = args.session.resolve()
    if not humanego.is_dir():
        raise FileNotFoundError(humanego)
    os.chdir(humanego)
    cfg = args.cfg if args.cfg.is_absolute() else humanego / args.cfg
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
    from realsense_humanego.visualization import export_official_videos

    sys.path.insert(0, str(humanego))
    from preprocess.CoTrackerOffline import reset_cotracker_offline
    from preprocess.Preprocess import Preprocess
    from preprocess.OrientAnything import (
        estimate_frame_pca1,
        estimate_frame_pca2,
        estimate_frame_vlm,
        get_crop_from_2d_kpts,
    )
    from preprocess.VisualKpts import reset_visualkpts
    from utils.utils_io import load_cfg_dynamic_task

    # Construct only the downstream part of the orchestrator. Calling its constructor
    # would initialize Project Aria providers and require sample.vrs/MPS files.
    engine = Preprocess.__new__(Preprocess)
    engine.mps_path = str(session)
    engine.cfg_path = str(cfg)
    engine.task = args.task
    # Final MP4s are streamed after all selected stages, avoiding the original
    # implementation's full-sequence RAM accumulation. Keep the old path only
    # when a GIF was explicitly requested.
    engine.export_video = args.gif
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
    if not args.keep_stage_cache:
        for path in clear_invalidated_results(session, args.from_stage):
            print(f"[RealSense bridge] removed stale result: {path}")

    def humanego_pose_estimator(
        method, points_cam, is_anchor, anchor_center_cam, image_path, mask_path
    ):
        common = {
            "is_anchor": is_anchor,
            "anchor_center_cam": anchor_center_cam,
        }
        if method == "pca1":
            return estimate_frame_pca1(pts_cam=points_cam, **common)
        if method == "pca2":
            return estimate_frame_pca2(pts_cam=points_cam, **common)
        if method == "vlm":
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise ValueError(f"cannot build VLM crop from {image_path} / {mask_path}")
            y, x = np.nonzero(mask > 0)
            if x.size == 0:
                raise ValueError(f"empty object mask: {mask_path}")
            crop = get_crop_from_2d_kpts(image, np.column_stack([x, y]))
            return estimate_frame_vlm(
                image=crop,
                t_cam=np.mean(points_cam, axis=0),
                do_rm_bkg=True,
                **common,
            )
        raise ValueError(f"unsupported HumanEgo pose_method: {method!r}")

    methods = {
        "dinosam": engine.preprocess_dinosam,
        "kptsselector": engine.preprocess_kptsselector,
        "cotracker": engine.preprocess_cotracker,
        "camtriangulator": engine.preprocess_camtriangulator,
        "lama": engine.preprocess_lama,
        "visualkpts": engine.preprocess_visualkpts,
        "datasetgen": engine.preprocess_datasetgen,
    }
    for stage in STAGES[start:end + 1]:
        if args.object_pose == "rgbd" and stage == "camtriangulator":
            print("[RealSense bridge] HumanEgo stage: rgbd_object_pose")
            camtriangulator_cfg = getattr(engine.cfg, "CamTriangulator", {})
            if hasattr(camtriangulator_cfg, "get"):
                pose_methods = camtriangulator_cfg.get("pose_method", "pca1")
            else:
                pose_methods = getattr(camtriangulator_cfg, "pose_method", "pca1")
            generate_rgbd_object_poses(
                session,
                [Path(path) for path in engine.object_centric_image_list],
                pose_methods=pose_methods,
                pose_estimator=humanego_pose_estimator,
            )
            continue
        print(f"[RealSense bridge] HumanEgo stage: {stage}")
        methods[stage]()
    if args.video:
        print("[RealSense bridge] exporting official diagnostic videos")
        try:
            videos = export_official_videos(session)
            for name, info in videos.items():
                print(f"[RealSense bridge] {name}: {info['path']} ({info['frames']} frames)")
        except FileNotFoundError as error:
            print(f"[RealSense bridge] video export incomplete: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
