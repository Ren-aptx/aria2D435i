"""Convert a native capture into the file contract consumed by HumanEgo."""

from __future__ import annotations

import csv
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import (
    interpolate_pose,
    load_tum_trajectory,
    rotation_angle,
    rotation_to_rpy_zyx,
)
from .hands import (
    DetectedHand,
    HandProcessor,
    MediaPipeDetector,
    optimize_hand_sequence,
)


@dataclass
class ExportConfig:
    session: Path
    trajectory: Optional[Path] = None
    max_pose_gap_s: float = 0.20
    hand_mode: str = "auto"
    landmarks_dir: Optional[Path] = None
    close_ratio: float = 0.55
    open_ratio: float = 0.72
    grasp_smooth_window: int = 5
    depth_patch_radius: int = 3
    min_valid_depth_pixels: int = 4
    min_depth_joints: int = 8
    min_palm_depth_joints: int = 3
    hand_confidence_threshold: float = 0.3
    hand_min_segment_frames: int = 6
    hand_interp_max_gap: int = 10
    hand_smooth_window: int = 21
    hand_smooth_polyorder: int = 2
    hand_ema_alpha: float = 0.15
    hand_motion_mps: float = 0.15
    hand_stable_frames: int = 5
    hand_transition_frames: int = 10
    finish_frames: int = 60
    stop_linear_mps: float = 0.04
    stop_angular_rps: float = 0.15
    rotate_angular_rps: float = 0.35
    rotate_max_linear_mps: float = 0.15


@dataclass
class SourceFrame:
    source_idx: int
    rgb_timestamp_ns: int
    depth_timestamp_ns: int
    rgb_path: Path
    aligned_depth_path: Path


@dataclass
class ExportFrame:
    idx: int
    source: SourceFrame
    c2w: np.ndarray
    t_world: np.ndarray
    rpy_rad: np.ndarray
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    yaw_unwrapped_rad: float = 0.0


def _raw_directory(session: Path) -> Path:
    raw = session / "raw"
    return raw if raw.is_dir() else session


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_calibration(raw: Path) -> Dict:
    with _require_file(raw / "calibration.json").open("r", encoding="utf-8") as stream:
        calibration = json.load(stream)
    for key in ("depth_scale_m", "color", "T_color_to_left_ir"):
        if key not in calibration:
            raise KeyError(f"calibration.json is missing {key!r}")
    return calibration


def _source_path(raw: Path, value: str, fallback_directory: str, index: int) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else raw / path
    return raw / fallback_directory / f"{index:06d}.png"


def _load_source_frames(raw: Path) -> List[SourceFrame]:
    result: List[SourceFrame] = []
    with _require_file(raw / "timestamps.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source_idx = int(row["idx"])
            rgb_path = _source_path(raw, row.get("rgb_file", ""), "rgb", source_idx)
            depth_path = _source_path(
                raw, row.get("aligned_depth_file", ""), "aligned_depth", source_idx
            )
            if not rgb_path.is_file() or not depth_path.is_file():
                warnings.warn(f"skipping source frame {source_idx}: RGB or aligned depth is missing")
                continue
            result.append(SourceFrame(
                source_idx=source_idx,
                rgb_timestamp_ns=int(row["rgb_timestamp_ns"]),
                depth_timestamp_ns=int(row.get("aligned_depth_timestamp_ns")
                                       or row.get("depth_timestamp_ns") or 0),
                rgb_path=rgb_path,
                aligned_depth_path=depth_path,
            ))
    if not result:
        raise ValueError(f"no usable rows in {raw / 'timestamps.csv'}")
    result.sort(key=lambda frame: frame.rgb_timestamp_ns)
    return result


def _camera_matrix(calibration: Dict) -> np.ndarray:
    color = calibration["color"]
    return np.array([
        [color["fx"], 0.0, color["cx"]],
        [0.0, color["fy"], color["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _json_dump(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def _load_landmarks(path: Path, image_shape: Tuple[int, int]) -> List[DetectedHand]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    entries = document.get("hands", document) if isinstance(document, dict) else document
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list or {{'hands': [...]}}")
    height, width = image_shape
    detected: List[DetectedHand] = []
    for entry in entries:
        points = np.asarray(
            entry.get("landmarks_2d", entry.get("landmarks")), dtype=np.float64
        )
        if points.shape != (21, 2):
            raise ValueError(f"{path}: each landmarks_2d must have shape (21, 2)")
        if bool(entry.get("normalized", False)) or float(np.nanmax(np.abs(points))) <= 2.0:
            points = points * np.array([width, height], dtype=np.float64)
        world = entry.get("world_landmarks")
        detected.append(DetectedHand(
            side=str(entry["side"]).lower(),
            confidence=float(entry.get("confidence", 1.0)),
            landmarks_2d=points,
            world_landmarks=None if world is None else np.asarray(world, dtype=np.float64),
        ))
    return detected


def _landmark_file(directory: Path, source_idx: int) -> Optional[Path]:
    candidates = [directory / f"{source_idx:06d}.json", directory / f"{source_idx:05d}.json"]
    return next((path for path in candidates if path.is_file()), None)


def _phase_windows(modes: Iterable[int]) -> Dict[str, List[List[int]]]:
    values = list(modes)
    windows: Dict[str, List[List[int]]] = {str(mode): [] for mode in range(5)}
    if not values:
        return windows
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            windows[str(values[start])].append([start, index - 1])
            start = index
    return windows


def _calculate_kinematics(frames: List[ExportFrame]) -> None:
    if not frames:
        return
    raw_yaws = []
    for index, frame in enumerate(frames):
        raw_yaws.append(frame.rpy_rad[2])
        if index == 0:
            continue
        previous = frames[index - 1]
        dt = (frame.source.rgb_timestamp_ns - previous.source.rgb_timestamp_ns) * 1e-9
        if dt <= 0:
            continue
        frame.linear_speed = float(np.linalg.norm(frame.t_world - previous.t_world) / dt)
        frame.angular_speed = rotation_angle(previous.c2w[:3, :3].T @ frame.c2w[:3, :3]) / dt
    unwrapped = np.unwrap(np.asarray(raw_yaws))
    for frame, yaw in zip(frames, unwrapped):
        frame.yaw_unwrapped_rad = float(yaw)


def _phase_modes(
    frames: List[ExportFrame],
    config: ExportConfig,
    hand_documents: Optional[List[Dict]] = None,
) -> List[int]:
    stopped_flags = np.asarray([
        frame.linear_speed < config.stop_linear_mps
        and frame.angular_speed < config.stop_angular_rps
        for frame in frames
    ], dtype=bool)
    modes: List[int] = []
    for frame, stopped in zip(frames, stopped_flags):
        if stopped:
            modes.append(0)
        elif (frame.angular_speed >= config.rotate_angular_rps
              and frame.linear_speed <= config.rotate_max_linear_mps):
            modes.append(2)
        else:
            modes.append(1)

    # Match HumanEgo's hand-kinematic cleanup: a camera stop becomes manipulation
    # only after a hand has settled; reach/withdrawal portions are transitions.
    if hand_documents is not None:
        for start, end in _continuous_true_runs(stopped_flags):
            speeds = np.full(end - start, np.inf, dtype=np.float64)
            for local_index, document in enumerate(hand_documents[start:end]):
                active = []
                for side in ("hand_r", "hand_l"):
                    hand = document.get(side)
                    velocity = None if hand is None else hand.get(
                        "midpoint_lin_vel_opt_world"
                    )
                    if velocity is not None:
                        active.append(float(np.linalg.norm(velocity)))
                if active:
                    speeds[local_index] = max(active)
            stable = speeds < config.hand_motion_mps
            width = max(1, config.hand_stable_frames)
            stable_windows = [
                index for index in range(0, max(0, len(stable) - width + 1))
                if bool(np.all(stable[index:index + width]))
            ]
            if not stable_windows:
                modes[start:end] = [3] * (end - start)
                continue
            first_stable = stable_windows[0]
            last_stable = stable_windows[-1] + width - 1
            entry_end = min(end, start + first_stable + config.hand_transition_frames)
            exit_start = max(start, start + last_stable + 1 - config.hand_transition_frames)
            for index in range(start, entry_end):
                modes[index] = 3
            for index in range(exit_start, end):
                modes[index] = 3

    # Match HumanEgo's terminal convention using the actual camera stop mask.
    tail_start = len(modes)
    while tail_start > 0 and stopped_flags[tail_start - 1]:
        tail_start -= 1
    finish_start = max(tail_start, len(modes) - max(0, config.finish_frames))
    for index in range(finish_start, len(modes)):
        modes[index] = 4
    return modes


def _continuous_true_runs(values: np.ndarray) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    start = None
    for index in range(len(values) + 1):
        active = index < len(values) and bool(values[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            result.append((start, index))
            start = None
    return result


def _select_hand_source(config: ExportConfig):
    mode = config.hand_mode.lower()
    if mode not in {"auto", "none", "mediapipe", "landmarks"}:
        raise ValueError("hand_mode must be auto, none, mediapipe, or landmarks")
    if mode == "landmarks" and config.landmarks_dir is None:
        raise ValueError("hand_mode=landmarks requires landmarks_dir")
    if config.landmarks_dir is not None and mode in {"auto", "landmarks"}:
        return "landmarks", None
    if mode == "none":
        return "none", None
    try:
        return "mediapipe", MediaPipeDetector()
    except RuntimeError:
        if mode == "mediapipe":
            raise
        warnings.warn("MediaPipe is unavailable; exporting explicit null hand observations")
        return "none", None


def export_session(config: ExportConfig) -> Dict:
    session = Path(config.session).resolve()
    raw = _raw_directory(session)
    calibration = _load_calibration(raw)
    source_frames = _load_source_frames(raw)
    trajectory_path = Path(config.trajectory) if config.trajectory else raw / "trajectory_rgb.txt"
    poses = load_tum_trajectory(_require_file(trajectory_path))
    if not poses:
        raise ValueError(f"trajectory contains no valid poses: {trajectory_path}")

    frames: List[ExportFrame] = []
    dropped: List[Dict] = []
    for source in source_frames:
        c2w = interpolate_pose(poses, source.rgb_timestamp_ns * 1e-9, config.max_pose_gap_s)
        if c2w is None:
            dropped.append({"source_idx": source.source_idx, "reason": "invalid_or_missing_pose"})
            continue
        frames.append(ExportFrame(
            idx=len(frames), source=source, c2w=c2w,
            t_world=c2w[:3, 3].copy(), rpy_rad=rotation_to_rpy_zyx(c2w[:3, :3]),
        ))
    if not frames:
        raise ValueError(
            "no RGB frame has a valid bracketing SLAM pose; check time domains and max_pose_gap_s"
        )
    _calculate_kinematics(frames)

    intrinsics = _camera_matrix(calibration)
    color = calibration["color"]
    distortion = np.asarray(color.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    color_to_left = np.asarray(calibration["T_color_to_left_ir"], dtype=np.float64)
    if color_to_left.shape != (4, 4):
        raise ValueError("T_color_to_left_ir must have shape (4, 4)")
    depth_scale = float(calibration["depth_scale_m"])
    timestamps = np.asarray([frame.source.rgb_timestamp_ns for frame in frames], dtype=np.int64)
    median_dt = float(np.median(np.diff(timestamps))) * 1e-9 if len(frames) > 1 else 1 / 30
    fps = int(round(1.0 / median_dt)) if median_dt > 0 else 30
    fps = max(1, fps)
    fov = math.degrees(2.0 * math.atan(float(color["height"]) / (2.0 * intrinsics[1, 1])))
    hand_source, detector = _select_hand_source(config)
    hand_processor = HandProcessor(
        close_ratio=config.close_ratio,
        open_ratio=config.open_ratio,
        smooth_window=config.grasp_smooth_window,
        patch_radius=config.depth_patch_radius,
        min_valid_pixels=config.min_valid_depth_pixels,
        min_depth_joints=config.min_depth_joints,
        min_palm_depth_joints=config.min_palm_depth_joints,
    )

    preprocess = session / "preprocess"
    all_data = preprocess / "all_data"
    all_data.mkdir(parents=True, exist_ok=True)
    first_t = frames[0].t_world
    first_rpy_deg = np.degrees(frames[0].rpy_rad)
    mode_names = {0: "STOP", 1: "FORWARD", 2: "ROTATE", 3: "TRANSITION", 4: "FINISHED"}
    hand_documents: List[Dict] = []

    try:
        for frame in frames:
            frame_dir = all_data / f"{frame.idx:05d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            rgb_target = frame_dir / "rgb.png"
            depth_target = frame_dir / "depth.png"
            shutil.copy2(frame.source.rgb_path, rgb_target)
            shutil.copy2(frame.source.aligned_depth_path, depth_target)

            left_to_world = frame.c2w @ np.linalg.inv(color_to_left)
            camera_document = {
                "idx": frame.idx,
                "source_idx": frame.source.source_idx,
                "source": "realsense_d435i",
                "ts": frame.source.rgb_timestamp_ns,
                "fov": fov,
                "h": int(color["height"]),
                "w": int(color["width"]),
                "k": intrinsics.tolist(),
                "d": distortion.tolist(),
                "c2w": frame.c2w.tolist(),
                "c2d": color_to_left.tolist(),
                "d2w": left_to_world.tolist(),
                "rgb_path": str(Path("preprocess/all_data") / f"{frame.idx:05d}" / "rgb.png"),
                "fps": fps,
                "pose_valid": True,
            }
            _json_dump(frame_dir / "aria_cam_rgb.json", camera_document)
            _json_dump(frame_dir / "depth_meta.json", {
                "idx": frame.idx,
                "source_idx": frame.source.source_idx,
                "ts": frame.source.depth_timestamp_ns,
                "depth_path": str(Path("preprocess/all_data") / f"{frame.idx:05d}" / "depth.png"),
                "aligned_to": "rgb",
                "scale_m": depth_scale,
                "invalid_value": 0,
            })

            rpy_deg = np.degrees(frame.rpy_rad)
            delta_rpy = rpy_deg - first_rpy_deg
            delta_rpy[2] = (delta_rpy[2] + 180.0) % 360.0 - 180.0
            slam_document = {
                "idx": frame.idx,
                "ts": frame.source.rgb_timestamp_ns,
                "t_world": frame.t_world.tolist(),
                "rpy_deg": rpy_deg.tolist(),
                "delta_t_world": (frame.t_world - first_t).tolist(),
                "delta_rpy_deg": delta_rpy.tolist(),
                "linear_speed_mps": frame.linear_speed,
                "angular_speed_rps": frame.angular_speed,
                "yaw_unwrapped_deg": math.degrees(frame.yaw_unwrapped_rad),
                "tracking_valid": True,
            }
            _json_dump(frame_dir / "aria_slam.json", slam_document)

            image = cv2.imread(str(rgb_target), cv2.IMREAD_COLOR)
            depth_raw = cv2.imread(str(depth_target), cv2.IMREAD_UNCHANGED)
            if image is None or depth_raw is None:
                raise ValueError(f"failed to read exported RGB/depth for frame {frame.idx}")
            depth_m = depth_raw.astype(np.float32) * depth_scale
            detections: List[DetectedHand] = []
            if hand_source == "mediapipe":
                detections = detector.detect(image)
            elif hand_source == "landmarks":
                landmark_path = _landmark_file(config.landmarks_dir, frame.source.source_idx)
                if landmark_path is not None:
                    detections = _load_landmarks(landmark_path, image.shape[:2])
            hands = hand_processor.process(
                detections, depth_m, intrinsics, frame.c2w,
                frame.source.rgb_timestamp_ns * 1e-9,
            )
            hand_documents.append({
                "idx": frame.idx,
                "ts": frame.source.rgb_timestamp_ns,
                "hand_r": hands["hand_r"],
                "hand_l": hands["hand_l"],
            })
    finally:
        if detector is not None:
            detector.close()

    optimization_stats = optimize_hand_sequence(
        hand_documents,
        timestamps.astype(np.float64) * 1e-9,
        confidence_threshold=config.hand_confidence_threshold,
        min_segment_frames=config.hand_min_segment_frames,
        fill_max_gap=config.hand_interp_max_gap,
        smooth_window=config.hand_smooth_window,
        smooth_polyorder=config.hand_smooth_polyorder,
        ema_alpha=config.hand_ema_alpha,
    )
    phases = _phase_modes(frames, config, hand_documents)
    hand_counts = {
        "right": sum(document["hand_r"] is not None for document in hand_documents),
        "left": sum(document["hand_l"] is not None for document in hand_documents),
    }
    for frame, mode, hands_document in zip(frames, phases, hand_documents):
        frame_dir = all_data / f"{frame.idx:05d}"
        _json_dump(frame_dir / "aria_hands.json", hands_document)
        camera_stopped = (
            frame.linear_speed < config.stop_linear_mps
            and frame.angular_speed < config.stop_angular_rps
        )
        _json_dump(frame_dir / "aria_phases.json", {
            "idx": frame.idx,
            "ts": frame.source.rgb_timestamp_ns,
            "stop": int(camera_stopped),
            "mode": mode,
            "mode_str": mode_names[mode],
            "linear_speed_mps": frame.linear_speed,
            "angular_speed_rps": frame.angular_speed,
            "yaw_unwrapped_deg": math.degrees(frame.yaw_unwrapped_rad),
        })

    _json_dump(preprocess / "aria_cam_rgb_config.json", {
        "total_frames": len(frames),
        "fps": fps,
        "first_ts": frames[0].source.rgb_timestamp_ns,
        "h": int(color["height"]),
        "w": int(color["width"]),
        "k": intrinsics.tolist(),
        "d": distortion.tolist(),
        "c2d": color_to_left.tolist(),
        "source": "realsense_d435i",
    })
    windows = _phase_windows(phases)
    _json_dump(preprocess / "aria_phases_results.json", {
        "total_frames": len(frames),
        "fps_median": fps,
        "duration_s": (timestamps[-1] - timestamps[0]) * 1e-9 if len(frames) > 1 else 0.0,
        "stage_window_check": {"windows": windows},
        "source": "realsense_d435i",
    })
    manifest = {
        "source": "realsense_d435i",
        "source_frames": len(source_frames),
        "exported_frames": len(frames),
        "dropped_frames": dropped,
        "max_pose_gap_s": config.max_pose_gap_s,
        "trajectory": str(trajectory_path.resolve()),
        "hand_source": hand_source,
        "hand_observations": hand_counts,
        "hand_optimization": optimization_stats,
    }
    _json_dump(preprocess / "realsense_manifest.json", manifest)
    return manifest
