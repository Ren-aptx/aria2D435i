"""Streaming diagnostic-video export for RealSense HumanEgo sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


# Aria ordering used by aria_hands.json (thumb/index/middle/ring/pinky tips
# first, wrist at 5, then each finger's inner joints).
_FULL_HAND_CONNECTIONS = (
    (5, 6), (6, 7), (7, 0),
    (5, 8), (8, 9), (9, 10), (10, 1),
    (5, 11), (11, 12), (12, 13), (13, 2),
    (5, 14), (14, 15), (15, 16), (16, 3),
    (5, 17), (17, 18), (18, 19), (19, 4),
    (8, 11), (11, 14), (14, 17),
)
_GRIPPER_CONNECTIONS = ((5, 6), (6, 0), (5, 8), (8, 1), (0, 1))


def _load_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _frame_directories(session: Path) -> list[Path]:
    root = Path(session) / "preprocess" / "all_data"
    if not root.is_dir():
        raise FileNotFoundError(root)
    frames = sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    if not frames:
        raise ValueError(f"no frame directories in {root}")
    return frames


def _session_fps(session: Path) -> float:
    config = _load_json(Path(session) / "preprocess" / "aria_cam_rgb_config.json")
    return max(1.0, float(config.get("fps", 30.0)))


def _point(points: np.ndarray, index: int, shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    value = points[index]
    if value.shape[0] < 2 or not np.all(np.isfinite(value[:2])):
        return None
    x, y = int(round(float(value[0]))), int(round(float(value[1])))
    height, width = shape
    return (x, y) if 0 <= x < width and 0 <= y < height else None


def _draw_hand(
    image: np.ndarray,
    hand: Dict,
    *,
    full_skeleton: bool,
) -> None:
    points = np.asarray(hand.get("kpts_2d", []), dtype=np.float64)
    if points.shape != (21, 2):
        return
    closed = int(hand.get("grasp_state", 0)) == 1
    line_color = (255, 80, 220) if closed else (0, 210, 255)
    point_color = (255, 230, 255) if closed else (50, 255, 255)
    connections = _FULL_HAND_CONNECTIONS if full_skeleton else _GRIPPER_CONNECTIONS
    shape = image.shape[:2]
    for first, second in connections:
        p1, p2 = _point(points, first, shape), _point(points, second, shape)
        if p1 is not None and p2 is not None:
            cv2.line(image, p1, p2, line_color, 2, cv2.LINE_AA)
    indices: Iterable[int] = range(21) if full_skeleton else (5, 6, 0, 8, 1)
    for index in indices:
        position = _point(points, index, shape)
        if position is not None:
            cv2.circle(image, position, 4, point_color, -1, cv2.LINE_AA)


def _draw_hud(image: np.ndarray, frame_index: int, hands: Dict, phase: Dict) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 62), (14, 18, 25), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
    hand = hands.get("hand_r") or hands.get("hand_l")
    if hand is None:
        state, state_color, ratio = "HAND: N/A", (150, 150, 150), None
    else:
        closed = int(hand.get("grasp_state", 0)) == 1
        state = "GRIPPER: CLOSED" if closed else "GRIPPER: OPEN"
        state_color = (255, 80, 220) if closed else (0, 210, 255)
        ratio = hand.get("grasp_ratio_2d", hand.get("grasp_ratio"))
    phase_name = str(phase.get("mode_str", "UNKNOWN"))
    cv2.putText(
        image, f"FRAME {frame_index:05d}  |  PHASE: {phase_name}", (14, 24),
        cv2.FONT_HERSHEY_DUPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA,
    )
    ratio_text = "" if ratio is None else f"  |  pinch/palm={float(ratio):.3f}"
    cv2.putText(
        image, state + ratio_text, (14, 50), cv2.FONT_HERSHEY_DUPLEX,
        0.62, state_color, 2, cv2.LINE_AA,
    )


def _open_writer(path: Path, fps: float, shape: Tuple[int, int]) -> cv2.VideoWriter:
    height, width = shape
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    return writer


def _write_image_sequence(
    frames: list[Path], image_name: str, output: Path, fps: float
) -> int:
    writer: Optional[cv2.VideoWriter] = None
    shape: Optional[Tuple[int, int]] = None
    written = 0
    try:
        for frame_dir in frames:
            image = cv2.imread(str(frame_dir / image_name), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if writer is None:
                shape = image.shape[:2]
                writer = _open_writer(output, fps, shape)
            if image.shape[:2] != shape:
                image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            writer.write(image)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    if written == 0:
        raise FileNotFoundError(f"no {image_name} frames available for {output}")
    return written


def export_official_videos(session: Path) -> Dict[str, Dict[str, object]]:
    """Create the two standard HumanEgo diagnostics without loading all frames in RAM.

    Outputs:
      * aria_vis.mp4: raw RGB + hand/grasp/phase diagnostics
      * visualkpts_vis.mp4: arm-inpainted RGB + virtual gripper/object keypoints
    """
    session = Path(session).resolve()
    frames = _frame_directories(session)
    fps = _session_fps(session)
    vis_dir = session / "preprocess" / "vis"
    aria_path = vis_dir / "aria_vis.mp4"
    visual_path = vis_dir / "visualkpts_vis.mp4"

    writer: Optional[cv2.VideoWriter] = None
    shape: Optional[Tuple[int, int]] = None
    aria_count = 0
    try:
        for frame_dir in frames:
            image = cv2.imread(str(frame_dir / "rgb.png"), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if writer is None:
                shape = image.shape[:2]
                writer = _open_writer(aria_path, fps, shape)
            hands = _load_json(frame_dir / "aria_hands.json")
            for side in ("hand_r", "hand_l"):
                hand = hands.get(side)
                if hand is not None:
                    _draw_hand(image, hand, full_skeleton=False)
            _draw_hud(image, int(frame_dir.name), hands, _load_json(frame_dir / "aria_phases.json"))
            writer.write(image)
            aria_count += 1
    finally:
        if writer is not None:
            writer.release()
    if aria_count == 0:
        raise FileNotFoundError(f"no rgb.png frames available for {aria_path}")

    visual_count = _write_image_sequence(
        frames, "rgb_WoArm_WArmObjKpts.png", visual_path, fps
    )
    return {
        "aria_vis": {"path": str(aria_path), "frames": aria_count, "fps": fps},
        "visualkpts_vis": {
            "path": str(visual_path), "frames": visual_count, "fps": fps,
        },
    }
