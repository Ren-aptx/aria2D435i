"""MediaPipe detection plus robust aligned-depth hand lifting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .geometry import rotation_angle


# MP_TO_ARIA[aria index] = MediaPipe index. Palm center is synthesized.
MP_TO_ARIA = [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


@dataclass
class DetectedHand:
    side: str
    confidence: float
    landmarks_2d: np.ndarray
    world_landmarks: Optional[np.ndarray] = None


def remap_mediapipe_to_aria(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.shape[0] != 21:
        raise ValueError(f"expected 21 MediaPipe landmarks, got {points.shape}")
    aria = np.empty((21,) + points.shape[1:], dtype=points.dtype)
    aria[:20] = points[MP_TO_ARIA]
    aria[20] = (points[0] + points[5] + points[9]) / 3.0
    return aria


def patch_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int = 3,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.0,
    min_valid_pixels: int = 4,
    continuity_m: float = 0.06,
) -> Optional[float]:
    """Return a continuity-filtered median from a square depth neighborhood."""
    height, width = depth_m.shape[:2]
    x, y = int(round(float(u))), int(round(float(v)))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    valid = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    valid = valid[np.isfinite(valid) & (valid >= min_depth_m) & (valid <= max_depth_m)]
    if valid.size < min_valid_pixels:
        return None
    median = float(np.median(valid))
    continuous = valid[np.abs(valid - median) <= continuity_m]
    if continuous.size < min_valid_pixels:
        return None
    return float(np.median(continuous))


def visual_depth_fallback(
    points_2d: np.ndarray,
    world_landmarks: Optional[np.ndarray],
    intrinsics: np.ndarray,
    known_palm_m: float = 0.085,
) -> Optional[np.ndarray]:
    """Original HumanEgo hand-size estimate, retained only as a depth fallback."""
    wrist, middle_mcp = points_2d[0], points_2d[9]
    pixel_distance = float(np.linalg.norm(middle_mcp - wrist))
    if pixel_distance < 5.0:
        return None
    palm_m = known_palm_m
    if world_landmarks is not None:
        measured = float(np.linalg.norm(world_landmarks[9] - world_landmarks[0]))
        if 0.03 <= measured <= 0.15:
            palm_m = measured
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    z = 0.5 * (fx + fy) * palm_m / pixel_distance
    if not 0.05 <= z <= 3.0:
        return None
    wrist_3d = np.array(
        [(wrist[0] - cx) * z / fx, (wrist[1] - cy) * z / fy, z], dtype=np.float64
    )
    if world_landmarks is not None:
        return wrist_3d + (world_landmarks - world_landmarks[0])
    result = np.empty((21, 3), dtype=np.float64)
    result[:, 2] = z
    result[:, 0] = (points_2d[:, 0] - cx) * z / fx
    result[:, 1] = (points_2d[:, 1] - cy) * z / fy
    return result


def recover_keypoints_rgbd(
    points_2d: np.ndarray,
    world_landmarks: Optional[np.ndarray],
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    previous_3d: Optional[np.ndarray] = None,
    patch_radius: int = 3,
    min_valid_pixels: int = 4,
) -> Optional[np.ndarray]:
    """Lift 21 landmarks, using depth first and visual/temporal values only for holes."""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.shape != (21, 2):
        raise ValueError(f"expected landmarks shape (21, 2), got {points_2d.shape}")
    if world_landmarks is not None:
        world_landmarks = np.asarray(world_landmarks, dtype=np.float64)
        if world_landmarks.shape != (21, 3):
            raise ValueError("world_landmarks must have shape (21, 3)")
    fallback = visual_depth_fallback(points_2d, world_landmarks, intrinsics)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    result = np.full((21, 3), np.nan, dtype=np.float64)
    depth_hits = 0
    for index, (u, v) in enumerate(points_2d):
        z = patch_depth(
            depth_m, u, v, radius=patch_radius, min_valid_pixels=min_valid_pixels
        )
        if z is not None:
            result[index] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
            depth_hits += 1
        elif previous_3d is not None and np.all(np.isfinite(previous_3d[index])):
            old = previous_3d[index]
            if old[2] > 0:
                old_uv = np.array([old[0] / old[2] * fx + cx, old[1] / old[2] * fy + cy])
                if np.linalg.norm(old_uv - points_2d[index]) <= 20.0:
                    z = float(old[2])
                    result[index] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
        if not np.all(np.isfinite(result[index])) and fallback is not None:
            result[index] = fallback[index]

    # At least a few joints must have measured depth; otherwise the result is visual-only
    # and is allowed only when the fallback itself is valid.
    if depth_hits == 0 and fallback is None:
        return None
    if not np.all(np.isfinite(result)) or np.any(result[:, 2] <= 0):
        return None
    return result


def _normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-7 else vector / norm


def hand_frame(points_world_aria: np.ndarray, previous: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Construct the stable HumanEgo pinch frame from wrist/MCP landmarks."""
    thumb_base, index_base = points_world_aria[6], points_world_aria[8]
    wrist = points_world_aria[5]
    x_axis = _normalize(index_base - thumb_base)
    if x_axis is None:
        return previous
    arm = 0.5 * (thumb_base + index_base) - wrist
    y_axis = _normalize(arm - float(np.dot(arm, x_axis)) * x_axis)
    if y_axis is None:
        return previous
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return previous
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if previous is not None and float(np.dot(previous[:, 0], rotation[:, 0])) < 0:
        rotation[:, 0] *= -1
        rotation[:, 1] *= -1
        rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
    return rotation


class MediaPipeDetector:
    """Optional detector kept separate so conversion also works with supplied landmarks."""

    def __init__(self, max_num_hands: int = 2, confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe is not installed; install the 'hands' extra or use --landmarks"
            ) from error
        if not hasattr(mp, "solutions"):
            raise RuntimeError("this MediaPipe build does not provide mp.solutions.hands")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )

    def detect(self, image_bgr: np.ndarray) -> List[DetectedHand]:
        height, width = image_bgr.shape[:2]
        result = self._hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        detected: List[DetectedHand] = []
        if not result.multi_hand_landmarks or not result.multi_handedness:
            return detected
        worlds = result.multi_hand_world_landmarks or [None] * len(result.multi_hand_landmarks)
        for landmarks, world, handedness in zip(
            result.multi_hand_landmarks, worlds, result.multi_handedness
        ):
            category = handedness.classification[0]
            points_2d = np.array(
                [[point.x * width, point.y * height] for point in landmarks.landmark],
                dtype=np.float64,
            )
            world_points = None if world is None else np.array(
                [[point.x, point.y, point.z] for point in world.landmark], dtype=np.float64
            )
            detected.append(DetectedHand(
                side=category.label.lower(), confidence=float(category.score),
                landmarks_2d=points_2d, world_landmarks=world_points,
            ))
        return detected

    def close(self) -> None:
        self._hands.close()


class HandProcessor:
    def __init__(
        self,
        close_ratio: float = 0.55,
        open_ratio: float = 0.72,
        smooth_window: int = 5,
        patch_radius: int = 3,
        min_valid_pixels: int = 4,
    ):
        if close_ratio >= open_ratio:
            raise ValueError("close_ratio must be smaller than open_ratio")
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.smooth_window = max(1, smooth_window)
        self.patch_radius = patch_radius
        self.min_valid_pixels = min_valid_pixels
        self._history: Dict[str, Dict] = {}

    def process(
        self,
        detections: Iterable[DetectedHand],
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        color_to_world: np.ndarray,
        timestamp_s: float,
    ) -> Dict[str, Optional[Dict]]:
        output: Dict[str, Optional[Dict]] = {"hand_r": None, "hand_l": None}
        for detection in detections:
            side = detection.side.lower()
            if side not in {"left", "right"}:
                continue
            history = self._history.setdefault(side, {
                "points_cam": None, "rotation": None, "timestamp_s": None,
                "midpoint": None, "grasp": False, "votes": deque(maxlen=self.smooth_window),
            })
            points_mp_cam = recover_keypoints_rgbd(
                detection.landmarks_2d, detection.world_landmarks, depth_m, intrinsics,
                previous_3d=history["points_cam"], patch_radius=self.patch_radius,
                min_valid_pixels=self.min_valid_pixels,
            )
            if points_mp_cam is None:
                continue
            points_cam = remap_mediapipe_to_aria(points_mp_cam)
            points_2d = remap_mediapipe_to_aria(
                np.column_stack([detection.landmarks_2d, np.zeros(21)])
            )[:, :2]
            points_world = (
                color_to_world[:3, :3] @ points_cam.T
                + color_to_world[:3, 3:4]
            ).T
            rotation = hand_frame(points_world, history["rotation"])
            if rotation is None:
                continue

            midpoint = 0.5 * (points_world[0] + points_world[1])
            palm_size = float(np.linalg.norm(points_world[11] - points_world[5]))
            pinch_distance = float(np.linalg.norm(points_world[0] - points_world[1]))
            ratio = pinch_distance / palm_size if palm_size > 0.01 else np.inf
            raw_grasp = ratio < (self.open_ratio if history["grasp"] else self.close_ratio)
            history["votes"].append(bool(raw_grasp))
            grasp = sum(history["votes"]) * 2 >= len(history["votes"])

            linear_velocity = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)
            if history["timestamp_s"] is not None and timestamp_s > history["timestamp_s"]:
                dt = timestamp_s - history["timestamp_s"]
                linear_velocity = (midpoint - history["midpoint"]) / dt
                relative_rotation = history["rotation"].T @ rotation
                angle = rotation_angle(relative_rotation)
                if angle > 1e-9:
                    skew = np.array([
                        relative_rotation[2, 1] - relative_rotation[1, 2],
                        relative_rotation[0, 2] - relative_rotation[2, 0],
                        relative_rotation[1, 0] - relative_rotation[0, 1],
                    ])
                    axis = skew / (2.0 * np.sin(angle) + 1e-12)
                    angular_velocity = axis * angle / dt

            midpoint_pose = np.eye(4, dtype=np.float64)
            midpoint_pose[:3, :3] = rotation
            midpoint_pose[:3, 3] = midpoint
            wrist_pose_cam = np.eye(4, dtype=np.float64)
            wrist_pose_cam[:3, :3] = color_to_world[:3, :3].T @ rotation
            wrist_pose_cam[:3, 3] = points_cam[5]
            wrist_pose_world = np.eye(4, dtype=np.float64)
            wrist_pose_world[:3, :3] = rotation
            wrist_pose_world[:3, 3] = points_world[5]

            def list_of(value: np.ndarray):
                return np.asarray(value).tolist()

            hand = {
                "d2c": np.eye(4).tolist(),
                "c2w": color_to_world.tolist(),
                "confidence": float(detection.confidence),
                "grasp_state": int(grasp),
                "grasp_ratio": None if not np.isfinite(ratio) else float(ratio),
                "depth_keypoints_valid": int(sum(
                    patch_depth(depth_m, *point, radius=self.patch_radius,
                                min_valid_pixels=self.min_valid_pixels) is not None
                    for point in detection.landmarks_2d
                )),
                "wrist_pose": list_of(wrist_pose_cam),
                "palm_pose": list_of(wrist_pose_cam),
                "kpts_3d": list_of(points_cam),
                "kpts_2d": list_of(points_2d),
                "joint_angles": {},
                "wrist_pose_raw_world": list_of(wrist_pose_world),
                "wrist_pose_opt_world": list_of(wrist_pose_world),
                "wrist_lin_vel_raw_world": list_of(linear_velocity),
                "wrist_ang_vel_raw_world": list_of(angular_velocity),
                "wrist_lin_vel_opt_world": list_of(linear_velocity),
                "wrist_ang_vel_opt_world": list_of(angular_velocity),
                "index_translation_raw_world": list_of(points_world[1]),
                "index_translation_opt_world": list_of(points_world[1]),
                "thumb_translation_raw_world": list_of(points_world[0]),
                "thumb_translation_opt_world": list_of(points_world[0]),
                "midpoint_pose_raw_world": list_of(midpoint_pose),
                "midpoint_pose_opt_world": list_of(midpoint_pose),
                "midpoint_translation_raw_world": list_of(midpoint),
                "midpoint_orientation_raw_world": list_of(rotation.reshape(-1)),
                "midpoint_translation_opt_world": list_of(midpoint),
                "midpoint_orientation_opt_world": list_of(rotation.reshape(-1)),
                "midpoint_lin_vel_raw_world": list_of(linear_velocity),
                "midpoint_ang_vel_raw_world": list_of(angular_velocity),
                "midpoint_lin_vel_opt_world": list_of(linear_velocity),
                "midpoint_ang_vel_opt_world": list_of(angular_velocity),
                "distance_midpoint2wrist_raw_world": float(np.linalg.norm(midpoint - points_world[5])),
                "distance_midpoint2wrist_opt_world": float(np.linalg.norm(midpoint - points_world[5])),
            }
            output["hand_r" if side == "right" else "hand_l"] = hand
            history.update({
                "points_cam": points_mp_cam, "rotation": rotation,
                "timestamp_s": timestamp_s, "midpoint": midpoint, "grasp": bool(grasp),
            })
        return output
