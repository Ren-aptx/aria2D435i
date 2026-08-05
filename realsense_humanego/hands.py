"""MediaPipe detection plus robust aligned-depth hand lifting."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .geometry import rotation_angle


# MP_TO_ARIA[aria index] = MediaPipe index. Palm center is synthesized.
MP_TO_ARIA = [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
REQUIRED_PALM_MP_INDICES = (0, 5, 9, 13, 17)


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
    continuity_m: float = 0.025,
) -> Optional[float]:
    """Select the center-guided or nearest coherent depth surface in a patch."""
    height, width = depth_m.shape[:2]
    x, y = int(round(float(u))), int(round(float(v)))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64)
    valid = patch.reshape(-1)
    valid = valid[np.isfinite(valid) & (valid >= min_depth_m) & (valid <= max_depth_m)]
    if valid.size < min_valid_pixels:
        return None

    if 0 <= x < width and 0 <= y < height:
        center = float(depth_m[y, x])
        if np.isfinite(center) and min_depth_m <= center <= max_depth_m:
            cluster = valid[np.abs(valid - center) <= continuity_m]
            if cluster.size >= min_valid_pixels:
                return float(np.median(cluster))

    ordered = np.sort(valid)
    groups = np.split(ordered, np.where(np.diff(ordered) > continuity_m)[0] + 1)
    stable = [group for group in groups if group.size >= min_valid_pixels]
    if not stable:
        return None
    return float(np.median(min(stable, key=lambda group: float(np.median(group)))))


def _similarity_align(
    source_all: np.ndarray, source_fit: np.ndarray, target_fit: np.ndarray
) -> Optional[np.ndarray]:
    """Align a full point set from paired anchors with an Umeyama similarity."""
    if len(source_fit) < 3:
        return None
    source_center = np.mean(source_fit, axis=0)
    target_center = np.mean(target_fit, axis=0)
    source_zero = source_fit - source_center
    target_zero = target_fit - target_center
    variance = float(np.sum(source_zero * source_zero))
    if variance < 1e-10:
        return None
    u, singular, vt = np.linalg.svd(source_zero.T @ target_zero)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or not 0.2 <= scale <= 5.0:
        return None
    translation = target_center - scale * (rotation @ source_center)
    return (scale * (rotation @ source_all.T)).T + translation


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
    previous_points_world: Optional[np.ndarray] = None,
    current_color_to_world: Optional[np.ndarray] = None,
    patch_radius: int = 3,
    min_valid_pixels: int = 4,
    min_depth_joints: int = 8,
    min_palm_depth_joints: int = 3,
    return_quality: bool = False,
):
    """Lift 21 landmarks, using depth first and visual/temporal values only for holes."""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    if points_2d.shape != (21, 2):
        raise ValueError(f"expected landmarks shape (21, 2), got {points_2d.shape}")
    if world_landmarks is not None:
        world_landmarks = np.asarray(world_landmarks, dtype=np.float64)
        if world_landmarks.shape != (21, 3):
            raise ValueError("world_landmarks must have shape (21, 3)")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    result = np.full((21, 3), np.nan, dtype=np.float64)
    depth_valid = np.zeros(21, dtype=bool)
    for index, (u, v) in enumerate(points_2d):
        z = patch_depth(
            depth_m, u, v, radius=patch_radius, min_valid_pixels=min_valid_pixels
        )
        if z is not None:
            result[index] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
            depth_valid[index] = True

    depth_hits = int(np.sum(depth_valid))
    palm_hits = int(np.sum(depth_valid[list(REQUIRED_PALM_MP_INDICES)]))
    if depth_hits < min_depth_joints or palm_hits < min_palm_depth_joints:
        return (None, None) if return_quality else None

    aligned_fallback = None
    if world_landmarks is not None:
        aligned_fallback = _similarity_align(
            world_landmarks, world_landmarks[depth_valid], result[depth_valid]
        )

    previous_current_cam = None
    if previous_points_world is not None and current_color_to_world is not None:
        previous_points_world = np.asarray(previous_points_world, dtype=np.float64)
        current_color_to_world = np.asarray(current_color_to_world, dtype=np.float64)
        if previous_points_world.shape == (21, 3) and current_color_to_world.shape == (4, 4):
            world_to_current = np.linalg.inv(current_color_to_world)
            previous_current_cam = (
                world_to_current[:3, :3] @ previous_points_world.T
                + world_to_current[:3, 3:4]
            ).T

    visual_fallback = visual_depth_fallback(points_2d, world_landmarks, intrinsics)
    for index, (u, v) in enumerate(points_2d):
        if np.all(np.isfinite(result[index])):
            continue
        if previous_current_cam is not None and np.all(np.isfinite(previous_current_cam[index])):
            old = previous_current_cam[index]
            if old[2] > 0:
                old_uv = np.array([old[0] / old[2] * fx + cx, old[1] / old[2] * fy + cy])
                if np.linalg.norm(old_uv - points_2d[index]) <= 20.0:
                    z = float(old[2])
                    result[index] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
        if not np.all(np.isfinite(result[index])) and aligned_fallback is not None:
            result[index] = aligned_fallback[index]
        if not np.all(np.isfinite(result[index])) and visual_fallback is not None:
            result[index] = visual_fallback[index]

    if not np.all(np.isfinite(result)) or np.any(result[:, 2] <= 0):
        return (None, None) if return_quality else None
    quality = {
        "depth_valid": depth_valid,
        "depth_hits": depth_hits,
        "palm_depth_hits": palm_hits,
        "depth_ratio": depth_hits / 21.0,
    }
    return (result, quality) if return_quality else result


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
        min_depth_joints: int = 8,
        min_palm_depth_joints: int = 3,
    ):
        if close_ratio >= open_ratio:
            raise ValueError("close_ratio must be smaller than open_ratio")
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.smooth_window = max(1, smooth_window)
        self.patch_radius = patch_radius
        self.min_valid_pixels = min_valid_pixels
        self.min_depth_joints = min_depth_joints
        self.min_palm_depth_joints = min_palm_depth_joints
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
                "points_world": None, "rotation": None, "timestamp_s": None,
                "midpoint": None, "wrist": None, "grasp": False,
                "votes": deque(maxlen=self.smooth_window),
            })
            points_mp_cam, quality = recover_keypoints_rgbd(
                detection.landmarks_2d, detection.world_landmarks, depth_m, intrinsics,
                previous_points_world=history["points_world"],
                current_color_to_world=color_to_world,
                patch_radius=self.patch_radius,
                min_valid_pixels=self.min_valid_pixels,
                min_depth_joints=self.min_depth_joints,
                min_palm_depth_joints=self.min_palm_depth_joints,
                return_quality=True,
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

            midpoint_linear_velocity = np.zeros(3, dtype=np.float64)
            wrist_linear_velocity = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)
            if history["timestamp_s"] is not None and timestamp_s > history["timestamp_s"]:
                dt = timestamp_s - history["timestamp_s"]
                midpoint_linear_velocity = (midpoint - history["midpoint"]) / dt
                wrist_linear_velocity = (points_world[5] - history["wrist"]) / dt
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

            temporal_score = 1.0
            if history["points_world"] is not None:
                displacement = float(np.median(np.linalg.norm(
                    points_world - history["points_world"], axis=1
                )))
                temporal_score = float(np.exp(-max(0.0, displacement - 0.03) / 0.12))
            bone_score = 1.0 if 0.03 <= palm_size <= 0.16 else 0.25
            depth_score = min(1.0, quality["depth_hits"] / 12.0)
            pose_confidence = (
                float(detection.confidence) * depth_score * temporal_score * bone_score
            )

            hand = {
                "d2c": np.eye(4).tolist(),
                "c2w": color_to_world.tolist(),
                "confidence": pose_confidence,
                "detection_confidence": float(detection.confidence),
                "grasp_state": int(grasp),
                "grasp_ratio": None if not np.isfinite(ratio) else float(ratio),
                "depth_keypoints_valid": quality["depth_hits"],
                "depth_palm_keypoints_valid": quality["palm_depth_hits"],
                "wrist_pose": list_of(wrist_pose_cam),
                "palm_pose": list_of(wrist_pose_cam),
                "kpts_3d": list_of(points_cam),
                "kpts_2d": list_of(points_2d),
                "joint_angles": {},
                "wrist_pose_raw_world": list_of(wrist_pose_world),
                "wrist_pose_opt_world": None,
                "wrist_lin_vel_raw_world": list_of(wrist_linear_velocity),
                "wrist_ang_vel_raw_world": list_of(angular_velocity),
                "wrist_lin_vel_opt_world": None,
                "wrist_ang_vel_opt_world": None,
                "index_translation_raw_world": list_of(points_world[1]),
                "index_translation_opt_world": None,
                "thumb_translation_raw_world": list_of(points_world[0]),
                "thumb_translation_opt_world": None,
                "thumb_base_raw_world": list_of(points_world[6]),
                "thumb_base_opt_world": None,
                "index_base_raw_world": list_of(points_world[8]),
                "index_base_opt_world": None,
                "midpoint_pose_raw_world": list_of(midpoint_pose),
                "midpoint_pose_opt_world": None,
                "midpoint_translation_raw_world": list_of(midpoint),
                "midpoint_orientation_raw_world": list_of(rotation.reshape(-1)),
                "midpoint_translation_opt_world": None,
                "midpoint_orientation_opt_world": None,
                "midpoint_lin_vel_raw_world": list_of(midpoint_linear_velocity),
                "midpoint_ang_vel_raw_world": list_of(angular_velocity),
                "midpoint_lin_vel_opt_world": None,
                "midpoint_ang_vel_opt_world": None,
                "distance_midpoint2wrist_raw_world": float(np.linalg.norm(midpoint - points_world[5])),
                "distance_midpoint2wrist_opt_world": None,
            }
            output["hand_r" if side == "right" else "hand_l"] = hand
            history.update({
                "points_world": points_world, "rotation": rotation,
                "timestamp_s": timestamp_s, "midpoint": midpoint,
                "wrist": points_world[5], "grasp": bool(grasp),
            })
        return output


def _continuous_segments(present: np.ndarray) -> List[tuple[int, int]]:
    segments: List[tuple[int, int]] = []
    start = None
    for index in range(len(present) + 1):
        active = index < len(present) and bool(present[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            segments.append((start, index))
            start = None
    return segments


def _interpolate_rotation(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    blended = (1.0 - alpha) * left + alpha * right
    u, _, vt = np.linalg.svd(blended)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _interpolate_pose(left, right, alpha: float):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _interpolate_rotation(left[:3, :3], right[:3, :3], alpha)
    pose[:3, 3] = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return pose.tolist()


def _smooth_positions(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    for index in range(len(values)):
        local = values[max(0, index - 2):min(len(values), index + 3)]
        median = np.median(local, axis=0)
        if np.linalg.norm(values[index] - median) > 0.15:
            values[index] = median
    size = min(int(window) | 1, len(values) if len(values) % 2 else len(values) - 1)
    if size < 5:
        return values
    degree = min(polyorder, size - 2)
    radius = size // 2
    smoothed = np.empty_like(values)
    for index in range(len(values)):
        start = min(max(0, index - radius), len(values) - size)
        end = start + size
        offsets = np.arange(start, end, dtype=np.float64) - index
        for axis in range(values.shape[1]):
            coefficients = np.polyfit(offsets, values[start:end, axis], degree)
            smoothed[index, axis] = np.polyval(coefficients, 0.0)
    return smoothed


def _ema_rotation(raw: np.ndarray, previous: Optional[np.ndarray], alpha: float) -> np.ndarray:
    x_axis, y_axis = raw[:, 0].copy(), raw[:, 1].copy()
    if previous is not None:
        if np.dot(x_axis, previous[:, 0]) < 0:
            x_axis *= -1
        if np.dot(y_axis, previous[:, 1]) < 0:
            y_axis *= -1
        x_axis = (1.0 - alpha) * previous[:, 0] + alpha * x_axis
        y_axis = (1.0 - alpha) * previous[:, 1] + alpha * y_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _angular_velocity(previous: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
    relative = previous.T @ current
    angle = rotation_angle(relative)
    if angle < 1e-9 or dt <= 0:
        return np.zeros(3, dtype=np.float64)
    skew = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])
    if abs(np.sin(angle)) < 1e-7:
        return np.zeros(3, dtype=np.float64)
    return skew / (2.0 * np.sin(angle)) * angle / dt


def optimize_hand_sequence(
    documents: List[Dict],
    timestamps_s: np.ndarray,
    confidence_threshold: float = 0.3,
    min_segment_frames: int = 6,
    fill_max_gap: int = 10,
    smooth_window: int = 21,
    smooth_polyorder: int = 2,
    ema_alpha: float = 0.15,
) -> Dict[str, int]:
    """Clean and optimize a complete sequence of hand JSON documents in-place."""
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if len(documents) != len(timestamps_s):
        raise ValueError("hand documents and timestamps must have the same length")
    stats = {"filtered": 0, "interpolated": 0, "optimized": 0}

    vector_fields = (
        "index_translation_raw_world", "thumb_translation_raw_world",
        "thumb_base_raw_world", "index_base_raw_world",
        "midpoint_translation_raw_world", "kpts_3d", "kpts_2d",
    )
    pose_fields = ("wrist_pose_raw_world", "midpoint_pose_raw_world", "c2w")

    for side in ("hand_r", "hand_l"):
        for document in documents:
            hand = document.get(side)
            if hand is not None and float(hand.get("confidence", 0.0)) < confidence_threshold:
                document[side] = None
                stats["filtered"] += 1

        observed = [index for index, document in enumerate(documents) if document.get(side)]
        for left_index, right_index in zip(observed[:-1], observed[1:]):
            gap = right_index - left_index - 1
            if gap <= 0 or gap > fill_max_gap:
                continue
            left, right = documents[left_index][side], documents[right_index][side]
            for offset in range(1, gap + 1):
                alpha = offset / (gap + 1.0)
                hand = copy.deepcopy(left)
                hand["confidence"] = min(left["confidence"], right["confidence"]) * 0.8
                hand["depth_keypoints_valid"] = 0
                hand["depth_palm_keypoints_valid"] = 0
                hand["interpolated"] = True
                for field in vector_fields:
                    if left.get(field) is not None and right.get(field) is not None:
                        lv, rv = np.asarray(left[field]), np.asarray(right[field])
                        hand[field] = ((1.0 - alpha) * lv + alpha * rv).tolist()
                for field in pose_fields:
                    if left.get(field) is not None and right.get(field) is not None:
                        hand[field] = _interpolate_pose(left[field], right[field], alpha)
                raw_mid_pose = np.asarray(hand["midpoint_pose_raw_world"])
                hand["midpoint_orientation_raw_world"] = raw_mid_pose[:3, :3].reshape(-1).tolist()
                for key in list(hand):
                    if "_opt_world" in key:
                        hand[key] = None
                documents[left_index + offset][side] = hand
                stats["interpolated"] += 1

        present = np.array([document.get(side) is not None for document in documents])
        for start, end in _continuous_segments(present):
            if end - start < min_segment_frames:
                for index in range(start, end):
                    documents[index][side] = None
                    stats["filtered"] += 1

        present = np.array([document.get(side) is not None for document in documents])
        for start, end in _continuous_segments(present):
            hands = [documents[index][side] for index in range(start, end)]

            def smooth(field: str) -> np.ndarray:
                return _smooth_positions(
                    np.asarray([hand[field] for hand in hands], dtype=np.float64),
                    smooth_window, smooth_polyorder,
                )

            wrist = _smooth_positions(
                np.asarray([
                    np.asarray(hand["wrist_pose_raw_world"], dtype=np.float64)[:3, 3]
                    for hand in hands
                ]),
                smooth_window,
                smooth_polyorder,
            )
            thumb = smooth("thumb_translation_raw_world")
            index_tip = smooth("index_translation_raw_world")
            thumb_base = smooth("thumb_base_raw_world")
            index_base = smooth("index_base_raw_world")
            midpoint = 0.5 * (thumb + index_tip)
            previous_wrist_rotation = None
            previous_mid_rotation = None

            for local_index, hand in enumerate(hands):
                raw_wrist_rotation = np.asarray(hand["wrist_pose_raw_world"])[:3, :3]
                wrist_rotation = _ema_rotation(
                    raw_wrist_rotation, previous_wrist_rotation, ema_alpha
                )
                frame_points = np.zeros((21, 3), dtype=np.float64)
                frame_points[5] = wrist[local_index]
                frame_points[6] = thumb_base[local_index]
                frame_points[8] = index_base[local_index]
                mid_rotation = hand_frame(frame_points, previous_mid_rotation)
                if mid_rotation is None:
                    mid_rotation = wrist_rotation
                mid_rotation = _ema_rotation(mid_rotation, previous_mid_rotation, ema_alpha)

                wrist_pose = np.eye(4)
                wrist_pose[:3, :3] = wrist_rotation
                wrist_pose[:3, 3] = wrist[local_index]
                midpoint_pose = np.eye(4)
                midpoint_pose[:3, :3] = mid_rotation
                midpoint_pose[:3, 3] = midpoint[local_index]
                hand.update({
                    "wrist_pose_opt_world": wrist_pose.tolist(),
                    "index_translation_opt_world": index_tip[local_index].tolist(),
                    "thumb_translation_opt_world": thumb[local_index].tolist(),
                    "thumb_base_opt_world": thumb_base[local_index].tolist(),
                    "index_base_opt_world": index_base[local_index].tolist(),
                    "midpoint_pose_opt_world": midpoint_pose.tolist(),
                    "midpoint_translation_opt_world": midpoint[local_index].tolist(),
                    "midpoint_orientation_opt_world": mid_rotation.reshape(-1).tolist(),
                    "distance_midpoint2wrist_opt_world": float(np.linalg.norm(
                        midpoint[local_index] - wrist[local_index]
                    )),
                })
                previous_wrist_rotation = wrist_rotation
                previous_mid_rotation = mid_rotation
                stats["optimized"] += 1

            for local_index, hand in enumerate(hands):
                if local_index == 0:
                    wrist_velocity = midpoint_velocity = np.zeros(3)
                    wrist_angular = midpoint_angular = np.zeros(3)
                else:
                    dt = timestamps_s[start + local_index] - timestamps_s[start + local_index - 1]
                    previous = hands[local_index - 1]
                    wrist_pose = np.asarray(hand["wrist_pose_opt_world"])
                    previous_wrist_pose = np.asarray(previous["wrist_pose_opt_world"])
                    midpoint_pose = np.asarray(hand["midpoint_pose_opt_world"])
                    previous_midpoint_pose = np.asarray(previous["midpoint_pose_opt_world"])
                    if dt <= 0:
                        wrist_velocity = midpoint_velocity = np.zeros(3)
                        wrist_angular = midpoint_angular = np.zeros(3)
                    else:
                        wrist_velocity = (
                            wrist_pose[:3, 3] - previous_wrist_pose[:3, 3]
                        ) / dt
                        midpoint_velocity = (
                            midpoint_pose[:3, 3] - previous_midpoint_pose[:3, 3]
                        ) / dt
                        wrist_angular = _angular_velocity(
                            previous_wrist_pose[:3, :3], wrist_pose[:3, :3], dt
                        )
                        midpoint_angular = _angular_velocity(
                            previous_midpoint_pose[:3, :3], midpoint_pose[:3, :3], dt
                        )
                hand["wrist_lin_vel_opt_world"] = wrist_velocity.tolist()
                hand["wrist_ang_vel_opt_world"] = wrist_angular.tolist()
                hand["midpoint_lin_vel_opt_world"] = midpoint_velocity.tolist()
                hand["midpoint_ang_vel_opt_world"] = midpoint_angular.tolist()

            grasp_values = np.asarray([int(hand.get("grasp_state", 0)) for hand in hands])
            radius = 2
            for local_index, hand in enumerate(hands):
                window = grasp_values[
                    max(0, local_index - radius):min(len(hands), local_index + radius + 1)
                ]
                hand["grasp_state"] = int(np.mean(window) >= 0.5)

    return stats
