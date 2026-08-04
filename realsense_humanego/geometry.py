"""Small, dependency-free SE(3) helpers used by the converter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class TimedPose:
    timestamp_s: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = quaternion_to_matrix(self.quaternion_xyzw)
        result[:3, 3] = self.translation
        return result


def normalize_quaternion(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(quaternion_xyzw), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError("quaternion is zero or non-finite")
    return q / norm


def quaternion_to_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an xyzw quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"expected rotation shape (3, 3), got {m.shape}")
    # Project small numeric drift back onto SO(3).
    u, _, vt = np.linalg.svd(m)
    m = u @ vt
    if np.linalg.det(m) < 0:
        u[:, -1] *= -1
        m = u @ vt

    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                          (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                          (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                          (m[1, 0] - m[0, 1]) / s])
    return normalize_quaternion(q)


def slerp(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, fraction: float) -> np.ndarray:
    q0 = normalize_quaternion(q0_xyzw)
    q1 = normalize_quaternion(q1_xyzw)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(q0 + fraction * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return normalize_quaternion(
        np.sin((1.0 - fraction) * theta) / sin_theta * q0
        + np.sin(fraction * theta) / sin_theta * q1
    )


def load_tum_trajectory(path: Path) -> List[TimedPose]:
    """Load `timestamp tx ty tz qx qy qz qw`, accepting spaces or commas."""
    poses: List[TimedPose] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.split("#", 1)[0].strip().replace(",", " ")
            if not line:
                continue
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 fields, got {len(fields)}")
            values = np.asarray([float(value) for value in fields], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                continue
            poses.append(TimedPose(values[0], values[1:4], normalize_quaternion(values[4:8])))
    poses.sort(key=lambda pose: pose.timestamp_s)
    # A restarted sensor can emit a duplicate timestamp. The last result is the most recent.
    deduplicated = {pose.timestamp_s: pose for pose in poses}
    return [deduplicated[key] for key in sorted(deduplicated)]


def interpolate_pose(
    poses: List[TimedPose], timestamp_s: float, max_gap_s: float
) -> Optional[np.ndarray]:
    """Interpolate without extrapolating or bridging a tracking-loss sized gap."""
    if not poses:
        return None
    times = np.fromiter((pose.timestamp_s for pose in poses), dtype=np.float64)
    right = int(np.searchsorted(times, timestamp_s, side="left"))
    if right < len(poses) and abs(poses[right].timestamp_s - timestamp_s) < 1e-9:
        return poses[right].matrix
    if right == 0 or right == len(poses):
        return None
    before, after = poses[right - 1], poses[right]
    gap = after.timestamp_s - before.timestamp_s
    if gap <= 0 or gap > max_gap_s:
        return None
    alpha = (timestamp_s - before.timestamp_s) / gap
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (1.0 - alpha) * before.translation + alpha * after.translation
    pose[:3, :3] = quaternion_to_matrix(
        slerp(before.quaternion_xyzw, after.quaternion_xyzw, alpha)
    )
    return pose


def rotation_angle(rotation: np.ndarray) -> float:
    """Return the SO(3) angle in radians."""
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rotation_to_rpy_zyx(rotation: np.ndarray) -> np.ndarray:
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)
