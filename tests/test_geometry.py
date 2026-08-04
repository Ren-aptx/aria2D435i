from pathlib import Path

import numpy as np

from realsense_humanego.geometry import (
    TimedPose,
    interpolate_pose,
    matrix_to_quaternion,
    quaternion_to_matrix,
    slerp,
)


def test_quaternion_matrix_round_trip():
    quaternion = np.array([0.2, -0.1, 0.3, 0.9])
    matrix = quaternion_to_matrix(quaternion)
    recovered = matrix_to_quaternion(matrix)
    assert np.allclose(matrix, quaternion_to_matrix(recovered), atol=1e-8)
    assert np.isclose(np.linalg.det(matrix), 1.0)


def test_slerp_uses_short_arc():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(slerp(q, -q, 0.5), q)


def test_interpolation_rejects_extrapolation_and_large_gap():
    poses = [
        TimedPose(1.0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
        TimedPose(1.1, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
    ]
    assert interpolate_pose(poses, 0.9, 0.2) is None
    assert interpolate_pose(poses, 1.2, 0.2) is None
    assert interpolate_pose(poses, 1.05, 0.05) is None
    pose = interpolate_pose(poses, 1.05, 0.2)
    assert np.allclose(pose[:3, 3], [0.5, 0.0, 0.0])

