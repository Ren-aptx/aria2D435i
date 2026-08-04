import numpy as np

from realsense_humanego.hands import (
    patch_depth,
    recover_keypoints_rgbd,
    remap_mediapipe_to_aria,
)


def test_patch_depth_uses_neighborhood_median_and_range_filter():
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[7:14, 7:14] = 0.6
    depth[8, 8] = 1.5
    depth[9, 9] = 0.0
    assert np.isclose(patch_depth(depth, 10, 10), 0.6)
    assert patch_depth(depth, 1, 1) is None


def test_rgbd_recovery_backprojects_metric_depth():
    points = np.tile(np.array([[10.0, 10.0]]), (21, 1))
    points[:, 0] += np.arange(21) * 0.1
    points[9] = [15.0, 10.0]
    depth = np.full((30, 30), 0.8, dtype=np.float32)
    intrinsics = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    recovered = recover_keypoints_rgbd(points, None, depth, intrinsics)
    assert recovered is not None
    assert np.allclose(recovered[:, 2], 0.8)
    assert np.allclose(recovered[0], [0.0, 0.0, 0.8])


def test_aria_remap_builds_palm_center():
    points = np.arange(63, dtype=np.float64).reshape(21, 3)
    aria = remap_mediapipe_to_aria(points)
    assert np.array_equal(aria[0], points[4])
    assert np.array_equal(aria[5], points[0])
    assert np.allclose(aria[20], (points[0] + points[5] + points[9]) / 3.0)
