import numpy as np

from realsense_humanego.hands import (
    _normalized_pinch_ratio_2d,
    DetectedHand,
    HandProcessor,
    optimize_hand_sequence,
    patch_depth,
    recover_keypoints_rgbd,
    remap_mediapipe_to_aria,
)


def test_2d_pinch_ratio_separates_open_and_closed_hand_apertures():
    points = np.zeros((21, 2), dtype=np.float64)
    points[0] = [10.0, 10.0]   # wrist
    points[9] = [10.0, 30.0]   # middle MCP: palm scale = 20 px
    points[4], points[8] = [3.0, 30.0], [17.0, 30.0]
    assert np.isclose(_normalized_pinch_ratio_2d(points), 0.7)

    points[4], points[8] = [-2.0, 30.0], [22.0, 30.0]
    assert np.isclose(_normalized_pinch_ratio_2d(points), 1.2)


def test_patch_depth_uses_neighborhood_median_and_range_filter():
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[7:14, 7:14] = 0.6
    depth[8, 8] = 1.5
    depth[9, 9] = 0.0
    assert np.isclose(patch_depth(depth, 10, 10), 0.6)
    assert patch_depth(depth, 1, 1) is None


def test_patch_depth_prefers_nearest_stable_surface_without_center_depth():
    depth = np.full((9, 9), 1.2, dtype=np.float32)
    depth[2:4, 2:5] = 0.55
    depth[4, 4] = 0.0
    assert np.isclose(patch_depth(depth, 4, 4, radius=4), 0.55)


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


def test_rgbd_recovery_rejects_too_few_measured_joints():
    points = np.column_stack([np.arange(21) + 2.0, np.full(21, 5.0)])
    depth = np.zeros((12, 30), dtype=np.float32)
    for u, v in points[:7].astype(int):
        depth[v, u] = 0.8
    intrinsics = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]])
    assert recover_keypoints_rgbd(
        points, None, depth, intrinsics, patch_radius=0, min_valid_pixels=1
    ) is None


def test_temporal_fallback_transforms_world_point_into_current_camera():
    points = np.column_stack([np.arange(21) + 30.0, np.full(21, 5.0)])
    points[20] = [10.0, 10.0]
    depth = np.zeros((20, 100), dtype=np.float32)
    for u, v in points[:20].astype(int):
        depth[v, u] = 0.8
    intrinsics = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    previous_world = np.zeros((21, 3), dtype=np.float64)
    # Keep the observed joints consistent with the current RGB-D points so
    # this fixture exercises temporal hole filling rather than the sequence
    # jump-rejection guard.
    previous_world[:20, 0] = (points[:20, 0] - 10.0) * 0.8 / 100.0 + 0.5
    previous_world[:20, 1] = (points[:20, 1] - 10.0) * 0.8 / 100.0
    previous_world[:20, 2] = 0.8
    previous_world[20, 2] = 1.0
    previous_world[20] = [0.5, 0.0, 1.0]
    current_c2w = np.eye(4)
    current_c2w[0, 3] = 0.5
    recovered = recover_keypoints_rgbd(
        points, None, depth, intrinsics,
        previous_points_world=previous_world,
        current_color_to_world=current_c2w,
        patch_radius=0,
        min_valid_pixels=1,
    )
    assert recovered is not None
    assert np.allclose(recovered[20], [0.0, 0.0, 1.0])


def test_hand_processor_leaves_opt_fields_for_sequence_pass():
    points = np.tile(np.array([[30.0, 30.0]]), (21, 1))
    points[0] = [30.0, 40.0]
    points[2] = [24.0, 35.0]
    points[4] = [20.0, 25.0]
    points[5] = [36.0, 34.0]
    points[8] = [40.0, 25.0]
    points[9] = [30.0, 28.0]
    points[13] = [26.0, 30.0]
    points[17] = [22.0, 32.0]
    detection = DetectedHand("right", 0.9, points)
    intrinsics = np.array([[100.0, 0.0, 30.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]])
    hands = HandProcessor().process(
        [detection], np.full((60, 60), 0.8, np.float32),
        intrinsics, np.eye(4), 1.0,
    )
    hand = hands["hand_r"]
    assert hand is not None
    assert hand["wrist_pose_raw_world"] is not None
    assert hand["wrist_pose_opt_world"] is None
    assert hand["midpoint_pose_opt_world"] is None


def _hand_document(x: float):
    rotation = np.eye(3)
    wrist = np.array([x, 0.0, 0.8])
    thumb = wrist + np.array([-0.03, 0.08, 0.0])
    index = wrist + np.array([0.03, 0.08, 0.0])
    thumb_base = wrist + np.array([-0.025, 0.04, 0.0])
    index_base = wrist + np.array([0.025, 0.04, 0.0])
    midpoint = 0.5 * (thumb + index)
    wrist_pose = np.eye(4)
    wrist_pose[:3, :3] = rotation
    wrist_pose[:3, 3] = wrist
    midpoint_pose = np.eye(4)
    midpoint_pose[:3, :3] = rotation
    midpoint_pose[:3, 3] = midpoint
    hand = {
        "confidence": 0.9,
        "grasp_state": 0,
        "depth_keypoints_valid": 21,
        "wrist_pose_raw_world": wrist_pose.tolist(),
        "wrist_pose_opt_world": None,
        "index_translation_raw_world": index.tolist(),
        "index_translation_opt_world": None,
        "thumb_translation_raw_world": thumb.tolist(),
        "thumb_translation_opt_world": None,
        "thumb_base_raw_world": thumb_base.tolist(),
        "thumb_base_opt_world": None,
        "index_base_raw_world": index_base.tolist(),
        "index_base_opt_world": None,
        "midpoint_pose_raw_world": midpoint_pose.tolist(),
        "midpoint_pose_opt_world": None,
        "midpoint_translation_raw_world": midpoint.tolist(),
        "midpoint_translation_opt_world": None,
        "midpoint_orientation_raw_world": rotation.reshape(-1).tolist(),
        "midpoint_orientation_opt_world": None,
        "wrist_lin_vel_opt_world": None,
        "wrist_ang_vel_opt_world": None,
        "midpoint_lin_vel_opt_world": None,
        "midpoint_ang_vel_opt_world": None,
        "distance_midpoint2wrist_opt_world": None,
    }
    return {"hand_r": hand, "hand_l": None}


def test_sequence_optimizer_populates_real_opt_fields():
    raw_x = [0.00, 0.01, 0.02, 0.03, 0.18, 0.05, 0.06, 0.07, 0.08]
    documents = [_hand_document(value) for value in raw_x]
    optimize_hand_sequence(
        documents, np.arange(len(documents)) / 30.0,
        min_segment_frames=6, smooth_window=7,
    )
    optimized = documents[4]["hand_r"]
    assert optimized["midpoint_pose_opt_world"] is not None
    assert optimized["midpoint_lin_vel_opt_world"] is not None
    assert not np.isclose(optimized["midpoint_translation_opt_world"][0], raw_x[4])


def test_aria_remap_builds_palm_center():
    points = np.arange(63, dtype=np.float64).reshape(21, 3)
    aria = remap_mediapipe_to_aria(points)
    assert np.array_equal(aria[0], points[4])
    assert np.array_equal(aria[5], points[0])
    assert np.allclose(aria[20], (points[0] + points[5] + points[9]) / 3.0)
