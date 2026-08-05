import json

import cv2
import numpy as np

from realsense_humanego.object_pose import (
    generate_rgbd_object_poses,
    masked_depth_points,
)


def test_masked_depth_points_deprojects_mask():
    depth = np.full((20, 30), 1000, np.uint16)
    mask = np.zeros((20, 30), np.uint8)
    mask[5:15, 10:20] = 255
    intrinsics = np.array([[100.0, 0, 15.0], [0, 100.0, 10.0], [0, 0, 1]])
    points = masked_depth_points(depth, mask, intrinsics, 0.001)
    assert len(points) > 20
    assert np.allclose(points[:, 2], 1.0)


def test_rgbd_object_pose_writes_humanego_schema(tmp_path):
    session = tmp_path / "session"
    raw = session / "raw"
    raw.mkdir(parents=True)
    (raw / "calibration.json").write_text(json.dumps({"depth_scale_m": 0.001}))
    frame_dir = session / "preprocess" / "all_data" / "00000"
    frame_dir.mkdir(parents=True)
    rgb = frame_dir / "rgb.png"
    cv2.imwrite(str(rgb), np.zeros((40, 50, 3), np.uint8))
    cv2.imwrite(str(frame_dir / "depth.png"), np.full((40, 50), 800, np.uint16))
    mask = np.zeros((40, 50), np.uint8)
    mask[10:30, 15:35] = 255
    cv2.imwrite(str(frame_dir / "mask_obj1.png"), mask)
    camera = {
        "idx": 0,
        "k": [[80.0, 0, 25.0], [0, 80.0, 20.0], [0, 0, 1]],
        "c2w": np.eye(4).tolist(),
    }
    (frame_dir / "aria_cam_rgb.json").write_text(json.dumps(camera))
    document = generate_rgbd_object_poses(session, [rgb], min_points=20)
    object_data = document["objects"]["obj1"]
    assert np.asarray(object_data["object_to_cam0_matrix"]).shape == (4, 4)
    assert len(object_data["points_3d_cam0"]) >= 20
    assert (session / "preprocess" / "camtriangulator_results.json").is_file()


def test_rgbd_pose_delegates_axes_to_task_estimator(tmp_path):
    session = tmp_path / "session"
    raw = session / "raw"
    raw.mkdir(parents=True)
    (raw / "calibration.json").write_text(json.dumps({"depth_scale_m": 0.001}))
    frame_dir = session / "preprocess" / "all_data" / "00000"
    frame_dir.mkdir(parents=True)
    rgb = frame_dir / "rgb.png"
    cv2.imwrite(str(rgb), np.zeros((40, 50, 3), np.uint8))
    cv2.imwrite(str(frame_dir / "depth.png"), np.full((40, 50), 800, np.uint16))
    for key, bounds in {"obj1": (5, 20), "obj2": (25, 40)}.items():
        mask = np.zeros((40, 50), np.uint8)
        mask[10:30, bounds[0]:bounds[1]] = 255
        cv2.imwrite(str(frame_dir / f"mask_{key}.png"), mask)
    camera = {
        "idx": 0,
        "k": [[80.0, 0, 25.0], [0, 80.0, 20.0], [0, 0, 1]],
        "c2w": np.eye(4).tolist(),
    }
    (frame_dir / "aria_cam_rgb.json").write_text(json.dumps(camera))
    calls = []

    def estimator(method, points, is_anchor, anchor_center, image_path, mask_path):
        calls.append((method, is_anchor, None if anchor_center is None else anchor_center.copy()))
        pose = np.eye(4)
        pose[:3, 3] = np.mean(points, axis=0)
        return pose, {"method": f"fake_{method}"}

    document = generate_rgbd_object_poses(
        session, [rgb], min_points=20,
        pose_methods={"obj1": "pca1", "obj2": "pca2"},
        pose_estimator=estimator,
    )
    assert calls[0][0:2] == ("pca1", True)
    assert calls[0][2] is None
    assert calls[1][0:2] == ("pca2", False)
    assert calls[1][2] is not None
    assert document["objects"]["obj2"]["info"]["configured_pose_method"] == "pca2"
