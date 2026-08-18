import csv
import json
from pathlib import Path

import cv2
import numpy as np

from realsense_humanego.exporter import (
    ExportConfig,
    ExportFrame,
    SourceFrame,
    _phase_modes,
    export_session,
)


def _write_fixture(session: Path):
    raw = session / "raw"
    for name in ("rgb", "aligned_depth"):
        (raw / name).mkdir(parents=True, exist_ok=True)
    calibration = {
        "depth_scale_m": 0.001,
        "color": {
            "width": 32, "height": 24, "fx": 30.0, "fy": 30.0,
            "cx": 16.0, "cy": 12.0, "coeffs": [0, 0, 0, 0, 0],
        },
        "T_color_to_left_ir": np.eye(4).tolist(),
    }
    (raw / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    rows = []
    for index, timestamp_ns in enumerate((1_000_000_000, 1_033_000_000, 1_066_000_000)):
        name = f"{index:06d}.png"
        cv2.imwrite(str(raw / "rgb" / name), np.full((24, 32, 3), index, np.uint8))
        cv2.imwrite(str(raw / "aligned_depth" / name), np.full((24, 32), 700, np.uint16))
        rows.append({
            "idx": index, "rgb_timestamp_ns": timestamp_ns,
            "aligned_depth_timestamp_ns": timestamp_ns,
            "rgb_file": f"rgb/{name}", "aligned_depth_file": f"aligned_depth/{name}",
        })
    with (raw / "timestamps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # The final source frame is deliberately outside the trajectory and must be dropped.
    (raw / "trajectory_rgb.txt").write_text(
        "0.99 0 0 0 0 0 0 1\n"
        "1.00 0 0 0 0 0 0 1\n"
        "1.033 0.1 0 0 0 0 0 1\n"
        "1.05 0.2 0 0 0 0 0 1\n",
        encoding="utf-8",
    )


def test_export_session_writes_humanego_contract_and_drops_invalid_pose(tmp_path):
    session = tmp_path / "session"
    _write_fixture(session)
    manifest = export_session(ExportConfig(
        session=session, fixed_camera=False, hand_mode="none"
    ))
    assert manifest["source_frames"] == 3
    assert manifest["exported_frames"] == 2
    assert manifest["dropped_frames"] == [
        {"source_idx": 2, "reason": "invalid_or_missing_pose"}
    ]
    frame = session / "preprocess" / "all_data" / "00001"
    for filename in (
        "rgb.png", "depth.png", "aria_cam_rgb.json", "aria_hands.json",
        "aria_slam.json", "aria_phases.json",
    ):
        assert (frame / filename).is_file()
    camera = json.loads((frame / "aria_cam_rgb.json").read_text(encoding="utf-8"))
    assert camera["source"] == "realsense_d435i"
    assert np.isclose(camera["c2w"][0][3], 0.1)
    hands = json.loads((frame / "aria_hands.json").read_text(encoding="utf-8"))
    assert hands["hand_l"] is None and hands["hand_r"] is None


def test_phase_modes_use_optimized_hand_motion_for_transition():
    source = SourceFrame(0, 0, 0, Path("rgb.png"), Path("depth.png"))
    frames = [
        ExportFrame(index, source, np.eye(4), np.zeros(3), np.zeros(3))
        for index in range(9)
    ]
    documents = []
    for index in range(9):
        speed = 0.3 if index < 2 else 0.0
        documents.append({
            "hand_r": {"midpoint_lin_vel_opt_world": [speed, 0.0, 0.0]},
            "hand_l": None,
        })
    config = ExportConfig(
        session=Path("."), fixed_camera=False, finish_frames=0, hand_stable_frames=2,
        hand_transition_frames=0,
    )
    assert _phase_modes(frames, config, documents) == [3, 3, 0, 0, 0, 0, 0, 0, 0]


def test_fixed_camera_export_needs_no_trajectory_and_writes_static_slam(tmp_path):
    session = tmp_path / "session"
    _write_fixture(session)
    (session / "raw" / "trajectory_rgb.txt").unlink()

    manifest = export_session(ExportConfig(
        session=session,
        fixed_camera=True,
        hand_mode="none",
        manip_start=1,
        manip_end=1,
        finished_start=2,
    ))

    assert manifest["camera_mode"] == "fixed"
    assert manifest["trajectory"] is None
    assert manifest["exported_frames"] == 3
    assert manifest["grasp_detection"]["ratio"] == "2d_thumb_index_over_wrist_middle_mcp"
    assert manifest["closed_grasp_observations"] == {"right": 0, "left": 0}
    for index, expected_mode in enumerate((3, 0, 4)):
        frame = session / "preprocess" / "all_data" / f"{index:05d}"
        camera = json.loads((frame / "aria_cam_rgb.json").read_text())
        slam = json.loads((frame / "aria_slam.json").read_text())
        phase = json.loads((frame / "aria_phases.json").read_text())
        assert np.allclose(camera["c2w"], np.eye(4))
        assert slam["t_world"] == [0.0, 0.0, 0.0]
        assert slam["delta_t_world"] == [0.0, 0.0, 0.0]
        assert slam["linear_speed_mps"] == 0.0
        assert slam["angular_speed_rps"] == 0.0
        assert phase["mode"] == expected_mode


def test_fixed_camera_hand_phase_never_emits_navigation_modes():
    source = SourceFrame(0, 0, 0, Path("rgb.png"), Path("depth.png"))
    frames = [
        ExportFrame(index, source, np.eye(4), np.zeros(3), np.zeros(3))
        for index in range(8)
    ]
    speeds = (None, 0.3, 0.01, 0.02, 0.01, 0.4, None, None)
    documents = [{
        "hand_r": None if speed is None else {
            "midpoint_lin_vel_opt_world": [speed, 0.0, 0.0]
        },
        "hand_l": None,
    } for speed in speeds]
    config = ExportConfig(
        session=Path("."), fixed_camera=True, finish_frames=1,
        hand_stable_frames=3,
    )

    modes = _phase_modes(frames, config, documents)

    assert modes == [3, 3, 0, 0, 0, 3, 3, 4]
    assert not ({1, 2} & set(modes))
