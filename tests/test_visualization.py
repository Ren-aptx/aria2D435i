import json

import cv2
import numpy as np

from realsense_humanego.visualization import export_official_videos


def test_export_official_videos_writes_two_streamed_mp4s(tmp_path):
    session = tmp_path / "session"
    preprocess = session / "preprocess"
    all_data = preprocess / "all_data"
    all_data.mkdir(parents=True)
    (preprocess / "aria_cam_rgb_config.json").write_text(
        json.dumps({"fps": 15}), encoding="utf-8"
    )
    points = np.column_stack([
        np.linspace(15, 45, 21), np.linspace(20, 40, 21)
    ]).tolist()
    for index in range(3):
        frame = all_data / f"{index:05d}"
        frame.mkdir()
        image = np.full((64, 96, 3), 40 + index * 20, dtype=np.uint8)
        cv2.imwrite(str(frame / "rgb.png"), image)
        cv2.imwrite(str(frame / "rgb_WoArm_WArmObjKpts.png"), image)
        (frame / "aria_hands.json").write_text(json.dumps({
            "hand_r": {
                "kpts_2d": points,
                "grasp_state": int(index == 1),
                "grasp_ratio_2d": 0.7 if index == 1 else 1.2,
            },
            "hand_l": None,
        }), encoding="utf-8")
        (frame / "aria_phases.json").write_text(
            json.dumps({"mode_str": "MANIPULATION"}), encoding="utf-8"
        )

    outputs = export_official_videos(session)

    assert set(outputs) == {"aria_vis", "visualkpts_vis"}
    for info in outputs.values():
        path = info["path"]
        assert info["frames"] == 3
        capture = cv2.VideoCapture(path)
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        capture.release()
