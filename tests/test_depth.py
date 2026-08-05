from pathlib import Path

import cv2
import numpy as np
import pytest


def test_recorded_object_depth_has_valid_pixels():
    """Optional diagnostic for the local sample session, not a collection side effect."""
    frame = Path(
        "/home/tenda/HumanEgodata/data/serve_bread/realsense/"
        "rs_serve_bread_000/preprocess/all_data/00150"
    )
    depth_path = frame / "depth.png"
    mask_path = frame / "mask_obj1.png"
    if not depth_path.is_file() or not mask_path.is_file():
        pytest.skip("local recorded-depth fixture is unavailable")

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert depth is not None and mask is not None
    object_depth = depth[mask > 0]
    assert object_depth.size > 0
    assert np.count_nonzero(object_depth) / object_depth.size > 0.5
