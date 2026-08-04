"""RGB-D object point-cloud fusion with HumanEgo triangulator-compatible output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def masked_depth_points(
    depth_raw: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale_m: float,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.0,
    erode_pixels: int = 1,
) -> np.ndarray:
    """Deproject a mask, suppressing boundary mixing and gross depth outliers."""
    if depth_raw.shape[:2] != mask.shape[:2]:
        raise ValueError("aligned depth and object mask have different dimensions")
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if erode_pixels > 0:
        size = 2 * erode_pixels + 1
        eroded = cv2.erode(binary, np.ones((size, size), np.uint8))
        # Tiny masks can disappear; the uneroded mask is safer than returning nothing.
        if np.count_nonzero(eroded) >= 20:
            binary = eroded
    v, u = np.nonzero(binary)
    z = depth_raw[v, u].astype(np.float64) * float(depth_scale_m)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    u, v, z = u[valid], v[valid], z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    tolerance = max(0.08, 5.0 * 1.4826 * mad)
    coherent = np.abs(z - median) <= tolerance
    u, v, z = u[coherent], v[coherent], z[coherent]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    return np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])


def robust_pca_pose(points_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate centroid/orientation with deterministic PCA signs."""
    points = np.asarray(points_cam, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("at least three 3D object points are required")
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    cutoff = float(np.quantile(radii, 0.98)) if len(points) >= 50 else float(np.max(radii))
    filtered = points[radii <= cutoff]
    center = np.median(filtered, axis=0)
    covariance = np.cov((filtered - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    x_axis = axes[:, 0]
    y_axis = axes[:, 1]
    if x_axis[int(np.argmax(np.abs(x_axis)))] < 0:
        x_axis *= -1
    if y_axis[int(np.argmax(np.abs(y_axis)))] < 0:
        y_axis *= -1
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    # Prefer +Z toward the scene in the reference optical frame.
    if z_axis[2] < 0:
        y_axis *= -1
        z_axis *= -1
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    pose[:3, 3] = center
    return pose, filtered


def generate_rgbd_object_poses(
    session: Path,
    image_paths: Iterable[Path],
    min_points: int = 50,
    max_points_per_frame: int = 1500,
    max_output_points: int = 4000,
) -> Dict:
    """Fuse static pre-manipulation RGB-D masks and write triangulator-compatible JSON."""
    session = Path(session).resolve()
    paths = [Path(path).resolve() for path in image_paths]
    if not paths:
        raise ValueError("no RGB frames supplied for RGB-D object pose generation")
    calibration_path = session / "raw" / "calibration.json"
    if not calibration_path.is_file():
        calibration_path = session / "calibration.json"
    calibration = _read_json(calibration_path)
    depth_scale = float(calibration["depth_scale_m"])

    cam0_document = _read_json(paths[0].with_name("aria_cam_rgb.json"))
    cam0_to_world = np.asarray(cam0_document["c2w"], dtype=np.float64)
    world_to_cam0 = np.linalg.inv(cam0_to_world)

    object_keys = sorted({
        mask.stem[len("mask_"):]
        for image in paths
        for mask in image.parent.glob("mask_obj*.png")
        if mask.stem not in {"mask_obj", "mask_objects"}
    })
    if not object_keys:
        raise FileNotFoundError("no mask_obj*.png files found in the selected frames")

    objects: Dict[str, Dict] = {}
    for object_key in object_keys:
        world_clouds: List[np.ndarray] = []
        used_frames: List[int] = []
        for image_path in paths:
            frame_dir = image_path.parent
            mask_path = frame_dir / f"mask_{object_key}.png"
            depth_path = frame_dir / "depth.png"
            camera_path = frame_dir / "aria_cam_rgb.json"
            if not mask_path.is_file() or not depth_path.is_file() or not camera_path.is_file():
                continue
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            camera = _read_json(camera_path)
            intrinsics = np.asarray(camera["k"], dtype=np.float64)
            points_cam = masked_depth_points(depth, mask, intrinsics, depth_scale)
            if len(points_cam) < min_points:
                continue
            if len(points_cam) > max_points_per_frame:
                choice = np.linspace(0, len(points_cam) - 1, max_points_per_frame, dtype=int)
                points_cam = points_cam[choice]
            c2w = np.asarray(camera["c2w"], dtype=np.float64)
            points_world = (c2w[:3, :3] @ points_cam.T + c2w[:3, 3:4]).T
            world_clouds.append(points_world)
            used_frames.append(int(camera["idx"]))
        if not world_clouds:
            raise ValueError(f"{object_key}: no frame has {min_points} valid masked depth points")

        points_world = np.concatenate(world_clouds, axis=0)
        if len(points_world) > max_output_points:
            choice = np.linspace(0, len(points_world) - 1, max_output_points, dtype=int)
            points_world = points_world[choice]
        points_cam0 = (
            world_to_cam0[:3, :3] @ points_world.T + world_to_cam0[:3, 3:4]
        ).T
        object_to_cam0, filtered_cam0 = robust_pca_pose(points_cam0)
        # Keep exported world/camera point lists paired after robust filtering.
        filtered_world = (
            cam0_to_world[:3, :3] @ filtered_cam0.T + cam0_to_world[:3, 3:4]
        ).T
        objects[object_key] = {
            "points_3d_world": filtered_world.tolist(),
            "points_3d_cam0": filtered_cam0.tolist(),
            "object_to_cam0_matrix": object_to_cam0.tolist(),
            "info": {
                "method": "realsense_rgbd_multiframe_pca",
                "used_frame_indices": used_frames,
                "point_count": len(filtered_cam0),
                "orientation_ambiguity": "PCA axes are deterministic but not semantic",
            },
        }

    document = {
        "cam0_c2w": cam0_to_world.tolist(),
        "objects": objects,
        "source": "realsense_d435i_aligned_depth",
    }
    output = session / "preprocess" / "camtriangulator_results.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
    return document
