# Fixed D435i → HumanEgo RGB-D bridge

This repository records a table/stand-mounted D435i and generates HumanEgo's existing
base-data contract without ROS, IMU, IR stereo, or ORB-SLAM3. The RGB camera frame is
the world frame by default, so every frame has the same `c2w` and no estimated-pose
jitter is injected into hand or object coordinates.

The default data path is:

```text
D435i RGB + Depth → aligned RGB-D → constant c2w → RGB-D hands/objects → HumanEgo
```

The old moving-camera Stereo-Inertial recorder remains available behind an explicit
CMake option, but it is not part of the fixed-camera build or workflow.

```text
SESSION/
├── raw/
│   ├── calibration.json
│   ├── timestamps.csv
│   ├── rgb/ depth/ aligned_depth/
│   └── capture_summary.json
└── preprocess/
    ├── aria_cam_rgb_config.json
    ├── aria_phases_results.json
    ├── realsense_manifest.json
    └── all_data/00000/
        ├── rgb.png
        ├── depth.png
        ├── depth_meta.json
        ├── aria_cam_rgb.json
        ├── aria_slam.json
        ├── aria_hands.json
        └── aria_phases.json
```

The `aria_` prefix is retained only because HumanEgo treats those names as an interface.
`aria_slam.json` is intentionally still generated, but it describes a static camera:
constant pose, zero deltas, and zero linear/angular speed.

## 1. Build the RGB-D recorder

Only librealsense2 and OpenCV are needed for the default recorder:

```bash
cd /home/tenda/aria2D435i
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

To build the legacy moving-camera recorder too, configure with
`-DBUILD_ORB_SLAM_RECORDER=ON -DORB_SLAM3_ROOT=/path/to/ORB_SLAM3`.

## 2. Record

The default serial is the attached D435i, `261622079447`:

```bash
./build/RealSenseRGBDRecorder \
  --output /absolute/path/to/data/serve_bread/realsense/rs_serve_bread_000
```

Press Ctrl+C to stop, or use `--max-frames 600`. The recorder saves color, raw depth,
and depth aligned to color at 848×480@30. It discards 60 warm-up frames by default
(`--warmup-frames N`) and refuses to overwrite a session containing `timestamps.csv`.
Use `--record` only when an additional librealsense stream recording is useful for
diagnostics; it is not needed by HumanEgo.

## 3. Export HumanEgo base data

Fixed-camera export is the CLI default. It does not require `trajectory_rgb.txt`:

```bash
python3 -m realsense_humanego \
  --session /absolute/path/to/rs_serve_bread_000 \
  --hands mediapipe \
  --manip-start 120 --manip-end 450 --finished-start 510
```

Manual phase boundaries use exported-frame indices and are inclusive. Frames outside
the manipulation/finished windows are TRANSITION. This is the recommended mode while
validating geometry and masks. Without manual boundaries, fixed-camera phase detection
uses optimized hand velocity: sufficiently long stable-hand runs are MANIPULATION,
reach/withdrawal or missing-hand frames are TRANSITION, and the final
`--finish-frames` are FINISHED. Fixed mode never emits FORWARD or ROTATE.

By default `c2w` is identity, making RGB-camera coordinates world coordinates. To use a
fixed calibrated world transform, pass `--fixed-c2w transform.json`; the file may be a
4×4 JSON array or `{\"c2w\": [[...]]}`. The transform is checked for a valid rigid
rotation before export.

With MediaPipe installed, `--hands mediapipe` detects hands and lifts every keypoint
with aligned depth. It uses a center-guided 7×7 depth cluster, requires at least eight
measured joints including three palm anchors, aligns MediaPipe's relative hand with an
SVD similarity transform, and performs temporal fallback in world coordinates. A second
sequence pass filters/interpolates tracks, applies Savitzky–Golay/EMA smoothing, rebuilds
the gripper frame, and only then writes the `*_opt_world` fields. Phase labels are generated
after this pass so hand motion can mark manipulation transitions:

```bash
python3 -m pip install -e '.[hands]'
realsense-humanego --session /absolute/path/to/session --hands mediapipe
```

Grasp labels use the image-space thumb-tip/index-tip distance normalized by wrist-to-middle-
MCP palm length. The default hysteresis closes below `0.85` and re-opens above `1.00`;
override these with `--grasp-close-ratio` and `--grasp-open-ratio` only after checking the
recording's ratio distribution.

For detections produced in another environment, use `--hands landmarks --landmarks DIR`.
Each `DIR/000000.json` (source-frame numbering) has this form; points may be pixels or
normalized coordinates:

```json
{
  "hands": [{
    "side": "right",
    "confidence": 0.95,
    "normalized": true,
    "landmarks_2d": [[0.5, 0.8]],
    "world_landmarks": [[0.0, 0.0, 0.0]]
  }]
}
```

Both landmark arrays must contain 21 rows. Missing detections still produce
`aria_hands.json` with null left/right entries, so frame discovery remains deterministic.

For legacy moving-camera captures, pass `--slam-camera`. SLAM output is then interpreted
as `timestamp tx ty tz qx qy qz qw`; translation is linearly interpolated and rotation
uses quaternion Slerp. Frames outside the tracked interval or inside a gap larger than
`--max-pose-gap` are dropped and reindexed.

## 4. Run the existing HumanEgo object pipeline

The local HumanEgo checkout has unrelated/uncommitted RealSense experiments, so this
repository does not overwrite them. The supplied runner bypasses only VRS/MPS base-data
initialization. By default it runs DINO-SAM, KptsSelector and CoTracker, replaces only
CamTriangulator's 3D point source with masked RealSense depth, then resumes LaMa,
VisualKpts and DatasetGen. Object axes still use the task's HumanEgo `pca1`, `pca2`, or
`vlm` pose method, including anchor/context constraints:

```bash
python3 scripts/run_humanego_downstream.py \
  --humanego /home/tenda/HumanEgo \
  --session /absolute/path/to/rs_serve_bread_000 \
  --task serve_bread
```

Like the official pipeline, a completed run exports two videos by default under
`SESSION/preprocess/vis/`: `aria_vis.mp4` shows RGB hand tracking, OPEN/CLOSED state,
pinch ratio, and phase; `visualkpts_vis.mp4` shows the arm-inpainted training view with
the virtual gripper and object keypoints. Video writing is streamed to keep memory bounded.
Pass `--no-video` to disable both.

Use `--from-stage`/`--to-stage` to resume or run a subset. Model weights and task prompts
remain the responsibility of the existing HumanEgo configuration. No policy/training code
changes are needed because the resulting `training_data.json` schema is unchanged.
Cached keypoint/tracker/pose JSON invalidated by the selected start stage is removed by
default; pass `--keep-stage-cache` only for deliberate cache reuse. Pass
`--object-pose triangulation` to retain HumanEgo's original bundle-adjusted 3D point source.

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

The environment's globally enabled `langsmith` pytest plugin is missing its own
`typing_extensions` dependency, hence plugin autoload is disabled for the project tests.
