# D435i → HumanEgo native bridge

This repository implements the plan in `chatGPT.md` without ROS1/ROS2 and without
pretending that a D435i capture is an Aria VRS/MPS recording.

The native C++14 executable captures color, raw/aligned depth, both IR cameras and
IMU samples, calls ORB-SLAM3 `TrackStereo()` directly, and writes a RGB-camera `c2w`
trajectory. The Python converter interpolates that trajectory onto RGB timestamps and
generates HumanEgo's existing compatibility contract:

```text
SESSION/
├── raw/
│   ├── sample.db3
│   ├── calibration.json
│   ├── orbslam3_runtime.yaml
│   ├── timestamps.csv
│   ├── imu.csv
│   ├── trajectory_left_ir.txt
│   ├── trajectory_rgb.txt
│   ├── rgb/ depth/ aligned_depth/ ir_left/ ir_right/
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

The `aria_` prefix is retained only because `DatasetGen.py` treats those names as an
interface. Every camera JSON also records `source: realsense_d435i`.

## 1. Build ORB-SLAM3 and the recorder

The specified `/home/tenda/ORB_SLAM3` checkout currently contains source only. Build it
once (including extracting its vocabulary), then build this project:

```bash
cd /home/tenda/ORB_SLAM3
tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
./build.sh

cd /home/tenda/aria2D435i
cmake -S . -B build -DORB_SLAM3_ROOT=/home/tenda/ORB_SLAM3 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

The bridge itself uses C++14. ORB-SLAM3 remains its native C++ library; no ROS wrapper
is built or used. For a prebuilt checkout elsewhere, set `ORB_SLAM3_ROOT` accordingly.

## 2. Record

The default serial is the attached D435i, `261622079447`:

```bash
./build/RealSenseStereoInertial \
  --vocabulary /home/tenda/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --output /absolute/path/to/data/serve_bread/realsense/rs_serve_bread_000
```

Press Ctrl+C to stop. Add `--max-frames 300` for a bounded test or `--no-bag` (also
available as `--no-recording`) if the raw stream recording is not wanted. The installed
ROS2 librealsense writes that recording as `raw/sample.db3`; its writer rejects the old
`.bag` suffix. The configured streams are color 1280×720@30,
depth/left IR/right IR 848×480@30, gyro 200 Hz and accelerometer 200 Hz. The projected
IR emitter is disabled for stereo feature tracking. Runtime factory calibration is
written both to `calibration.json` and to the generated ORB-SLAM3 YAML. The infrared
pair is declared as `Camera.type: Rectified` with the factory baseline, because D435i
already supplies rectified stereo frames.

At build time this project compiles a local compatibility copy of ORB-SLAM3's
`Settings.cc`, initializing its camera pointers before the `Rectified` configuration is
printed. This fixes the startup segmentation fault without modifying the external
`/home/tenda/ORB_SLAM3` checkout or changing the camera geometry.

The capture queue is bounded. If disk or SLAM processing cannot keep up,
`capture_summary.json` reports dropped frames instead of silently growing memory.

## 3. Export HumanEgo base data

For camera/SLAM/phases and explicit null hands:

```bash
python3 -m realsense_humanego \
  --session /absolute/path/to/rs_serve_bread_000 \
  --hands none
```

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

SLAM output is interpreted as `timestamp tx ty tz qx qy qz qw`. Translation is linearly
interpolated and rotation uses quaternion Slerp. Frames outside the tracked interval or
inside a gap larger than `--max-pose-gap` are dropped and reindexed; they are never filled
by copying the preceding pose. The original index and drop reasons are retained in
`realsense_manifest.json`.

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
