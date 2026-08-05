我对照阅读了 `aria2D435i` 和 HumanEgo 的预处理代码。**目前不是一个单点误差，而是几处接口语义错位叠加。**其中前 3 项足以解释你看到的 `visualkpts_vis.mp4` 精度很差。

以下结论来自源码静态检查；尚未对你某个具体 session 的 JSON 数值做统计验证。

# 一、最严重问题：默认 RGB-D 模式跳过了 CoTracker，但 VisualKpts 仍读取 CoTracker 结果

你的运行脚本中：

```python
if args.object_pose == "rgbd" and stage in {
    "kptsselector", "cotracker", "camtriangulator"
}:
    generate_rgbd_object_poses(...)
    continue
```

这意味着默认的：

```bash
--object-pose rgbd
```

会同时跳过：

```text
KptsSelector
CoTracker
CamTriangulator
```

但 HumanEgo 的 `VisualKpts` 并不会读取 `camtriangulator_results.json` 来绘制物体点。它只会检查：

```text
preprocess/cotracker_results.json
```

并从其中的二维 `tracks[g_idx]` 绘制物体关键点和轨迹。

因此：

> 假如你使用 `rgbd` 模式后，视频里依然出现了物体彩色关键点，那么这些点几乎肯定来自旧的 `cotracker_results.json`，不是本次运行产生的。

`reset_cotracker_offline()` 只重置 Python 全局状态，并不会删除硬盘上的旧 JSON；`VisualKpts` 只要发现旧文件存在就会直接读取。

## 正确修改

RGB-D 只应替换 `CamTriangulator`，不应跳过 KptsSelector 和 CoTracker：

```python
rgbd_pose_done = False

for stage in STAGES[start:end + 1]:
    if args.object_pose == "rgbd" and stage == "camtriangulator":
        print("[RealSense bridge] HumanEgo stage: rgbd_object_pose")
        generate_rgbd_object_poses(
            session,
            [Path(path) for path in engine.object_centric_image_list],
        )
        rgbd_pose_done = True
        continue

    print(f"[RealSense bridge] HumanEgo stage: {stage}")
    methods[stage]()
```

修改后的流程应为：

```text
DINO-SAM
→ KptsSelector
→ CoTracker
→ RGB-D object pose
→ LaMa
→ VisualKpts
→ DatasetGen
```

同时，在从头运行时删除旧结果：

```python
stale_files = [
    session / "preprocess" / "kptsselector_results.json",
    session / "preprocess" / "cotracker_results.json",
    session / "preprocess" / "camtriangulator_results.json",
]

for path in stale_files:
    path.unlink(missing_ok=True)
```

先检查当前 session：

```bash
stat \
  /home/tenda/HumanEgodata/data/serve_bread/realsense/rs_serve_bread_000/preprocess/cotracker_results.json
```

若文件修改时间早于本次运行，就是已经确认的缓存污染。

---

# 二、ORB-SLAM3 的 IR 图像被错误地再次整流

这是采集端最重要的问题。

ORB-SLAM3 官方 D435i 示例明确使用：

```yaml
Camera.type: "Rectified"
Stereo.b: ...
```

官方 D435i 示例代码在把左右 IR 图传给 `TrackStereo()` 前也明确注明：

```cpp
// Stereo images are already rectified.
SLAM.TrackStereo(im, imRight, timestamp, vImuMeas);
```

也就是 librealsense 输出的 D435i 左右 IR 流被当作**已经整流的立体图像**。

但你的代码动态生成的是：

```yaml
Camera.type: "PinHole"
Camera1.fx: ...
Camera2.fx: ...
Stereo.T_c1_c2: ...
```

`Camera.type=PinHole` 会使 ORB-SLAM3 设置 `bNeedToRectify_=true`，然后调用 `stereoRectify()` 和 `initUndistortRectifyMap()` 再做一次整流。更严重的是，你虽然从 RealSense 读取并保存了畸变系数，却没有把左右 IR 的 `k1/k2/p1/p2` 写入 ORB-SLAM3 YAML，因此这个二次整流使用的是空畸变参数。

这可能造成：

* 左右特征垂直不对齐；
* 视差质量下降；
* 特征匹配数量下降；
* SLAM 位姿抖动；
* IMU 初始化更不稳定；
* 世界坐标中的手和物体一起漂移。

## 推荐修改

动态 YAML 改成：

```cpp
const Eigen::Matrix4f right_to_left =
    extrinsics_matrix(right.get_extrinsics_to(left));

const float baseline =
    right_to_left.block<3, 1>(0, 3).norm();

out << "%YAML:1.0\n---\n"
    << "File.version: \"1.0\"\n"
    << "Camera.type: \"Rectified\"\n"
    << "Camera1.fx: " << left_intrinsics.fx << "\n"
    << "Camera1.fy: " << left_intrinsics.fy << "\n"
    << "Camera1.cx: " << left_intrinsics.ppx << "\n"
    << "Camera1.cy: " << left_intrinsics.ppy << "\n"
    << "Stereo.b: " << baseline << "\n"
    << "Camera.width: " << left_intrinsics.width << "\n"
    << "Camera.height: " << left_intrinsics.height << "\n"
    << "Camera.fps: 30\n"
    << "Camera.RGB: 1\n"
    << "Stereo.ThDepth: 40.0\n";
```

不再写：

```yaml
Camera2.*
Stereo.T_c1_c2
```

你的注释提到 `Rectified` 分支在所用 ORB-SLAM3 版本中会因 `originalCalib2_` 为空而崩溃。不要通过改成 `PinHole` 绕过，因为这改变了几何处理。应该在 ORB-SLAM3 的 `Settings::operator<<` 中做空指针保护：

```cpp
if (settings.originalCalib2_ != nullptr) {
    for (size_t i = 0; i < settings.originalCalib2_->size(); ++i) {
        // existing print code
    }
}
```

或者同步到已经修复该打印问题的分支。

## 不需要修改的部分

你对 ORB 输出位姿方向的处理是正确的：

```cpp
world_to_left = slam.TrackStereo(...);
left_to_world = world_to_left.inverse();
color_to_world = left_to_world * color_to_left;
```

这里没有发现 `c2w/w2c` 反转错误。

`IMU.T_b_c1` 使用 `left.get_extrinsics_to(gyro)`，以及 `Stereo.T_c1_c2` 原来使用 `right.get_extrinsics_to(left)`，方向也和 ORB-SLAM3 的实际约定相符，不要再额外求逆。

由于整流错误发生在 SLAM 输入端，修复后需要**重新采集或至少用 bag 重新运行 ORB-SLAM3**。仅重新运行 HumanEgo 下游不能修复已有轨迹。

---

# 三、你把原始手部姿态直接伪装成了优化姿态

HumanEgo 的 VisualKpts 和 DatasetGen 优先使用：

```text
midpoint_pose_opt_world
midpoint_translation_opt_world
midpoint_orientation_opt_world
```

DatasetGen 也直接把这些字段作为训练中的手部 SE(3)。

但你的 `hands.py` 中：

```python
"wrist_pose_raw_world": wrist_pose_world,
"wrist_pose_opt_world": wrist_pose_world,

"wrist_lin_vel_raw_world": linear_velocity,
"wrist_lin_vel_opt_world": linear_velocity,

"midpoint_pose_raw_world": midpoint_pose,
"midpoint_pose_opt_world": midpoint_pose,
```

也就是 `raw` 和 `opt` 完全相同，没有实际优化。

HumanEgo 原版并不是这样。它在所有帧检测结束后依次执行：

* 低置信度过滤；
* 短检测段抑制；
* 短缺失插值；
* 抓取状态平滑；
* `AriaHandsOptimizer` 全序列优化。

优化器使用 Savitzky–Golay 平滑位置、EMA 平滑旋转基向量、Gram–Schmidt 正交化、符号一致性处理，并重新计算优化后的速度。

这直接解释了视频中虚拟夹爪的：

* 高频抖动；
* 长度变化；
* 突然旋转；
* 左右摆动；
* 速度异常。

## 修复方式

不要在 `HandProcessor.process()` 中直接填写 `opt` 字段。第一遍只写：

```python
"midpoint_pose_raw_world": ...,
"midpoint_translation_raw_world": ...,
"midpoint_orientation_raw_world": ...,

"midpoint_pose_opt_world": None,
"midpoint_translation_opt_world": None,
"midpoint_orientation_opt_world": None,
```

整段 session 完成后，再做一个 sequence-level postprocess：

```text
所有 aria_hands.json
→ 按左右手组成连续轨迹
→ 删除异常点
→ 插值短缺失
→ 平滑位置
→ 平滑 SO(3)
→ 重建夹爪坐标系
→ 回写 *_opt_world
```

最省代码的路线是把字典转换成 HumanEgo 的 `AriaHands/AriaHandData`，直接调用：

```python
optimizer = AriaHandsOptimizer(cfg, dt)
optimizer.run(aria_hands)
```

然后重新序列化。

---

# 四、上一帧深度回退使用了错误的坐标系

当前代码保存：

```python
history["points_cam"]
```

它是**上一帧 RGB 相机坐标系**中的 3D 点。

下一帧发生深度缺失时，代码直接做：

```python
old = previous_3d[index]
old_uv = K @ old
z = old[2]
result[index] = backproject(current_uv, z)
```

但头戴相机在移动，上一帧的相机坐标不等于当前帧相机坐标。相机转动或平移后，直接使用上一帧 `x/y/z` 是几何错误。

## 正确方式

历史点应保存在世界坐标：

```python
history["points_world"] = points_world
```

下一帧回退时，先转换到当前相机：

```python
current_w2c = np.linalg.inv(color_to_world)

old_world_h = np.append(previous_points_world[index], 1.0)
old_current_cam = current_w2c @ old_world_h

if old_current_cam[2] > 0:
    old_uv = np.array([
        fx * old_current_cam[0] / old_current_cam[2] + cx,
        fy * old_current_cam[1] / old_current_cam[2] + cy,
    ])
```

再根据当前二维检测位置是否接近来决定是否使用。

建议函数签名改为：

```python
recover_keypoints_rgbd(
    ...,
    previous_points_world=None,
    current_color_to_world=None,
)
```

不要继续传 `previous_3d=history["points_cam"]`。

---

# 五、深度有效性控制过于宽松

当前逻辑只在：

```python
depth_hits == 0 and fallback is None
```

时拒绝结果。

这意味着：

```text
21 个关节中只有 1 个真实深度
其余 20 个全部使用视觉回退
```

仍会被当作有效 RGB-D 手部姿态。

同时你虽然保存了：

```json
"depth_keypoints_valid": 3
```

但 `confidence` 仍然只是 MediaPipe 的检测/左右手分类置信度：

```python
"confidence": float(detection.confidence)
```

DatasetGen 只检查 `confidence`，不会检查 `depth_keypoints_valid`。于是一个 MediaPipe 检测很确信、但 3D 深度很差的手，也会进入最终训练数据。

## 建议最低门限

不要只看总数量，还要看掌心关键点：

```python
required_mp_indices = {
    0,   # wrist
    5,   # index MCP
    9,   # middle MCP
    13,  # ring MCP
    17,  # pinky MCP
}
```

例如：

```python
valid_depth = np.asarray(depth_valid_flags)
palm_hits = sum(valid_depth[i] for i in required_mp_indices)
depth_ratio = float(valid_depth.mean())

pose_valid = (
    palm_hits >= 3
    and valid_depth.sum() >= 8
)

if not pose_valid:
    return None
```

并构建真正的 3D 质量分数：

```python
pose_confidence = (
    detection.confidence
    * min(1.0, valid_depth.sum() / 12.0)
    * temporal_consistency_score
    * bone_consistency_score
)
```

写入：

```python
"confidence": pose_confidence,
"detection_confidence": detection.confidence,
"depth_keypoints_valid": int(valid_depth.sum()),
```

---

# 六、7×7 深度中值容易采到桌面或物体，而不是手指

你的 `patch_depth()`：

```python
valid = depth_patch.flatten()
median = np.median(valid)
continuous = valid[np.abs(valid - median) <= 0.06]
return np.median(continuous)
```

它没有使用中心像素，也没有区分前景手指与背景。手指边缘通常只占 7×7 邻域中的少量像素；桌面或被抓物体占多数时，中值会落到背景表面。

特别是：

* 拇指尖；
* 食指尖；
* 手指与杯子/面包接触位置；
* 手掌边缘；

最容易被抬到桌面或物体深度。

## 更稳妥的深度选择

优先围绕中心像素深度选簇：

```python
def patch_depth_center_guided(depth_m, u, v, radius=3):
    x = int(round(u))
    y = int(round(v))

    patch = depth_m[
        max(0, y-radius):y+radius+1,
        max(0, x-radius):x+radius+1,
    ]

    values = patch[
        np.isfinite(patch)
        & (patch >= 0.15)
        & (patch <= 2.0)
    ]

    if values.size < 4:
        return None

    center_z = depth_m[y, x]

    if np.isfinite(center_z) and 0.15 <= center_z <= 2.0:
        cluster = values[np.abs(values - center_z) < 0.025]
        if cluster.size >= 3:
            return float(np.median(cluster))

    # 无中心深度时，选择最近的稳定深度簇，而不是全局中值
    values = np.sort(values)
    groups = np.split(
        values,
        np.where(np.diff(values) > 0.025)[0] + 1,
    )
    groups = [group for group in groups if len(group) >= 3]

    if not groups:
        return None

    nearest = min(groups, key=lambda group: np.median(group))
    return float(np.median(nearest))
```

对于手，通常优先选最近的连续表面比选全局中值更合理，但仍要通过掌部深度和骨长一致性排除误选。

---

# 七、MediaPipe 回退坐标结构未经对齐

当前回退：

```python
return wrist_3d + (
    world_landmarks - world_landmarks[0]
)
```

这里 `wrist_3d` 是 RealSense 光学相机坐标，而 MediaPipe `world_landmarks` 是手中心的相对 3D 结构。代码没有估计两者之间的旋转或相似变换，直接相加。

HumanEgo 自带的 MediaPipe 实现也使用了类似近似，但它随后执行全序列清理和 `AriaHandsOptimizer`；你的桥接代码保留了近似，却省略了后续优化。

更正确的处理是：

1. 使用有可靠深度的关节；
2. 取得这些关节的 MediaPipe 相对 3D；
3. 用 Umeyama/SVD 求 MediaPipe 手坐标到 RGB 相机坐标的相似变换；
4. 再用该变换补齐没有深度的关节。

形式为：

```text
p_camera ≈ scale · R · p_mediapipe + t
```

而不是：

```text
p_camera = wrist_camera + p_mediapipe_relative
```

---

# 八、阶段标签与 HumanEgo 的语义不一致

你的 `_phase_modes()` 只根据相机线速度和角速度分类：

```text
0 = STOP
1 = FORWARD
2 = ROTATE
4 = FINISHED
```

从来不会生成 mode 3，而且完全不使用手部运动。

HumanEgo 原始逻辑在基础相机运动分类后，还会使用手部速度修正 manipulation 边界，并生成 Transition。

这很重要，因为 `Preprocess.preprocess_indices()` 直接规定：

```python
manip_image_list       = modes 0, 4
nav_image_list         = modes 1, 2
transition_image_list  = mode 3
raw_manip_image_list   = modes 0, 3, 4
```

并用第一个 `raw_manip_image` 的位置确定 object-centric 窗口。

因此，当前代码会把：

* 开始时等待；
* 站着不动但没有操作；
* 相机静止但手没有出现；

全部当作 manipulation。

这会进一步导致：

* DINO-SAM 处理错误时间段；
* object reference frame 选择错误；
* RGB-D 物体点云来自不正确窗口；
* 导航/操作训练样本标签错误。

## 修复建议

最佳方式是复用 HumanEgo 的 phase 逻辑：

```text
导出 AriaSlam
+ 导出 AriaHands
→ AriaPhasesGenerator / AriaPhasesOps
→ aria_phases_results.json
```

至少也应把手部存在和手部速度加入判断：

```python
camera_stopped = linear_speed < v_th and angular_speed < w_th
hand_active = (
    hand_present
    and hand_midpoint_speed > hand_speed_th
)

if camera_stopped and hand_active:
    mode = 0      # manipulation
elif near_manip_boundary:
    mode = 3      # transition
elif rotating:
    mode = 2
else:
    mode = 1
```

阶段生成应放在手部序列生成和优化之后，而不是手部生成之前。

---

# 九、RGB-D PCA 物体朝向没有复现 HumanEgo 的任务级坐标系

你的 `robust_pca_pose()` 根据点云 PCA 轴和相机 0 坐标中的轴分量决定符号：

```python
if x_axis[argmax(abs(x_axis))] < 0:
    x_axis *= -1

if z_axis[2] < 0:
    z_axis *= -1
```

代码自己也在输出中标明：

```json
"orientation_ambiguity":
"PCA axes are deterministic but not semantic"
```



HumanEgo 原始 `CamTriangulator` 会读取每个物体的 `pose_method`，支持 `pca1`、`pca2` 等方式，并针对 anchor/context object 使用 anchor center 约束方向，而不是仅按当前相机坐标轴固定符号。

DatasetGen 随后把 `object_to_cam0_matrix` 直接作为静态物体坐标系，并将关键点转换到这个局部坐标系。若不同 session 的 PCA 轴出现 90°/180°差异，模型看到的同一个物体 token 坐标定义会不一致。

## 推荐处理

保留 RGB-D 点云生成，但把坐标系估计交回 HumanEgo：

```python
from preprocess.CamTriangulator import (
    estimate_frame_pca1,
    estimate_frame_pca2,
)
```

概念上改为：

```python
if pose_method == "pca1":
    T_o2c0, info = estimate_frame_pca1(
        pts_cam=filtered_cam0,
        is_anchor=is_anchor,
        anchor_center_cam=anchor_center_cam,
    )
elif pose_method == "pca2":
    T_o2c0, info = estimate_frame_pca2(
        pts_cam=filtered_cam0,
        is_anchor=is_anchor,
        anchor_center_cam=anchor_center_cam,
    )
```

也就是：

> 用 RealSense depth 替换 3D 点的来源，但不要替换 HumanEgo 的物体坐标系定义。

---

# 推荐修复优先级

## P0：必须先修

1. 将 ORB-SLAM3 D435i 输入恢复为 `Rectified`，避免错误二次整流。
2. RGB-D 模式只替换 CamTriangulator，继续运行 KptsSelector 和 CoTracker。
3. 每次完整运行前清理旧的 `cotracker_results.json`。
4. 不再把 raw hand pose 直接复制到 `*_opt_world`。

## P1：决定训练数据是否可用

5. 修复上一帧相机坐标回退。
6. 加入真实的 RGB-D pose confidence 和最低深度门限。
7. 加入全序列手部优化。
8. 使用手部运动重新生成 phase labels。

## P2：提高多 session 一致性

9. 使用 HumanEgo 的 `pca1/pca2` 物体坐标系逻辑。
10. 加入 ORB 轨迹跳变、重定位和地图重置检测。

---

# 建议你立即做的验证

先运行：

```bash
SESSION=/home/tenda/HumanEgodata/data/serve_bread/realsense/rs_serve_bread_000

grep -E \
'Camera.type|Stereo.b|Stereo.T_c1_c2|Camera1.fx|Camera2.fx' \
"$SESSION/raw/orbslam3_runtime.yaml"

stat "$SESSION/preprocess/cotracker_results.json"

python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

session = Path(
    "/home/tenda/HumanEgodata/data/serve_bread/realsense/"
    "rs_serve_bread_000"
)

counts = []
same_raw_opt = 0
hands_total = 0

for path in sorted(
    (session / "preprocess" / "all_data").glob("*/aria_hands.json")
):
    data = json.loads(path.read_text())
    for side in ("hand_r", "hand_l"):
        hand = data.get(side)
        if not hand:
            continue

        hands_total += 1
        counts.append(hand.get("depth_keypoints_valid", -1))

        raw = np.asarray(hand.get("midpoint_pose_raw_world"))
        opt = np.asarray(hand.get("midpoint_pose_opt_world"))

        if raw.shape == (4, 4) and opt.shape == (4, 4):
            same_raw_opt += int(np.allclose(raw, opt))

print("hand observations:", hands_total)
print("raw == opt:", same_raw_opt, "/", hands_total)

if counts:
    print("depth valid min/median/max:",
          min(counts), np.median(counts), max(counts))
    print("< 8 depth joints:",
          sum(value < 8 for value in counts), "/", len(counts))
PY
```

按当前源码，我预计你会看到：

```text
Camera.type: "PinHole"
raw == opt: 接近 100%
大量帧 depth_keypoints_valid < 8
cotracker_results.json 时间早于当前 RGB-D 运行
```

其中前两项是源码已经能确定的问题；第三、第四项需要由你的 session 输出确认。

**建议先修复 ORB 整流、CoTracker 跳过和手部优化这三项，再用 `.bag` 重放生成一套全新 session。旧目录不要复用，否则缓存文件会继续干扰判断。**
