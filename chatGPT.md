## 结论

**可以只使用一台 RealSense D435i 完成 HumanEgo 数据采集，并复现 HumanEgo 真正需要的大部分 Aria 功能。**

但不能把 D435i 当作 Aria 的直接替代品：

* D435i 不会直接输出类似 Aria MPS 的闭环 6DoF 轨迹、手部追踪和半稠密地图；Intel 官方规格也明确将 D435i 标为没有独立 Tracking Module，因此这些结果必须通过软件离线计算。([Intel][1])
* Aria 有两颗 150°×120° SLAM 相机、110°×110° RGB 相机和两组高频 IMU；D435i 的深度视场约为 87°×58°，视场明显更窄。([Facebook Research][2])
* D435i 没有眼动相机，但 **HumanEgo 当前预处理与训练链路并没有使用眼动、音频、GPS**，主要使用 RGB、相机位姿、手部位姿、物体位姿和抓取状态，所以眼动缺失不是阻碍。([github.com][3])

推荐实现方式是：

> **不要伪造 `.vrs` 或 MPS 文件。新增 RealSense 数据源，但让它最终生成 HumanEgo 当前所需的兼容 JSON 和图像文件。**

---

## 一、Aria 功能与 D435i 替代关系

| HumanEgo 所需数据 | Aria 来源                         | D435i 实现                                     |
| ------------- | ------------------------------- | -------------------------------------------- |
| 第一视角 RGB      | Aria RGB 相机                     | D435i RGB                                    |
| 相机内参          | VRS 标定                          | librealsense 内参                              |
| 相机 6DoF 位姿    | Aria MPS closed-loop trajectory | D435i 双红外 + IMU，运行 ORB-SLAM3 Stereo-Inertial |
| 手部 2D 关键点     | Aria MPS Hand Tracking          | MediaPipe、WiLoR 或 HaMeR                      |
| 手部 3D 关键点     | Aria MPS                        | 2D 关键点 + D435i 对齐深度                          |
| 手部 6DoF       | MPS 手腕/掌心姿态                     | 根据腕部、掌心和 MCP 关键点构建坐标系                        |
| 抓取状态          | MPS 手指距离                        | 拇指尖与食指尖距离，加时序平滑                              |
| 物体掩码          | Grounding DINO + SAM2           | 原代码不变                                        |
| 物体 3D 位姿      | SLAM + CoTracker 多视角三角化         | RGB-D 点云为主，多视角三角化为补充                         |
| 操作阶段          | Aria 运动和手部状态                    | RealSense SLAM 速度 + 手部速度/抓取状态                |

Aria MPS 的 closed-loop trajectory 是经过多传感器批处理和闭环优化的高频轨迹，因此 D435i 软件 SLAM 很难完全达到相同稳定性。([Facebook Research][4])

---

# 二、推荐整体架构

```text
D435i 头戴采集
   │
   ├── RGB 30 FPS
   ├── Depth 30 FPS
   ├── Left IR / Right IR
   ├── Gyroscope / Accelerometer
   └── 内参、外参、硬件时间戳
             │
             ▼
RealSenseRecorder
             │
             ├── Stereo-Inertial SLAM ──▶ camera c2w
             ├── RGB 手部检测 + Depth ──▶ hand 6DoF
             ├── DINO-SAM + Depth ──────▶ object 6DoF
             └── 运动/抓取分析 ─────────▶ phase
             │
             ▼
HumanEgo 兼容文件
             │
             ▼
原始 CoTracker / LaMa / DatasetGen / Training
```

HumanEgo 的训练器最终读取的是逐帧 `training_data.json`，其中包含相机、手部和物体的 SE(3) 位姿，而不是直接读取 `.vrs`。([GitHub][3])

---

# 三、采集端设计

## 1. 建议采集的数据

每次演示保存：

```text
data/<task>/realsense/rs_<task>_000/
├── raw/
│   ├── sample.bag
│   ├── calibration.json
│   ├── timestamps.csv
│   ├── imu.csv
│   ├── rgb/
│   ├── depth/
│   ├── ir_left/
│   └── ir_right/
└── preprocess/
```

建议流配置：

```text
RGB:       1280 × 720 @ 30 FPS
Depth:      848 × 480 @ 30 FPS
IR Left:    848 × 480 @ 30 FPS
IR Right:   848 × 480 @ 30 FPS
IMU:        使用设备支持的最高稳定频率
```

使用双红外图像做 SLAM，不要依赖 RGB 做主要视觉里程计。双红外深度成像器更适合运动场景，而 RGB 主要用于 HumanEgo 的视觉输入和语义分割。

D435i 的 IMU 数据带有与深度数据协调的时间戳，可用于视觉惯性融合。([RealSense][5])

## 2. RGB 与深度对齐

采集时生成：

```python
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
aligned_depth = aligned_frames.get_depth_frame()
color = aligned_frames.get_color_frame()
```

RealSense 官方示例也是使用 `rs.align(rs.stream.color)` 将深度投影到 RGB 视角。([GitHub][6])

但对齐后的深度是重投影得到的合成图像，会出现：

* 遮挡区域空洞；
* 边缘错位；
* 最近邻重采样误差；
* RGB 和深度不同视点造成的缺失。

RealSense 官方文档也明确说明了这些对齐伪影。([GitHub][7])

因此必须同时保存：

* 原始深度；
* 对齐到 RGB 的深度；
* 原始双红外图像；
* RGB、Depth、IR、IMU 的硬件时间戳。

---

# 四、相机 6DoF：替代 Aria MPS SLAM

## 推荐方案：ORB-SLAM3 Stereo-Inertial

ORB-SLAM3 官方仓库已经包含 D435i 的 Stereo-Inertial 示例，并将校正后的左右图像和 IMU 测量传给 `TrackStereo()`。([GitHub][8])

输入：

```text
Left IR
Right IR
Gyroscope
Accelerometer
```

输出：

```text
timestamp, T_world_to_left_ir
```

HumanEgo 需要的是：

```text
T_rgb_to_world，也就是 c2w
```

如果 ORB-SLAM3 返回 `T_world_to_camera`，必须先求逆：

```python
T_left_to_world = np.linalg.inv(T_world_to_left)
```

然后利用 RealSense 出厂外参：

```python
T_rgb_to_world = T_left_to_world @ T_rgb_to_left
```

这里 `T_rgb_to_left` 必须明确表示“RGB 坐标转换到 Left IR 坐标”。

最后将 SLAM 轨迹插值到 RGB 帧时间：

* 平移：线性插值；
* 旋转：四元数 Slerp；
* 较大时间间隔或 tracking lost：标为无效，不要直接复制上一个姿态。

HumanEgo 的 `AriaCamGenerator` 本质上也只是计算并保存 `c2w`；后面的 `AriaSlamGenerator` 从相机 `c2w` 计算速度、相对位移和转动，因此 RealSense 只要产生兼容的 `AriaCam` 对象，原来的 `AriaSlamGenerator` 可以继续使用。

---

# 五、手部 3D 与 6DoF

HumanEgo 当前已经实现了 MediaPipe、WiLoR 和 HaMeR 作为 Aria MPS 的替代手部方法，并让它们输出与 `aria_hands.json` 兼容的数据。([GitHub][9])

但仓库中的 MediaPipe 实现目前通过“已知手掌尺寸 + 2D 像素尺寸”估算手腕深度，并没有利用 RealSense 的真实深度。

应将 `_recover_absolute_3d()` 改成 RGB-D 版本。

## 1. 深度反投影

对于每个 MediaPipe 关键点 `(u, v)`：

```python
patch = depth_m[v-3:v+4, u-3:u+4]
valid = patch[(patch > 0.15) & (patch < 2.0)]

if len(valid) >= min_valid_pixels:
    z = np.median(valid)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_cam = np.array([x, y, z])
```

不要直接使用单个像素。手指边缘经常出现深度空洞，建议使用：

* 7×7 邻域；
* 中位数而不是均值；
* 深度连续性过滤；
* 前后帧预测作为回退；
* MediaPipe 原有手尺寸估计作为最后回退。

D435i 的推荐工作距离从约 0.3 米开始，因此手靠得过近时会出现无效深度，需要保留视觉估深回退。([RealSense][5])

## 2. 构建手部坐标系

可沿用 HumanEgo 当前代码的方式：

```text
原点：wrist 或 palm center
Y 轴：wrist → palm center
横向参考：middle MCP → index MCP
Z 轴：手掌法向
X 轴：Y × Z
```

然后：

```python
T_hand_to_world = T_cam_to_world @ T_hand_to_cam
```

## 3. 抓取状态

继续使用 HumanEgo 当前逻辑：

```python
grasp_ratio =
    distance(thumb_tip, index_tip) /
    distance(wrist, middle_mcp)

grasp = grasp_ratio < threshold
```

再进行滑动窗口、迟滞和短脉冲抑制。HumanEgo 现有 MediaPipe 实现已经使用归一化的拇指—食指距离和时序平滑。

---

# 六、物体 6DoF

HumanEgo 原始流程是：

```text
DINO-SAM 掩码
→ CoTracker 跟踪 2D 关键点
→ 利用 SLAM 相机位姿进行多视角三角化
→ 物体 6DoF
```

([GitHub][3])

使用 D435i 后，推荐改成：

```text
DINO-SAM 掩码
   ├── aligned depth → 当前帧物体点云
   ├── CoTracker → 稳定关键点身份
   └── camera c2w → 转换到世界坐标
                    ↓
          多帧融合的物体 6DoF
```

每个物体：

1. 从 `mask_obj*.png` 获取有效深度点。
2. 深度反投影为 RGB 相机坐标点云。
3. 用 `c2w` 转换到世界坐标。
4. 去除离群点和桌面平面。
5. 平移使用点云中位数或稳健质心。
6. 朝向使用：

   * CoTracker 关键点构建局部坐标系；
   * PCA；
   * HumanEgo 已有的 `pca1`、`pca2` 或 `vlm` 朝向方法。
7. 利用前一帧朝向选择符号，避免 PCA 轴突然翻转。
8. 检测到抓取后，使用 HumanEgo 已有的 latch-and-propagate 机制，让物体跟随手部，解决遮挡。([GitHub][10])

这比完全依赖多视角三角化更适合短距离桌面操作。

---

# 七、代码改造方案

## 新增文件

```text
datacollection/
└── RealSenseRecorder.py

preprocess/
├── RealSenseCam.py
├── RealSenseSlam.py
├── RealSenseDepthHands.py
├── RealSenseObjectPose.py
└── RealSensePreprocess.py

cfg/preprocess/base/
├── RealSenseCam.yaml
├── RealSenseSlam.yaml
└── RealSenseHands.yaml
```

## 1. `RealSenseCamGenerator`

不要另建整套数据类型。直接复用：

```python
from preprocess.AriaCamTypes import AriaCam, AriaCamData
```

接口保持：

```python
class RealSenseCamGenerator:
    def get_aria_cam(self) -> AriaCam:
        ...
```

每帧填入：

```python
AriaCamData(
    idx=idx,
    ts=rgb_timestamp_ns,
    img=rgb_bgr,
    h=h,
    w=w,
    k=K_rgb,
    d=np.zeros(5),
    c2w=T_rgb_to_world,
    c2d=T_rgb_to_left_ir,
    d2w=T_left_ir_to_world,
)
```

这样可以继续调用：

```python
aria_cam.save_aria_cam_json(label="rgb")
```

现有序列化器会生成：

```text
rgb.png
aria_cam_rgb.json
aria_cam_rgb_config.json
```

其中包含 `idx`、`ts`、`k`、`c2w`、`c2d`、`d2w` 和 `fps`。([GitHub][11])

另外自行保存：

```text
depth.png
depth_meta.json
ir_left.png
ir_right.png
```

## 2. `RealSenseDepthHandsGenerator`

可以继承当前 MediaPipe 实现：

```python
class RealSenseDepthHandsGenerator(MediaPipeHandsGenerator):
    def _recover_absolute_3d(...):
        # 优先使用 aligned depth
        # 缺失时调用父类的手尺寸估深
```

返回类型继续使用现有 `AriaHands` 和 `AriaHandData`，最终保存成：

```text
aria_hands.json
```

## 3. `RealSensePreprocess`

在 `Preprocess.py` 增加：

```bash
python -m preprocess.Preprocess \
    --source realsense \
    --data_path data/serve_bread/realsense/rs_serve_bread_000 \
    --task serve_bread
```

建议结构：

```python
if source == "aria":
    self.preprocess_aria()
elif source == "realsense":
    self.preprocess_realsense()
```

当前 `Preprocess.py` 的初始化直接创建 VRS provider，并读取 MPS 的 `hand_tracking_results.csv`，因此直接把 RealSense 目录传给当前入口会失败。([GitHub][9])

---

# 八、必须生成的兼容文件

`DatasetGen.py` 当前硬编码要求每帧存在：

```text
aria_cam_rgb.json
aria_hands.json
aria_phases.json
aria_slam.json
```

缺少任意一个就会丢弃这一帧。([GitHub][10])

所以 RealSense 预处理最终至少应形成：

```text
preprocess/all_data/00000/
├── rgb.png
├── depth.png
├── aria_cam_rgb.json
├── aria_hands.json
├── aria_slam.json
├── aria_phases.json
├── mask_obj1.png
├── mask_obj2.png
├── mask_arm.png
└── training_data.json
```

这些文件继续使用 `aria_` 名称只是兼容旧接口，并不表示数据来源仍然是 Aria。

---

# 九、训练端是否需要修改

模型本身不需要修改。

HumanEgo 策略实际接收：

1. RGB 或去除手臂后的 RGB；
2. 手和物体组成的 Interaction-Centric Tokens；
3. 未来手部 6DoF 轨迹作为监督目标。

因此只要 RealSense 数据最终提供相同的 `training_data.json`，策略网络并不关心输入最初来自 Aria 还是 D435i。([GitHub][12])

需要修改的只是数据发现方式，例如：

```yaml
data_sources:
  realsense: 20
eval_source: realsense
```

或者直接使用训练器支持的显式 session 路径。HumanEgo 文档说明训练器支持显式指定 train/eval session。([GitHub][12])

---

# 十、头戴安装与采集规范

由于 D435i 视场比 Aria 窄，安装和动作规范非常重要：

* 安装在额头中央，而不是胸前。
* 光轴向下倾斜约 10°–20°。
* 手在正常操作位置时应位于图像中央偏下区域。
* 双手任务必须先确认左右手同时能进入视野。
* 固定 USB 线，避免线缆拉动相机。
* 每段演示开始时静止 2–3 秒，随后进行轻微平移和旋转，帮助视觉惯性初始化。
* 避免快速甩头。
* 避免透明、纯黑、高反光物体；这些物体的深度往往不稳定。
* 物体和手尽量保持在约 0.35–1.2 米距离。

视场不足是 D435i 与 Aria 最大的硬件差距。Aria SLAM 相机达到 150°×120°，而 D435i 深度视场约为 87°×58°。([Facebook Research][2])

---

## 推荐实施顺序

第一阶段只实现：

```text
RealSenseRecorder
→ ORB-SLAM3 c2w
→ RGB/Depth/内参导出
→ aria_cam_rgb.json
→ aria_slam.json
```

第二阶段增加：

```text
MediaPipe 2D 手部
→ RealSense Depth 3D 手部
→ aria_hands.json
```

第三阶段增加：

```text
DINO-SAM
→ RGB-D 物体点云
→ 物体 6DoF
→ DatasetGen
```

第四阶段再处理：

```text
阶段自动分割
时序优化
抓取 latching
混合 Aria/RealSense 训练
```

**最合理的最小改造路线是：让 `RealSenseCamGenerator` 返回现有 `AriaCam`，让 `RealSenseDepthHandsGenerator` 返回现有 `AriaHands`，保留 HumanEgo 后半段流水线不变。** 这样改动最少，同时能够真正利用 D435i 的深度优势。

[1]: https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html?utm_source=chatgpt.com "Intel® RealSense™ Depth Camera D435i"
[2]: https://facebookresearch.github.io/projectaria_tools/docs/tech_spec/hardware_spec "Hardware Specifications | Aria Gen 1 Docs"
[3]: https://github.com/TX-Leo/HumanEgo/blob/main/preprocess/README.md "HumanEgo/preprocess/README.md at main · TX-Leo/HumanEgo · GitHub"
[4]: https://facebookresearch.github.io/projectaria_tools/docs/ARK/mps "Machine Perception Services (MPS) | Aria Gen 1 Docs"
[5]: https://realsenseai.com/products/depth-camera-d435i/ "D435i - RealSense"
[6]: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py "librealsense/wrappers/python/examples/align-depth2color.py at master · realsenseai/librealsense · GitHub"
[7]: https://github.com/IntelRealSense/librealsense/blob/master/examples/align/rs-align.cpp "librealsense/examples/align/rs-align.cpp at master · realsenseai/librealsense · GitHub"
[8]: https://github.com/UZ-SLAMLab/ORB_SLAM3?utm_source=chatgpt.com "ORB-SLAM3: An Accurate Open-Source Library for Visual ..."
[9]: https://github.com/TX-Leo/HumanEgo/blob/main/preprocess/Preprocess.py "HumanEgo/preprocess/Preprocess.py at main · TX-Leo/HumanEgo · GitHub"
[10]: https://github.com/TX-Leo/HumanEgo/blob/main/preprocess/DatasetGen.py "HumanEgo/preprocess/DatasetGen.py at main · TX-Leo/HumanEgo · GitHub"
[11]: https://github.com/TX-Leo/HumanEgo/blob/main/preprocess/AriaCamTypes.py "HumanEgo/preprocess/AriaCamTypes.py at main · TX-Leo/HumanEgo · GitHub"
[12]: https://github.com/TX-Leo/HumanEgo/blob/main/training/README.md "HumanEgo/training/README.md at main · TX-Leo/HumanEgo · GitHub"

注意：因为本地配置比较新，支持c++14，但ORB-SLAM3是C++11,所以使用ORB-SLAM3原生C++，不要使用ros2、ros1.由于最终目标是替代 Aria 数据采集 HumanEgo，建议不要直接编译 ORB-SLAM3 ROS wrapper，而是：

fork ORB-SLAM3
写一个 RealSenseStereoInertial.cc
直接调用：
SLAM.TrackStereo(
    imLeft,
    imRight,
    timestamp,
    vImuMeas
);

输出：

timestamp tx ty tz qx qy qz qw

然后 Python 转：

aria_slam.json。相关项目路径：/home/tenda/ORB_SLAM3，/home/tenda/HumanEgo,本机已经连接realsense D435i，编号261622079447 。