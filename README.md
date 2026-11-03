# Marvin Physical Validation

Marvin Pro 平行夹爪版本的独立 MuJoCo/Mink 轨迹回放与物理验证工具。

本项目基于VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI Data for VLA Training。https://github.com/TeleHuman/umi-vista

## 功能

- 双臂 7-DoF Mink IK/FK
- 左右平行夹爪控制
- UMI `session_ee` 文本轨迹回放
- 戴盟 UGripper base 轨迹到 TCP 的位姿转换
- 以 Marvin 双指末端中心作为 IK 控制点
- 轨迹连续性、工作空间和碰撞评分
- MuJoCo 记录机械臂与底座、左右机械臂之间的碰撞
- MuJoCo 可视化

真实机器人控制尚未接入；本项目仅支持仿真验证。

## 环境安装

使用 Conda 创建独立的 Python 3.10 环境：

```bash
cd /home/kernel/code/marvin_physical_validation
./setup_env.sh
conda activate marvin-physical-validation
```

`setup_env.sh` 会创建环境并安装依赖。环境名可通过
`MARVIN_ENV_NAME` 修改：

```bash
MARVIN_ENV_NAME=my-marvin-env ./setup_env.sh
```

模型资源已经放在 `model/`，无需额外下载：

- `model/marvin_pro/`：Marvin Pro 回放与物理验证模型
- `model/gento_luna/gento_luna.xml`：Gento Luna 模型
- `model/gento_skye/gento_skye.xml`：Gento Skye 模型

三个模型均包含各自的网格资源，不依赖外部 `Marvin_Description` 目录。轨迹
回放与评分默认使用 Marvin Pro，也可通过 `--robot` 选择 Gento Luna 或
Gento Skye。两个 Gento 模型均已安装与 Marvin Pro 相同的双指夹爪、开合关节、
腕部相机和双指末端中心 tool site，可直接参与双臂 IK、夹爪回放和轨迹评分。

## 回放轨迹

轨迹根目录应为 episode 层，例如：

```bash
./run_replay.sh \
  /home/kernel/code/umi-vista/physical_validation/converted_trajectories/episode_20260818_0009 \
  0
```

通过 `--robot` 选择仿真模型：

```bash
# 可选值：marvin_pro（默认）、gento_luna、gento_skye
./run_replay.sh /path/to/episode 0 --robot gento_luna
./run_replay.sh /path/to/episode 0 --robot gento_skye
```

也可以直接运行：

```bash
export SCORE_FOLDER_PATH=/path/to/episode
python3 replay.py 0 --robot gento_luna
```

`replay.py` 会根据 `--robot` 选择独立的机器人入口，不会让 Gento 模型经过
Marvin Pro 的适配模块：

| `--robot` | Mink 仿真入口 | 真实机器人接口 |
| --- | --- | --- |
| `marvin_pro` | `robot/marvin_pro_mink.py` | `robot/marvin_pro_interface.py` |
| `gento_luna` | `robot/gento_luna_mink.py` | `robot/gento_luna_interface.py` |
| `gento_skye` | `robot/gento_skye_mink.py` | `robot/gento_skye_interface.py` |

三个 Mink 入口分别固定绑定自己的模型。它们只复用
`robot/dual_arm_mink.py` 中与具体机器人无关的双臂 IK/FK、夹爪、碰撞检查及
MuJoCo 状态管理逻辑。目前三个 `*_interface.py` 均为真实硬件保护接口；在真实
控制器完成接入前，启用真实机器人模式会明确报错。

默认搜索三级目录：`转换目录/50hz/session_ee_*`。如果 episode 中有多套转换结果，请先通过索引确认要回放的序号。

## 目录结构

```text
marvin_physical_validation/
├── configs/
│   ├── marvin_pro.json
│   ├── gento_luna.json
│   └── gento_skye.json
├── model/
│   ├── marvin_pro/
│   ├── gento_luna/
│   └── gento_skye/
├── robot/
│   ├── dual_arm_mink.py
│   ├── marvin_pro_mink.py
│   ├── marvin_pro_interface.py
│   ├── gento_luna_mink.py
│   ├── gento_luna_interface.py
│   ├── gento_skye_mink.py
│   ├── gento_skye_interface.py
│   └── robot_joint_config.py
├── utils/
├── replay.py
├── trajectory_evaluation.py
├── convert_episode_to_robot.py
├── setup_env.sh
├── run_replay.sh
└── requirements.txt
```

## 从原始 episode 直接可视化

`convert_episode_to_robot.py` 会读取原始 episode 中的：

```text
pose_data/pose_data_left.csv
pose_data/pose_data_right.csv
view_files/sensor_left_clamp_angle.json
view_files/sensor_right_clamp_angle.json
```

原始 CSV 中的位姿是戴盟 UGripper base 位姿（戴盟文档里的TCP不是夹指中点，而是base）。转换器会先使用代码中内置的
`T_UGRIPPER_BASE_TCP` 计算戴盟 TCP 位姿：

```text
T_source_tcp = T_source_base @ T_base_tcp

T_base_tcp =
[[1, 0, 0, 0               ],
 [0, 1, 0, 0               ],
 [0, 0, 1, 0.129708200693  ],
 [0, 0, 0, 1               ]]
```

也就是 TCP 相对戴盟 base 沿局部 `+Z` 方向偏移
`0.129708200693 m`，两者坐标轴方向一致。该矩阵直接定义在`convert_episode_to_robot.py` 
得到戴盟 TCP 位姿后，转换器再执行 AprilGrid → Marvin robot-base
坐标变换，并生成以首帧为零点的相对 TCP 轨迹。MuJoCo 模型中的
`left_tool_site` 和 `right_tool_site` 位于 Marvin 两个夹指末端的中心，
因此回放时的完整控制链为：

```text
戴盟 gripper base 位姿
  → 戴盟 TCP 位姿
  → Marvin 双指末端中心目标位姿
  → Mink IK
```

Marvin 双指末端中心相对其夹爪 base 的位置约为：

```text
[0.000795, -0.002959, 0.162710] m
```

默认回放只使用原始 TCP Mink IK 和轨迹评分，不在每一帧强行避碰；这样不会为了
躲避不可避免的碰撞牺牲连续性和 workspace 分数。碰撞会被 MuJoCo 记录并计入
`collision` 分数，批量筛选时可按连续性、workspace、collision 以及总分（建议
总分大于 60）接受轨迹，其余轨迹删除。需要研究避碰方案时，才显式构造
`MarvinProMink(avoid_collisions=True)`；该模式保持 10 mm 间距，但可能牺牲
原始 TCP 跟踪，且不包含桌面约束。

底座碰撞代理使用半径 `50 mm`、高度 `1.3 m` 的中心胶囊体；它只用于 MuJoCo
碰撞评分，不会改变默认 IK 目标。

输出会缓存到本工程的 `converted_trajectories/`。转换元数据包含
`conversion_format_version`、`control_point: tcp`、实际使用的 `T_base_tcp` 以及
始终执行的 `plane_compensation`；旧缓存会自动失效并重新生成。
因此可视化脚本可以直接接收原始 episode 根目录：

```bash
python trajectory_evaluation.py \
  /home/kernel/Desktop/data/unified_episode/episode_20260818_0002 \
  --viewer \
  --robot gento_skye
```

也可以传入包含多个 `episode_*` 子目录的父目录进行批量评分：

```bash
python trajectory_evaluation.py \
  /home/kernel/Desktop/data/unified_episode
```

脚本会按 episode 名称的自然顺序逐个处理，并在最后打印每个 episode 的
总分、连续性分、工作空间分、碰撞分和状态。单个 episode 失败时，批量模式会
记录错误并继续计算其余 episode。批量计算通常不建议开启 `--viewer`。

转换器会根据首帧左右手连线与机器人 `YZ` 平面的夹角，自动计算最小旋转，并将
同一旋转应用于左右手的相对位置和相对姿态。旋转前后的双手距离保持不变：

```bash
python convert_episode_to_robot.py /path/to/episode --overwrite
```

常规回放脚本直接使用已经完成补偿的转换结果：

```bash
./run_replay.sh /path/to/converted/trajectory 0
```

启用 `--viewer` 时，如果原始 episode 的 `view_files/` 中存在
`cam_left.mp4` 和 `cam_right.mp4`，程序会在进入轨迹回放阶段时把左右相机
画面嵌入 MuJoCo viewer 的右上和右下区域，并带有 LEFT/RIGHT 标签。视频不再
创建独立窗口，因此拖动、最小化或关闭 MuJoCo 时两路真实观测会一起变化。
转换轨迹会通过 `conversion_metadata.json`
自动定位原始 episode，无需额外参数。也可以用 `CAMERA_VIEW_DIR` 显式指定
episode 根目录或 `view_files` 目录；若只需要 MuJoCo 窗口，可在调用前设置：

```bash
export MARVIN_CAMERA_VIEW=0
```

每路视频固定以 `640×360` 显示，并使用 Lanczos 高质量降采样与轻度锐化。
视频在 MuJoCo 完成回 Home、首帧位姿调整和等待阶段之后才启动。若视频比轨迹
回放短，程序会按左右视频各自的时长在开头保持第一帧，使视频末帧与轨迹末帧对齐。

显式指定视频来源的示例：

```bash
export CAMERA_VIEW_DIR=/home/kernel/Desktop/data/unified_episode/episode_20260814_0001
```

也可以单独执行转换：

```bash
python convert_episode_to_robot.py \
  /home/kernel/Desktop/data/unified_episode/episode_20260818_0002
```

如需无条件重写已有转换结果，可使用：

```bash
python convert_episode_to_robot.py \
  /home/kernel/Desktop/data/unified_episode/episode_20260818_0002 \
  --overwrite
```
