#!/usr/bin/env python3
"""Convert unified dual-UMI episodes to the LeRobot v2.1 layout.

The converter is intentionally self-contained.  It follows the AprilGrid to
robot-world and dual-hand plane-compensation math used by
``umi_physical_validation`` without importing or modifying that project.

Output state/action layout (14 float32 values):
    left xyz, left roll/pitch/yaw, left gripper,
    right xyz, right roll/pitch/yaw, right gripper

Positions are metres, RPY angles are radians, and grippers are normalized to
[0, 1].  Each hand pose is relative to its first synchronized frame.
``action[t]`` is exactly ``state[t + 1]``; the final action repeats the final
state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_INPUT_ROOT = Path("/home/kernel/Desktop/data/unified_episode")
DEFAULT_OUTPUT_ROOT = Path("/home/kernel/Desktop/umi-lerobot/lerobot_v21")
FPS = 20
CHUNK_SIZE = 1000
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_CODEC = "mpeg4"
GRIPPER_FULL_RANGE_RAD = 0.36

# T_source_tcp = T_source_gripper_base @ T_GRIPPER_BASE_TCP.
T_GRIPPER_BASE_TCP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.129708200693],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# AprilGrid axes -> robot-world axes, copied from the validated platform logic:
# robot_x = grid_z, robot_y = -grid_x, robot_z = -grid_y.
R_APRILGRID_TO_ROBOT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
R_SLAM_WORLD_TO_ROBOT = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

MOTOR_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_roll",
    "left_pitch",
    "left_yaw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_roll",
    "right_pitch",
    "right_yaw",
    "right_gripper",
]
CAMERA_KEYS = (
    "observation.images.left_camera_rgb_image",
    "observation.images.right_camera_rgb_image",
)


@dataclass(frozen=True)
class PoseSeries:
    time_sec: np.ndarray
    position: np.ndarray
    rotation: Rotation


@dataclass(frozen=True)
class VideoClock:
    path: Path
    start_sec: float
    duration_sec: float


@dataclass(frozen=True)
class PreparedEpisode:
    state: np.ndarray
    timestamp: np.ndarray
    global_time_sec: np.ndarray
    video_clocks: dict[str, VideoClock]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PlaneCompensation:
    rotation: np.ndarray
    measured_angle_deg: float
    original_vector: np.ndarray
    corrected_vector: np.ndarray


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _ffprobe_video(path: Path) -> dict[str, Any]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no video stream found in {path}")
    return streams[0]


def _video_duration(path: Path) -> float:
    value = _ffprobe_video(path).get("duration")
    if value is None:
        raise ValueError(f"ffprobe did not report duration for {path}")
    return float(value)


def _load_pose(path: Path) -> PoseSeries:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    required = {"Timestamp_us", "X", "Y", "Z", "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"}
    names = set(data.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(f"{path} is missing columns: {sorted(required - names)}")
    time_sec = np.asarray(data["Timestamp_us"], dtype=np.float64) * 1e-6
    position = np.column_stack([data["X"], data["Y"], data["Z"]]).astype(np.float64)
    quaternion = np.column_stack(
        [data["Quat_X"], data["Quat_Y"], data["Quat_Z"], data["Quat_W"]]
    ).astype(np.float64)
    finite = np.isfinite(time_sec) & np.isfinite(position).all(axis=1) & np.isfinite(quaternion).all(axis=1)
    time_sec, position, quaternion = time_sec[finite], position[finite], quaternion[finite]
    order = np.argsort(time_sec, kind="stable")
    time_sec, position, quaternion = time_sec[order], position[order], quaternion[order]
    time_sec, unique_index = np.unique(time_sec, return_index=True)
    position, quaternion = position[unique_index], quaternion[unique_index]
    if time_sec.size < 2:
        raise ValueError(f"not enough valid pose samples in {path}")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms <= 1e-9):
        raise ValueError(f"zero-norm quaternion in {path}")
    quaternion /= norms[:, None]
    return PoseSeries(time_sec, position, Rotation.from_quat(quaternion))


def _load_gripper(path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 2 or samples.shape[0] < 2:
        raise ValueError(f"unexpected gripper data shape in {path}: {samples.shape}")
    time_sec = samples[:, 0] * 1e-9
    angle_rad = samples[:, 1]
    finite = np.isfinite(time_sec) & np.isfinite(angle_rad)
    time_sec, angle_rad = time_sec[finite], angle_rad[finite]
    order = np.argsort(time_sec, kind="stable")
    time_sec, angle_rad = time_sec[order], angle_rad[order]
    time_sec, unique_index = np.unique(time_sec, return_index=True)
    angle_rad = angle_rad[unique_index]
    if time_sec.size < 2:
        raise ValueError(f"not enough valid gripper samples in {path}")
    # The source angle is negative when opening. 
    # opening, with 0 closed and 1 fully open.
    normalized = np.clip(-angle_rad / GRIPPER_FULL_RANGE_RAD, 0.0, 1.0)
    return time_sec, normalized


def _load_metadata(episode: Path) -> dict[str, Any]:
    path = episode / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_start_seconds(metadata: dict[str, Any], side: str) -> float:
    expected = f"cam_{side}.mkv"
    for item in metadata.get("video_details", []):
        if item.get("name") == expected and item.get("start_offset_us") is not None:
            return float(item["start_offset_us"]) * 1e-6
    raise ValueError(f"metadata.video_details has no start_offset_us for {expected}")


def _detect_input_control_point(episode: Path) -> str:
    logs = sorted((episode / "processing_logs").glob("*.log"))
    tcp_marker = "apply_base_to_tcp_trajectory.py"
    aprilgrid_marker = "Saved AprilGrid-aligned"
    for path in reversed(logs):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            tcp_at = text.rfind(tcp_marker)
            aprilgrid_at = text.rfind(aprilgrid_marker)
            if max(tcp_at, aprilgrid_at) >= 0:
                # Use the last decisive operation in the newest applicable log.
                # A full pipeline log contains both markers, with base->TCP last.
                return "tcp" if tcp_at > aprilgrid_at else "gripper-base"
        except OSError:
            continue
    raise ValueError(
        f"cannot safely infer whether {episode}/pose_data contains gripper-base or TCP poses; "
        "pass --input-control-point tcp or --input-control-point gripper-base"
    )


def _base_to_tcp(position: np.ndarray, rotation: Rotation) -> tuple[np.ndarray, Rotation]:
    offset = T_GRIPPER_BASE_TCP[:3, 3]
    tcp_position = position + rotation.apply(np.broadcast_to(offset, position.shape))
    tcp_rotation = Rotation.from_matrix(rotation.as_matrix() @ T_GRIPPER_BASE_TCP[:3, :3])
    return tcp_position, tcp_rotation


def _interpolate_pose(series: PoseSeries, target_time: np.ndarray) -> tuple[np.ndarray, Rotation]:
    position = np.column_stack(
        [np.interp(target_time, series.time_sec, series.position[:, axis]) for axis in range(3)]
    )
    rotation = Slerp(series.time_sec, series.rotation)(target_time)
    return position, rotation


def _normalized(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm <= 1e-9:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return vector / norm


def _compute_plane_compensation(initial_vector: np.ndarray) -> PlaneCompensation:
    vector = np.asarray(initial_vector, dtype=np.float64).reshape(3)
    direction = _normalized(vector, "initial left-right vector")
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    projected = direction - float(np.dot(direction, normal)) * normal
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= 1e-9:
        raise ValueError("initial dual-hand vector is parallel to the robot X axis")
    measured_rad = float(np.arcsin(np.clip(abs(float(np.dot(direction, normal))), 0.0, 1.0)))
    if measured_rad <= 1e-12:
        correction = np.eye(3, dtype=np.float64)
    else:
        target = projected / projected_norm
        axis = np.cross(direction, target)
        axis /= np.linalg.norm(axis)
        correction = Rotation.from_rotvec(axis * measured_rad).as_matrix()
    return PlaneCompensation(
        rotation=correction,
        measured_angle_deg=float(np.degrees(measured_rad)),
        original_vector=vector.copy(),
        corrected_vector=correction @ vector,
    )


def _continuous_quaternion(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _relative_pose_in_robot_world(
    position: np.ndarray,
    rotation: Rotation,
    source_to_robot: np.ndarray,
    plane_correction: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    relative_position = (source_to_robot @ (position - position[0]).T).T
    relative_matrix = rotation.as_matrix() @ rotation[0].as_matrix().T
    relative_matrix = (
        source_to_robot[None, :, :] @ relative_matrix @ source_to_robot.T[None, :, :]
    )
    relative_position = (plane_correction @ relative_position.T).T
    relative_matrix = (
        plane_correction[None, :, :] @ relative_matrix @ plane_correction.T[None, :, :]
    )
    quaternion = _continuous_quaternion(Rotation.from_matrix(relative_matrix).as_quat())
    relative_position[0] = 0.0
    quaternion[0] = np.array([0.0, 0.0, 0.0, 1.0])
    return relative_position, Rotation.from_quat(quaternion)


def _video_clocks(episode: Path, metadata: dict[str, Any]) -> dict[str, VideoClock]:
    clocks: dict[str, VideoClock] = {}
    for side in ("left", "right"):
        path = episode / "view_files" / f"cam_{side}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        clocks[side] = VideoClock(
            path=path,
            start_sec=_camera_start_seconds(metadata, side),
            duration_sec=_video_duration(path),
        )
    return clocks


def prepare_episode(
    episode: Path,
    *,
    fps: int,
    source_frame: str,
    input_control_point: str,
    plane_compensation: bool,
) -> PreparedEpisode:
    metadata = _load_metadata(episode)
    pose = {
        side: _load_pose(episode / "pose_data" / f"pose_data_{side}.csv")
        for side in ("left", "right")
    }
    gripper = {
        side: _load_gripper(episode / "view_files" / f"sensor_{side}_clamp_angle.json")
        for side in ("left", "right")
    }
    clocks = _video_clocks(episode, metadata)
    actual_control_point = (
        _detect_input_control_point(episode)
        if input_control_point == "auto"
        else input_control_point
    )
    source_to_robot = (
        R_APRILGRID_TO_ROBOT if source_frame == "aprilgrid" else R_SLAM_WORLD_TO_ROBOT
    )

    start_sec = max(
        *(pose[side].time_sec[0] for side in ("left", "right")),
        *(gripper[side][0][0] for side in ("left", "right")),
        *(clocks[side].start_sec for side in ("left", "right")),
    )
    stop_sec = min(
        *(pose[side].time_sec[-1] for side in ("left", "right")),
        *(gripper[side][0][-1] for side in ("left", "right")),
        *(clocks[side].start_sec + clocks[side].duration_sec for side in ("left", "right")),
    )
    if stop_sec <= start_sec:
        raise ValueError(f"no common sensor interval in {episode}")
    frame_count = int(math.floor((stop_sec - start_sec) * fps + 1e-7)) + 1
    if frame_count < 2:
        raise ValueError(f"common sensor interval is too short in {episode}")
    relative_time = np.arange(frame_count, dtype=np.float64) / float(fps)
    global_time = start_sec + relative_time

    absolute: dict[str, tuple[np.ndarray, Rotation]] = {}
    normalized_gripper: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        position, rotation = _interpolate_pose(pose[side], global_time)
        if actual_control_point == "gripper-base":
            position, rotation = _base_to_tcp(position, rotation)
        absolute[side] = (position, rotation)
        normalized_gripper[side] = np.interp(global_time, gripper[side][0], gripper[side][1])

    initial_vector = source_to_robot @ (absolute["left"][0][0] - absolute["right"][0][0])
    if plane_compensation:
        compensation = _compute_plane_compensation(initial_vector)
    else:
        compensation = PlaneCompensation(
            rotation=np.eye(3),
            measured_angle_deg=0.0,
            original_vector=initial_vector,
            corrected_vector=initial_vector,
        )

    relative: dict[str, tuple[np.ndarray, Rotation]] = {}
    for side in ("left", "right"):
        relative[side] = _relative_pose_in_robot_world(
            *absolute[side], source_to_robot, compensation.rotation
        )

    # scipy's lowercase xyz is fixed-axis roll/pitch/yaw
    # [roll, pitch, yaw] fields.
    left_rpy = relative["left"][1].as_euler("xyz", degrees=False)
    right_rpy = relative["right"][1].as_euler("xyz", degrees=False)
    state = np.column_stack(
        [
            relative["left"][0],
            left_rpy,
            normalized_gripper["left"],
            relative["right"][0],
            right_rpy,
            normalized_gripper["right"],
        ]
    ).astype(np.float32)
    state[0, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] = 0.0
    if not np.isfinite(state).all():
        raise ValueError(f"non-finite generated state in {episode}")

    diagnostics = {
        "source_episode": str(episode.resolve()),
        "source_frame": source_frame,
        "input_control_point": actual_control_point,
        "output_control_point": "tcp",
        "fps": fps,
        "frame_count": frame_count,
        "global_start_sec": start_sec,
        "global_stop_sec_available": stop_sec,
        "output_last_timestamp_sec": float(relative_time[-1]),
        "R_source_to_robot": source_to_robot.tolist(),
        "axis_mapping": (
            {"robot_x": "grid_z", "robot_y": "-grid_x", "robot_z": "-grid_y"}
            if source_frame == "aprilgrid"
            else {"robot_x": "-slam_world_y", "robot_y": "slam_world_x", "robot_z": "slam_world_z"}
        ),
        "relative_translation_formula": "dp_robot = R_source_to_robot @ (p_now - p_initial)",
        "relative_rotation_formula": "dR_robot = R_source_to_robot @ (R_now @ R_initial.T) @ R_source_to_robot.T",
        "plane_compensation": {
            "enabled": plane_compensation,
            "plane": "robot_yz",
            "measured_angle_deg": compensation.measured_angle_deg,
            "rotation": compensation.rotation.tolist(),
            "original_initial_left_minus_right_m": compensation.original_vector.tolist(),
            "corrected_initial_left_minus_right_m": compensation.corrected_vector.tolist(),
            "distance_m": float(np.linalg.norm(compensation.corrected_vector)),
        },
        "gripper_mapping": "clip(-angle_rad / 0.36, 0, 1)",
        "video": {
            side: {
                "source": str(clocks[side].path.resolve()),
                "source_start_sec": clocks[side].start_sec,
                "trim_offset_sec": start_sec - clocks[side].start_sec,
            }
            for side in ("left", "right")
        },
    }
    return PreparedEpisode(state, relative_time.astype(np.float32), global_time, clocks, diagnostics)


def _encode_video(
    clock: VideoClock,
    destination: Path,
    *,
    global_start_sec: float,
    frame_count: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = global_start_sec - clock.start_sec
    if offset < -1e-6:
        raise ValueError(f"negative video trim offset for {clock.path}: {offset}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clock.path),
        "-ss",
        f"{max(0.0, offset):.9f}",
        "-an",
        "-vf",
        f"fps={fps},scale={width}:{height}:flags=lanczos",
        "-frames:v",
        str(frame_count),
        "-c:v",
        VIDEO_CODEC,
        "-q:v",
        "3",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-vsync",
        "cfr",
        str(destination),
    ]
    try:
        _run(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {clock.path}: {exc.stderr}") from exc
    probe = _ffprobe_video(destination)
    encoded_frames = int(probe.get("nb_frames") or 0)
    if encoded_frames != frame_count:
        raise ValueError(
            f"encoded frame mismatch for {destination}: expected {frame_count}, got {encoded_frames}"
        )


def _numeric_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    axis = 0 if values.ndim > 1 else None
    count = int(values.shape[0]) if values.ndim else 1
    def convert(value: Any) -> Any:
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    return {
        "min": [convert(np.min(values, axis=axis))] if values.ndim == 1 else convert(np.min(values, axis=axis)),
        "max": [convert(np.max(values, axis=axis))] if values.ndim == 1 else convert(np.max(values, axis=axis)),
        "mean": [convert(np.mean(values, axis=axis))] if values.ndim == 1 else convert(np.mean(values, axis=axis)),
        "std": [convert(np.std(values, axis=axis))] if values.ndim == 1 else convert(np.std(values, axis=axis)),
        "count": [count],
    }


def _image_stats(video: Path, frame_count: int, sample_stride: int = 10) -> dict[str, Any]:
    # Approximate per-channel statistics on uniformly sampled, downscaled
    # frames. This keeps dataset creation fast; LeRobot image normalization
    # does not depend on full-resolution per-pixel statistics.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=not(mod(n\\,{sample_stride})),scale=64:64",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    pixels_per_frame = 64 * 64 * 3
    usable = raw.size - raw.size % pixels_per_frame
    if usable == 0:
        raise ValueError(f"could not sample frames for image stats: {video}")
    pixels = raw[:usable].reshape(-1, 3).astype(np.float64) / 255.0
    def shaped(values: np.ndarray) -> list[list[list[float]]]:
        return [[[float(value)]] for value in values]
    return {
        "min": shaped(pixels.min(axis=0)),
        "max": shaped(pixels.max(axis=0)),
        "mean": shaped(pixels.mean(axis=0)),
        "std": shaped(pixels.std(axis=0)),
        "count": [float(frame_count)],
    }


def _write_parquet(
    path: Path,
    *,
    state: np.ndarray,
    timestamp: np.ndarray,
    episode_index: int,
    global_start_index: int,
    task_index: int,
) -> np.ndarray:
    frame_count = len(state)
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    frame_index = np.arange(frame_count, dtype=np.int64)
    table = pa.table(
        {
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(frame_count, episode_index), type=pa.int64()),
            "index": pa.array(
                np.arange(global_start_index, global_start_index + frame_count), type=pa.int64()
            ),
            "task_index": pa.array(np.full(frame_count, task_index), type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return action


def _feature_info(fps: int, width: int, height: int) -> dict[str, Any]:
    video_feature = {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": VIDEO_CODEC,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    vector_feature = {
        "dtype": "float32",
        "shape": [14],
        "names": {"motors": MOTOR_NAMES},
    }
    scalar_features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    return {
        CAMERA_KEYS[0]: video_feature,
        CAMERA_KEYS[1]: json.loads(json.dumps(video_feature)),
        "action": vector_feature,
        "observation.state": json.loads(json.dumps(vector_feature)),
        **scalar_features,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _validate_source_episode(episode: Path) -> None:
    required = [
        episode / "metadata.json",
        episode / "pose_data" / "pose_data_left.csv",
        episode / "pose_data" / "pose_data_right.csv",
        episode / "view_files" / "cam_left.mp4",
        episode / "view_files" / "cam_right.mp4",
        episode / "view_files" / "sensor_left_clamp_angle.json",
        episode / "view_files" / "sensor_right_clamp_angle.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required source files:\n" + "\n".join(missing))


def convert_dataset(args: argparse.Namespace) -> Path:
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.episodes:
        episodes = [
            (Path(item).expanduser() if Path(item).is_absolute() else input_root / item).resolve()
            for item in args.episodes
        ]
    else:
        episodes = sorted(path.resolve() for path in input_root.glob("episode_*") if path.is_dir())
    if not episodes:
        raise ValueError(f"no episode directories selected under {input_root}")
    for episode in episodes:
        _validate_source_episode(episode)
    if output_root.exists():
        raise FileExistsError(
            f"output already exists: {output_root}. Choose a new --output-root; existing datasets are never overwritten."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()

    episodes_meta: list[dict[str, Any]] = []
    episodes_stats: list[dict[str, Any]] = []
    conversion_meta: list[dict[str, Any]] = []
    total_frames = 0
    try:
        for episode_index, episode in enumerate(episodes):
            print(f"[{episode_index + 1}/{len(episodes)}] preparing {episode.name}", flush=True)
            prepared = prepare_episode(
                episode,
                fps=args.fps,
                source_frame=args.source_frame,
                input_control_point=args.input_control_point,
                plane_compensation=not args.no_plane_compensation,
            )
            frame_count = len(prepared.state)
            chunk = episode_index // CHUNK_SIZE
            parquet_path = staging / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            action = _write_parquet(
                parquet_path,
                state=prepared.state,
                timestamp=prepared.timestamp,
                episode_index=episode_index,
                global_start_index=total_frames,
                task_index=0,
            )

            video_stats: dict[str, Any] = {}
            for side, camera_key in zip(("left", "right"), CAMERA_KEYS):
                video_path = (
                    staging
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / camera_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                print(f"    encoding {side} camera -> {video_path.relative_to(staging)}", flush=True)
                _encode_video(
                    prepared.video_clocks[side],
                    video_path,
                    global_start_sec=float(prepared.global_time_sec[0]),
                    frame_count=frame_count,
                    fps=args.fps,
                    width=args.video_width,
                    height=args.video_height,
                )
                video_stats[camera_key] = (
                    _image_stats(video_path, frame_count)
                    if not args.skip_video_stats
                    else {
                        "min": [[[0.0]], [[0.0]], [[0.0]]],
                        "max": [[[1.0]], [[1.0]], [[1.0]]],
                        "mean": [[[0.5]], [[0.5]], [[0.5]]],
                        "std": [[[0.5]], [[0.5]], [[0.5]]],
                        "count": [float(frame_count)],
                    }
                )

            frame_index = np.arange(frame_count, dtype=np.int64)
            global_index = np.arange(total_frames, total_frames + frame_count, dtype=np.int64)
            stats = {
                "action": _numeric_stats(action),
                "observation.state": _numeric_stats(prepared.state),
                "timestamp": _numeric_stats(prepared.timestamp),
                "frame_index": _numeric_stats(frame_index),
                "episode_index": _numeric_stats(np.full(frame_count, episode_index, dtype=np.int64)),
                "index": _numeric_stats(global_index),
                "task_index": _numeric_stats(np.zeros(frame_count, dtype=np.int64)),
                **video_stats,
            }
            episodes_meta.append(
                {"episode_index": episode_index, "tasks": [args.task], "length": frame_count}
            )
            episodes_stats.append({"episode_index": episode_index, "stats": stats})
            conversion_meta.append(
                {
                    "episode_index": episode_index,
                    "source_episode_name": episode.name,
                    **prepared.diagnostics,
                }
            )
            total_frames += frame_count

        meta_dir = staging / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "codebase_version": "v2.1",
            "robot_type": None,
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": len(episodes) * 2,
            "total_chunks": int(math.ceil(len(episodes) / CHUNK_SIZE)),
            "chunks_size": CHUNK_SIZE,
            "fps": args.fps,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _feature_info(args.fps, args.video_width, args.video_height),
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": args.task}])
        _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
        _write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats)
        _write_jsonl(meta_dir / "conversion.jsonl", conversion_meta)

        # Final consistency checks before making the dataset visible.
        if sum(record["length"] for record in episodes_meta) != total_frames:
            raise AssertionError("episode lengths do not sum to total_frames")
        for episode_index, record in enumerate(episodes_meta):
            chunk = episode_index // CHUNK_SIZE
            table = pq.read_table(
                staging / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet",
                columns=["action", "observation.state"],
            )
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if len(state) != record["length"] or not np.array_equal(action[:-1], state[1:]):
                raise AssertionError(f"state/action consistency failed for episode {episode_index}")
            if not np.array_equal(action[-1], state[-1]):
                raise AssertionError(f"final action padding failed for episode {episode_index}")
        staging.rename(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"completed: {output_root} ({len(episodes)} episodes, {total_frames} frames)")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="episode directory names or absolute paths; omit to convert every episode_* under input-root",
    )
    parser.add_argument("--task", default="Dual-arm manipulation task")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--video-width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--video-height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument("--source-frame", choices=("aprilgrid", "slam_world"), default="aprilgrid")
    parser.add_argument(
        "--input-control-point",
        choices=("auto", "tcp", "gripper-base"),
        default="auto",
        help="auto detects the base-to-TCP preprocessing marker in processing_logs",
    )
    parser.add_argument(
        "--no-plane-compensation",
        action="store_true",
        help="disable the common rotation that places the initial dual-hand line in the robot YZ plane",
    )
    parser.add_argument(
        "--skip-video-stats",
        action="store_true",
        help="write placeholder image statistics instead of sampling encoded frames",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.video_width <= 0 or args.video_height <= 0:
        parser.error("fps and video dimensions must be positive")
    return args


def main() -> None:
    try:
        convert_dataset(parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
