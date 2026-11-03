#!/usr/bin/env python3
"""Test replay after matching the complete initial left-right base vector."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TRAJECTORY_DIR = Path(
    "/home/kernel/Desktop/data/unified_episode/episode_20260818_0009"
)
DEFAULT_GOOD_OUTPUT_DIR = Path("/home/kernel/Desktop/扎带")
R_GRID_TO_ROBOT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=float
)
R_SLAM_WORLD_TO_ROBOT = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_dir", nargs="?", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--robot",
        choices=("marvin_pro", "gento_luna", "gento_skye"),
        default=os.environ.get("ROBOT_NAME", "marvin_pro"),
        help="simulation model (default: marvin_pro)",
    )
    parser.add_argument(
        "--good-score-threshold",
        type=float,
        default=60.0,
        help="minimum total score for an accepted trajectory (default: 60)",
    )
    parser.add_argument(
        "--good-output-dir",
        type=Path,
        default=DEFAULT_GOOD_OUTPUT_DIR,
        help=(
            "directory where GOOD episode data is copied "
            "(default: /home/kernel/Desktop/扎带)"
        ),
    )
    return parser.parse_args()


def source_initial_control_point_vector(metadata_path: Path) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    direct_vector = metadata.get("initial_control_point_vector_robot_m")
    if direct_vector is not None:
        vector = np.asarray(direct_vector, dtype=float)
        if vector.shape == (3,) and np.all(np.isfinite(vector)):
            return vector

    # Compatibility fallback for metadata written by older converters. This
    # uses nearest source samples and is less precise than the synchronized
    # vector written by convert_episode_to_robot.py.
    source = Path(metadata["source"])
    stats = metadata["stats"]
    positions = {}
    for side in ("left", "right"):
        samples = np.genfromtxt(
            source / "pose_data" / f"pose_data_{side}.csv",
            delimiter=",",
            names=True,
            dtype=float,
        )
        index = int(stats[side]["source_initial_index"])
        position = np.array([samples["X"][index], samples["Y"][index], samples["Z"][index]])
        positions[side] = position
    source_frame = metadata.get("source_frame", "aprilgrid")
    source_to_robot = R_SLAM_WORLD_TO_ROBOT if source_frame == "slam_world" else R_GRID_TO_ROBOT
    return source_to_robot @ (positions["left"] - positions["right"])


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _is_episode_input(path: Path) -> bool:
    return (path / "pose_data").is_dir() or (path / "50hz").is_dir()


def discover_episode_inputs(input_path: Path) -> tuple[list[Path], bool]:
    """Return episode inputs and whether ``input_path`` is a batch root."""
    if _is_episode_input(input_path):
        return [input_path], False
    episodes = sorted(
        (
            child
            for child in input_path.iterdir()
            if child.is_dir() and _is_episode_input(child)
        ),
        key=_natural_key,
    )
    if not episodes:
        raise FileNotFoundError(
            f"No episode directories containing pose_data/ or 50hz/ found under {input_path}"
        )
    return episodes, True


def _format_score(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "None"


def _is_good_row(row: dict[str, Any], good_score_threshold: float) -> bool:
    """Return whether a scored row meets the same GOOD status criteria used in reports."""
    if row.get("error") is not None or row.get("severe_collision"):
        return False
    try:
        total = row.get("total")
        return total is not None and float(total) > good_score_threshold
    except (TypeError, ValueError):
        return False


def _copy_good_results(
    rows: list[dict[str, Any]],
    destination: Path,
    good_score_threshold: float,
) -> int:
    """Copy complete input episode directories for rows classified as GOOD.

    ``dirs_exist_ok=True`` makes rerunning an evaluation update an existing
    episode directory without deleting unrelated files in the destination.
    """
    good_rows = [row for row in rows if _is_good_row(row, good_score_threshold)]
    if not good_rows:
        print("\n[GOOD COPY] no GOOD episodes to copy")
        return 0

    destination = destination.expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[GOOD COPY] failed to create destination {destination}: {exc}")
        return 0
    copied = 0
    for row in good_rows:
        source = Path(str(row["input_path"])).expanduser().resolve()
        target = destination / source.name
        if not source.is_dir():
            print(f"[GOOD COPY] skipped {source}: source directory does not exist")
            continue
        if source == target.resolve():
            print(f"[GOOD COPY] skipped {source}: source and destination are identical")
            continue
        try:
            shutil.copytree(source, target, dirs_exist_ok=True)
        except OSError as exc:
            print(f"[GOOD COPY] failed {source} -> {target}: {exc}")
            continue
        copied += 1
        print(f"[GOOD COPY] {source} -> {target}")
    print(f"[GOOD COPY] copied {copied}/{len(good_rows)} GOOD episode(s) to {destination}")
    return copied


def _run_episode(
    input_path: Path,
    trajectory_dir: Path,
    args: argparse.Namespace,
    replay: Any,
    home_vector: np.ndarray,
) -> dict[str, Any]:
    print(f"[TEST] input episode: {input_path}")
    print(f"[TEST] replay trajectory directory: {trajectory_dir}")
    os.environ["SCORE_FOLDER_PATH"] = str(trajectory_dir)
    replay.SCORE_FOLDER_PATH = str(trajectory_dir)

    metadata_path = trajectory_dir / "conversion_metadata.json"
    target_vector = source_initial_control_point_vector(metadata_path)
    adjustment = np.zeros(6, dtype=float)

    print(f"[TEST] target TCP vector (left-right) = {target_vector.tolist()}")
    print(f"[TEST] {args.robot} TCP vector (left-right) = {home_vector.tolist()}")
    print(f"[TEST] automatic vector alignment; manual correction = {adjustment.tolist()}")
    result = replay.replay(
        trajectory_index=args.index,
        adjust_ee_dist=adjustment,
    )
    collision = result["detail"].get("collision") or {
        "colliding": False,
        "score": 100.0,
    }
    print(f"[TEST] collision_min_score = {result['collision_min_score']}")
    print(f"[TEST] continuous_min_score = {result['continuous_min_score']}")
    print(f"[TEST] out_ws_min_score     = {result['out_ws_min_score']}")
    print(f"[TEST] total_score          = {_format_score(result.get('total_min_score'))}")
    return {
        "episode": input_path.name,
        "input_path": str(input_path),
        "trajectory_dir": str(trajectory_dir),
        "total": result.get("total_min_score"),
        "continuous": result.get("continuous_min_score"),
        "workspace": result.get("out_ws_min_score"),
        "collision": result.get("collision_min_score"),
        # Keep raw contact information separate from the zero-score severity
        # used for filtering: shallow contact can be retained when the total
        # score still passes the quality threshold.
        "contact": bool(collision.get("colliding", False)),
        "severe_collision": (
            result.get("collision_min_score") is not None
            and float(result["collision_min_score"]) <= 0.0
        ),
        "colliding": (
            result.get("collision_min_score") is not None
            and float(result["collision_min_score"]) <= 0.0
        ),
        "error": None,
    }


def _print_batch_summary(rows: list[dict[str, Any]], good_score_threshold: float) -> None:
    print("\n[BATCH SUMMARY]")
    print(
        f"{'episode':<25} {'total':>12} {'continuous':>12} "
        f"{'workspace':>12} {'collision':>12} {'status':>10}"
    )
    for row in rows:
        if row.get("error") is not None:
            status = "ERROR"
        elif row.get("severe_collision"):
            status = "COLLISION"
        elif _is_good_row(row, good_score_threshold):
            status = "GOOD"
        else:
            status = "REJECT"
        print(
            f"{row['episode']:<25} "
            f"{_format_score(row.get('total')):>12} "
            f"{_format_score(row.get('continuous')):>12} "
            f"{_format_score(row.get('workspace')):>12} "
            f"{_format_score(row.get('collision')):>12} "
            f"{status:>10}"
        )
        if row.get("error") is not None:
            print(f"  error: {row['error']}")


def main() -> int:
    args = parse_args()
    from convert_episode_to_robot import resolve_trajectory_input

    input_path = args.trajectory_dir.expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    episode_inputs, batch_mode = discover_episode_inputs(input_path)

    first_trajectory_dir = resolve_trajectory_input(episode_inputs[0])
    os.environ["SCORE_FOLDER_PATH"] = str(first_trajectory_dir)
    os.environ["ROBOT_NAME"] = args.robot

    import replay

    replay.FILE_SEARCH_LEVEL = 2
    replay.START_VIEWER = int(args.viewer)
    replay.LOG_FLAG = False

    robot = replay.RbtKin()
    home = robot.solve_fk()
    home_vector = np.asarray(home[0][:3]) - np.asarray(home[1][:3])
    rows: list[dict[str, Any]] = []
    for episode_number, episode_input in enumerate(episode_inputs, start=1):
        if batch_mode:
            print(
                f"\n[BATCH] ({episode_number}/{len(episode_inputs)}) "
                f"scoring {episode_input.name}"
            )
        try:
            trajectory_dir = (
                first_trajectory_dir
                if episode_number == 1
                else resolve_trajectory_input(episode_input)
            )
            rows.append(
                _run_episode(
                    episode_input,
                    trajectory_dir,
                    args,
                    replay,
                    home_vector,
                )
            )
        except Exception as exc:
            if not batch_mode:
                raise
            print(f"[BATCH] {episode_input.name} failed: {exc!r}")
            rows.append(
                {
                    "episode": episode_input.name,
                    "total": None,
                    "continuous": None,
                    "workspace": None,
                    "collision": None,
                    "colliding": False,
                    "error": repr(exc),
                }
            )

    if batch_mode:
        _print_batch_summary(rows, args.good_score_threshold)
        _copy_good_results(rows, args.good_output_dir, args.good_score_threshold)
        return 1 if any(row.get("error") is not None for row in rows) else 0
    if _is_good_row(rows[0], args.good_score_threshold):
        _copy_good_results(rows, args.good_output_dir, args.good_score_threshold)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
