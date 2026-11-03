"""Dual episode-camera playback used alongside the MuJoCo viewer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Iterable, Optional, Tuple

import numpy as np


VideoPair = Tuple[Path, Path]


def _pair_in(directory: Path) -> Optional[VideoPair]:
    view_dir = directory if directory.name == "view_files" else directory / "view_files"
    left = view_dir / "cam_left.mp4"
    right = view_dir / "cam_right.mp4"
    if left.is_file() and right.is_file():
        return left, right
    return None


def _metadata_candidates(paths: Iterable[Path]) -> Iterable[Path]:
    seen = set()
    for path in paths:
        current = path if path.is_dir() else path.parent
        for parent in (current, *current.parents):
            metadata = parent / "conversion_metadata.json"
            if metadata not in seen:
                seen.add(metadata)
                yield metadata


def resolve_episode_videos(
    score_folder: str | Path,
    selected_trajectory_path: str | Path | None = None,
) -> Optional[VideoPair]:
    """Find cam_left/right for either an original or converted episode path."""

    explicit = (os.environ.get("CAMERA_VIEW_DIR") or "").strip()
    direct_candidates = []
    if explicit:
        direct_candidates.append(Path(explicit).expanduser())
    direct_candidates.append(Path(score_folder).expanduser())
    if selected_trajectory_path is not None:
        direct_candidates.append(Path(selected_trajectory_path).expanduser())

    for candidate in direct_candidates:
        pair = _pair_in(candidate if candidate.is_dir() else candidate.parent)
        if pair is not None:
            return pair

    for metadata_path in _metadata_candidates(direct_candidates):
        if not metadata_path.is_file():
            continue
        try:
            source = json.loads(metadata_path.read_text(encoding="utf-8")).get("source")
            if source:
                pair = _pair_in(Path(source).expanduser())
                if pair is not None:
                    return pair
        except (OSError, ValueError, TypeError):
            continue
    return None


class DualCameraViewer:
    """Decode synchronized episode cameras for MuJoCo viewer.set_images()."""

    def __init__(
        self,
        left_video: Path,
        right_video: Path,
        playback_duration_sec: Optional[float] = None,
    ) -> None:
        self.left_video = Path(left_video)
        self.right_video = Path(right_video)
        # Fixed so stale shell exports cannot silently alter the UI layout.
        self.view_width = 640
        self.view_height = 360
        self.left_lead_in_sec = self._lead_in(self.left_video, playback_duration_sec)
        self.right_lead_in_sec = self._lead_in(self.right_video, playback_duration_sec)
        self._decoder: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

    @staticmethod
    def _video_duration(video: Path) -> Optional[float]:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=3.0,
            )
            duration = float(result.stdout.strip())
            return duration if duration > 0 else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    @classmethod
    def _lead_in(cls, video: Path, playback_duration_sec: Optional[float]) -> float:
        """Hold frame zero so video and trajectory finish at the same time."""
        if playback_duration_sec is None or playback_duration_sec <= 0:
            return 0.0
        video_duration = cls._video_duration(video)
        if video_duration is None:
            return 0.0
        # A very large mismatch likely means the wrong video was selected;
        # avoid hiding that mistake with a long frozen frame.
        difference = float(playback_duration_sec) - video_duration
        return difference if 0.0 < difference <= 5.0 else 0.0

    @staticmethod
    def available() -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def _decoder_command(self) -> list[str]:
        width = self.view_width
        height = self.view_height
        filter_graph = (
            f"[0:v]tpad=start_mode=clone:start_duration={self.left_lead_in_sec:.6f},"
            f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            "unsharp=5:5:0.8:3:3:0.35,"
            "drawtext=text=LEFT:x=16:y=16:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.55[left];"
            f"[1:v]tpad=start_mode=clone:start_duration={self.right_lead_in_sec:.6f},"
            f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
            "unsharp=5:5:0.8:3:3:0.35,"
            "drawtext=text=RIGHT:x=16:y=16:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.55[right];"
            "[left][right]vstack=inputs=2,fps=60[stacked]"
        )
        return [
            "ffmpeg",
            "-loglevel", "error",
            "-i", str(self.left_video),
            "-i", str(self.right_video),
            "-filter_complex", filter_graph,
            "-map", "[stacked]",
            "-an",
            # Feed uncompressed RGB directly to viewer.set_images(). This
            # avoids a second encode/decode generation and preserves labels.
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ]

    def start(self) -> bool:
        if self._decoder is not None:
            return True
        if not self.available():
            return False
        try:
            self._decoder = subprocess.Popen(
                self._decoder_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            assert self._decoder.stdout is not None
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._read_frames,
                name="dual-camera-frame-reader",
                daemon=True,
            )
            self._reader_thread.start()
        except OSError:
            self.stop()
            return False
        return True

    def _read_frames(self) -> None:
        decoder = self._decoder
        stream = decoder.stdout if decoder is not None else None
        if stream is None:
            return
        frame_size = self.view_width * self.view_height * 2 * 3
        frame_interval = 1.0 / 60.0
        first_frame_time = time.monotonic()
        frame_index = 0
        try:
            while not self._reader_stop.is_set():
                chunks = bytearray()
                while len(chunks) < frame_size and not self._reader_stop.is_set():
                    chunk = stream.read(frame_size - len(chunks))
                    if not chunk:
                        return
                    chunks.extend(chunk)
                if len(chunks) != frame_size:
                    return
                frame = np.frombuffer(bytes(chunks), dtype=np.uint8).reshape(
                    self.view_height * 2, self.view_width, 3
                )
                # Rawvideo has no timestamps for the Python consumer. Pace
                # publication here so tpad's cloned lead-in frames occupy
                # real time instead of being emitted in one burst.
                publish_time = first_frame_time + frame_index * frame_interval
                remaining = publish_time - time.monotonic()
                if remaining > 0 and self._reader_stop.wait(remaining):
                    return
                with self._frame_lock:
                    self._latest_frame = frame
                frame_index += 1
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the latest (left, right) RGB frames without blocking."""
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        return frame[: self.view_height], frame[self.view_height :]

    def stop(self) -> None:
        self._reader_stop.set()
        decoder, self._decoder = self._decoder, None
        if decoder is not None and decoder.poll() is None:
            decoder.terminate()
        if decoder is not None:
            try:
                decoder.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                decoder.kill()
                decoder.wait()
        reader, self._reader_thread = self._reader_thread, None
        if reader is not None:
            reader.join(timeout=1.0)
        with self._frame_lock:
            self._latest_frame = None
