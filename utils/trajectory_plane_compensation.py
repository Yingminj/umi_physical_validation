"""Shared plane-alignment compensation for dual-hand pose trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PlaneCompensation:
    rotation: np.ndarray
    measured_angle_deg: float
    applied_angle_deg: float
    limited: bool
    original_vector: np.ndarray
    corrected_vector: np.ndarray


def _normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(-1)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector, got {vector!r}")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def compute_plane_compensation(
    initial_vector: np.ndarray,
    plane_normal: np.ndarray = np.array([1.0, 0.0, 0.0]),
    max_angle_deg: float = 90.0,
) -> PlaneCompensation:
    """Return the minimal rotation that moves a vector towards a plane.

    The corrected vector keeps the original length.  When ``max_angle_deg`` is
    smaller than the measured line-to-plane angle, only the requested portion
    of the rotation is applied.
    """
    vector = np.asarray(initial_vector, dtype=float).reshape(-1)
    direction = _normalized(vector, name="initial_vector")
    normal = _normalized(plane_normal, name="plane_normal")
    max_angle_deg = float(max_angle_deg)
    if not np.isfinite(max_angle_deg) or not 0.0 <= max_angle_deg <= 90.0:
        raise ValueError(
            f"max_angle_deg must be within 0..90 degrees, got {max_angle_deg!r}"
        )

    projected = direction - float(np.dot(direction, normal)) * normal
    projected_norm = float(np.linalg.norm(projected))
    measured_angle_rad = float(
        np.arcsin(np.clip(abs(float(np.dot(direction, normal))), 0.0, 1.0))
    )
    measured_angle_deg = float(np.degrees(measured_angle_rad))

    if projected_norm <= 1e-9:
        raise ValueError(
            "initial_vector is parallel to the plane normal; its in-plane "
            "direction is undefined"
        )
    if measured_angle_rad <= 1e-12 or max_angle_deg == 0.0:
        rotation = np.eye(3, dtype=float)
        applied_angle_deg = 0.0
    else:
        target_direction = projected / projected_norm
        axis = np.cross(direction, target_direction)
        axis /= np.linalg.norm(axis)
        applied_angle_deg = min(measured_angle_deg, max_angle_deg)
        rotation = Rotation.from_rotvec(
            axis * np.radians(applied_angle_deg)
        ).as_matrix()

    corrected_vector = rotation @ vector
    return PlaneCompensation(
        rotation=rotation,
        measured_angle_deg=measured_angle_deg,
        applied_angle_deg=applied_angle_deg,
        limited=bool(applied_angle_deg + 1e-9 < measured_angle_deg),
        original_vector=vector.copy(),
        corrected_vector=corrected_vector,
    )


def _continuous_quaternions_xyzw(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float).copy()
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def apply_plane_compensation_to_pose_trajectories(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    compensation: PlaneCompensation,
    *,
    trajectories_are_relative: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one common compensation to left/right ``[t,p,q_xyzw]`` poses."""
    left = np.asarray(left_pose, dtype=float).copy()
    right = np.asarray(right_pose, dtype=float).copy()
    for name, pose in (("left_pose", left), ("right_pose", right)):
        if pose.ndim != 2 or pose.shape[1] != 8 or pose.shape[0] == 0:
            raise ValueError(f"{name} must have non-empty shape (N, 8), got {pose.shape}")

    correction = compensation.rotation
    if trajectories_are_relative:
        left[:, 1:4] = (correction @ left[:, 1:4].T).T
        right[:, 1:4] = (correction @ right[:, 1:4].T).T
    else:
        midpoint = 0.5 * (left[0, 1:4] + right[0, 1:4])
        left[:, 1:4] = midpoint + (correction @ (left[:, 1:4] - midpoint).T).T
        right[:, 1:4] = midpoint + (correction @ (right[:, 1:4] - midpoint).T).T

    for pose in (left, right):
        matrices = Rotation.from_quat(pose[:, 4:8]).as_matrix()
        if trajectories_are_relative:
            matrices = (
                correction[None, :, :]
                @ matrices
                @ correction.T[None, :, :]
            )
        else:
            matrices = correction[None, :, :] @ matrices
        pose[:, 4:8] = _continuous_quaternions_xyzw(
            Rotation.from_matrix(matrices).as_quat()
        )

    return left, right
