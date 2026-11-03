#!/usr/bin/env python3
"""Unit tests for dual-hand plane compensation."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.trajectory_plane_compensation import (
    apply_plane_compensation_to_pose_trajectories,
    compute_plane_compensation,
)


class PlaneCompensationTest(unittest.TestCase):
    def test_full_compensation_preserves_distance_and_removes_normal_component(self):
        vector = np.array([-0.2343683513, 0.3069857692, 0.0293847347])
        result = compute_plane_compensation(vector)

        self.assertAlmostEqual(result.corrected_vector[0], 0.0, places=10)
        self.assertAlmostEqual(
            np.linalg.norm(result.corrected_vector), np.linalg.norm(vector), places=10
        )
        self.assertAlmostEqual(result.measured_angle_deg, 37.23, places=1)
        self.assertFalse(result.limited)

    def test_angle_limit_is_respected(self):
        result = compute_plane_compensation(
            np.array([-0.2343683513, 0.3069857692, 0.0293847347]),
            max_angle_deg=15.0,
        )
        self.assertAlmostEqual(result.applied_angle_deg, 15.0)
        self.assertTrue(result.limited)

    def test_relative_trajectories_receive_one_common_rotation(self):
        pose = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.02, 0.1, 0.2, 0.3, *Rotation.from_euler("z", 20, degrees=True).as_quat()],
            ]
        )
        compensation = compute_plane_compensation(np.array([-0.2, 0.3, 0.05]))
        left, right = apply_plane_compensation_to_pose_trajectories(
            pose, pose, compensation, trajectories_are_relative=True
        )

        np.testing.assert_allclose(left, right, atol=1e-12)
        np.testing.assert_allclose(left[0, 1:4], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(left[0, 4:8], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(left[1, 1:4]), np.linalg.norm(pose[1, 1:4]))


if __name__ == "__main__":
    unittest.main()
