#!/usr/bin/env python3
"""Regression tests for Mink robot collision avoidance."""

import unittest

import mujoco
import numpy as np

from robot.robot_a_mink import (
    RobotAMink,
    _ARM_MAX_VELOCITIES_RAD_S,
    _COLLISION_DETECTION_DISTANCE_M,
    _COLLISION_MIN_DISTANCE_M,
    _HAND_ORIENTATION_COST,
    _HAND_POSITION_COST,
    _IK_DT_SEC,
    _IK_SUBSTEPS,
)


def minimum_base_distance(robot: RobotAMink) -> float:
    return min(
        float(
            mujoco.mj_geomDistance(
                robot.model,
                robot.data,
                geom1,
                geom2,
                1.0,
                np.empty(6),
            )
        )
        for geom1, geom2 in robot.base_collision_avoidance_limit.geom_id_pairs
    )


def minimum_inter_arm_distance(robot: RobotAMink) -> float:
    return min(
        float(
            mujoco.mj_geomDistance(
                robot.model,
                robot.data,
                geom1,
                geom2,
                1.0,
                np.empty(6),
            )
        )
        for geom1, geom2 in robot.inter_arm_collision_avoidance_limit.geom_id_pairs
    )


class CollisionAvoidanceTest(unittest.TestCase):
    def test_limits_match_replay_rate_and_robot_speed_configuration(self):
        robot = RobotAMink(avoid_collisions=True)
        self.assertAlmostEqual(_IK_DT_SEC, 0.02)
        self.assertEqual(_IK_SUBSTEPS, 4)
        self.assertAlmostEqual(_HAND_POSITION_COST, 100.0)
        self.assertAlmostEqual(_HAND_ORIENTATION_COST, 100.0)
        np.testing.assert_allclose(
            robot.arm_velocity_limit.limit,
            _ARM_MAX_VELOCITIES_RAD_S,
        )
        self.assertAlmostEqual(
            robot.base_collision_avoidance_limit.minimum_distance_from_collisions,
            _COLLISION_MIN_DISTANCE_M,
        )
        self.assertAlmostEqual(
            robot.base_collision_avoidance_limit.collision_detection_distance,
            _COLLISION_DETECTION_DISTANCE_M,
        )
        self.assertAlmostEqual(
            robot.inter_arm_collision_avoidance_limit.minimum_distance_from_collisions,
            _COLLISION_MIN_DISTANCE_M,
        )

    def test_feasible_motion_keeps_full_workspace_and_collision_scores(self):
        robot = RobotAMink(avoid_collisions=True)
        target = robot.solve_fk()
        for hand_target in target:
            hand_target[:3] += [0.02, 0.0, 0.0]

        for _ in range(300):
            robot.solve_ik(target)

        self.assertEqual(robot.check_collision_pairs()["score"], 100.0)
        self.assertEqual(robot.check_out_workspace()["score"], 100.0)

    def test_ik_substeps_do_not_modify_gripper_commands(self):
        robot = RobotAMink(avoid_collisions=True)
        target = robot.solve_fk()
        robot._set_gripper_side_locked("left", 25.0)
        robot._set_gripper_side_locked("right", 75.0)

        robot.solve_ik(target)

        self.assertAlmostEqual(robot._get_gripper_pct_locked("left"), 25.0)
        self.assertAlmostEqual(robot._get_gripper_pct_locked("right"), 75.0)

    def test_targets_inside_base_do_not_create_penetration(self):
        robot = RobotAMink(avoid_collisions=True)
        target = robot.solve_fk()
        target[0][:3] = [0.08, 0.13, 0.85]
        target[1][:3] = [0.08, -0.13, 0.85]

        minimum_distance = float("inf")
        for _ in range(300):
            robot.solve_ik(target)
            minimum_distance = min(minimum_distance, minimum_base_distance(robot))

        self.assertGreaterEqual(
            minimum_distance,
            _COLLISION_MIN_DISTANCE_M - 1e-6,
        )

    def test_crossed_hand_targets_keep_both_arms_ten_mm_apart(self):
        robot = RobotAMink(avoid_collisions=True)
        target = robot.solve_fk()
        left_position = target[0][:3].copy()
        target[0][:3] = target[1][:3]
        target[1][:3] = left_position

        minimum_distance = float("inf")
        for _ in range(300):
            robot.solve_ik(target)
            minimum_distance = min(
                minimum_distance,
                minimum_inter_arm_distance(robot),
            )

        self.assertGreaterEqual(
            minimum_distance,
            _COLLISION_MIN_DISTANCE_M - 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
