from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import math
import time
import threading
import atexit
import numpy as np
import utils.self_math as smath

import mink

_HERE = Path(__file__).parent.parent
SUPPORTED_ROBOTS = ("marvin_pro", "gento_luna", "gento_skye")
_MODEL_XML = {
    name: _HERE / "model" / name / "scene.xml"
    for name in SUPPORTED_ROBOTS
}
_ARM_JOINT_NAMES = [
    "Arm_L1_Joint",
    "Arm_L2_Joint",
    "Arm_L3_Joint",
    "Arm_L4_Joint",
    "Arm_L5_Joint",
    "Arm_L6_Joint",
    "Arm_L7_Joint",
    "Arm_R1_Joint",
    "Arm_R2_Joint",
    "Arm_R3_Joint",
    "Arm_R4_Joint",
    "Arm_R5_Joint",
    "Arm_R6_Joint",
    "Arm_R7_Joint",
]
_ARM_MAX_VELOCITIES_RAD_S = [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0] * 2
_IK_DT_SEC = 0.02
_IK_SUBSTEPS = 4
_HAND_POSITION_COST = 100.0
# Keep the default task weights compatible with the original TCP replay. A
# collision-constrained run may opt into stronger secondary-task weighting.
_HAND_ORIENTATION_COST = 100.0
_BASE_COLLISION_GEOM_NAME = "base_link_collision"
_COLLISION_MIN_DISTANCE_M = 0.01
_COLLISION_DETECTION_DISTANCE_M = 0.05
_COLLISION_BACKTRACK_ITERATIONS = 14
_LEFT_ARM_COLLISION_BODY_NAMES = {
    *(f"Arm_L{index}_Link" for index in range(2, 8)),
    "left_gripper_base",
    "left_gripper_left_finger",
    "left_gripper_right_finger",
}
_RIGHT_ARM_COLLISION_BODY_NAMES = {
    *(f"Arm_R{index}_Link" for index in range(2, 8)),
    "right_gripper_base",
    "right_gripper_left_finger",
    "right_gripper_right_finger",
}
_BASE_COLLISION_ARM_BODY_NAMES = (
    _LEFT_ARM_COLLISION_BODY_NAMES | _RIGHT_ARM_COLLISION_BODY_NAMES
)
_GRIPPER_JOINT_NAMES = {
    "left": {
        "driver": "left_gripper_left_finger_joint",
        "fingers": (
            "left_gripper_left_finger_joint",
            "left_gripper_right_finger_joint",
        ),
    },
    "right": {
        "driver": "right_gripper_left_finger_joint",
        "fingers": (
            "right_gripper_left_finger_joint",
            "right_gripper_right_finger_joint",
        ),
    },
}
_MAPPING = [
    {"mj_name": name, "robot_index": idx, "scale": 1.0, "offset": 0.0, "sign": 1.0}
    for idx, name in enumerate(_ARM_JOINT_NAMES)
] + [
    {
        "mj_name": _GRIPPER_JOINT_NAMES["left"]["driver"],
        "robot_index": 14,
        "scale": 1.0,
        "offset": 0.0,
        "sign": 1.0,
    },
    {
        "mj_name": _GRIPPER_JOINT_NAMES["right"]["driver"],
        "robot_index": 15,
        "scale": 1.0,
        "offset": 0.0,
        "sign": 1.0,
    },
]

_HOME_ARM_QPOS = {
    "marvin_pro": np.array(
        [
            1.7, -1.1, -1.1, -2.0, -0.37, 0.13, 0.55,
            -1.7, -1.1, 1.1, -2.0, 0.37, 0.13, -0.55,
        ],
        dtype=float,
    ),
    "gento_luna": np.array(
        [
            1.2, -0.8, -0.8, -1.4, -0.3, 0.1, 0.4,
            -1.2, -0.8, 0.8, -1.4, 0.3, 0.1, -0.4,
        ],
        dtype=float,
    ),
    "gento_skye": np.array(
        [
            1.2, -0.8, -0.8, -1.4, -0.3, 0.1, 0.4,
            -1.2, -0.8, 0.8, -1.4, 0.3, 0.1, -0.4,
        ],
        dtype=float,
    ),
}
_EXTRA_HOME_QPOS = {
    "gento_skye": {"waist_joint_1_prismatic": 0.86},
}

class DualArmMink:
    """Common dual-arm Mink implementation for bundled simulation models."""

    def __init__(
        self,
        avoid_collisions: bool = False,
        robot_name: str = "marvin_pro",
    ):
        robot_name = str(robot_name).strip().lower()
        if robot_name not in _MODEL_XML:
            choices = ", ".join(SUPPORTED_ROBOTS)
            raise ValueError(f"unsupported robot {robot_name!r}; choose one of: {choices}")
        if avoid_collisions and robot_name != "marvin_pro":
            raise ValueError(
                "collision-constrained IK is currently calibrated only for marvin_pro"
            )
        self.robot_name = robot_name
        model = mujoco.MjModel.from_xml_path(_MODEL_XML[robot_name].as_posix())

        self.configuration = mink.Configuration(model)
        self.hands = ["left_tool_site", "right_tool_site"]
        self.posture_task = mink.PostureTask(model, cost=1e-1)
        self.tasks = [
            self.posture_task,
        ]
        self.hand_tasks = []
        for hand in self.hands:
            task = mink.FrameTask(
                frame_name=hand,
                frame_type="site",
                position_cost=_HAND_POSITION_COST,
                orientation_cost=_HAND_ORIENTATION_COST,
                lm_damping=1.0,
            )
            self.hand_tasks.append(task)
        self.tasks.extend(self.hand_tasks)
        self.avoid_collisions = bool(avoid_collisions)
        self.limits = [mink.ConfigurationLimit(model)]
        self.arm_velocity_limit = None
        self.base_collision_avoidance_limit = None
        self.inter_arm_collision_avoidance_limit = None
        if self.avoid_collisions:
            base_geom_id = int(model.geom(_BASE_COLLISION_GEOM_NAME).id)
            arm_collision_geom_ids = []
            left_arm_collision_geom_ids = []
            right_arm_collision_geom_ids = []
            for geom_id in range(model.ngeom):
                if int(model.geom_group[geom_id]) != 3:
                    continue
                body_id = int(model.geom_bodyid[geom_id])
                body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                if body_name in _BASE_COLLISION_ARM_BODY_NAMES:
                    arm_collision_geom_ids.append(int(geom_id))
                if body_name in _LEFT_ARM_COLLISION_BODY_NAMES:
                    left_arm_collision_geom_ids.append(int(geom_id))
                if body_name in _RIGHT_ARM_COLLISION_BODY_NAMES:
                    right_arm_collision_geom_ids.append(int(geom_id))
            if not arm_collision_geom_ids:
                raise RuntimeError("no arm collision geoms found for robot-base avoidance")

            self.arm_velocity_limit = mink.VelocityLimit(
                model,
                velocities=dict(
                    zip(_ARM_JOINT_NAMES, _ARM_MAX_VELOCITIES_RAD_S, strict=True)
                ),
            )
            self.base_collision_avoidance_limit = mink.CollisionAvoidanceLimit(
                model,
                geom_pairs=[([base_geom_id], arm_collision_geom_ids)],
                gain=0.85,
                minimum_distance_from_collisions=_COLLISION_MIN_DISTANCE_M,
                collision_detection_distance=_COLLISION_DETECTION_DISTANCE_M,
            )
            self.inter_arm_collision_avoidance_limit = mink.CollisionAvoidanceLimit(
                model,
                geom_pairs=[
                    (left_arm_collision_geom_ids, right_arm_collision_geom_ids)
                ],
                gain=0.85,
                minimum_distance_from_collisions=_COLLISION_MIN_DISTANCE_M,
                collision_detection_distance=_COLLISION_DETECTION_DISTANCE_M,
            )
            self.limits.extend(
                [
                    self.arm_velocity_limit,
                    self.base_collision_avoidance_limit,
                    self.inter_arm_collision_avoidance_limit,
                ]
            )
        self.hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in self.hands]

        self.model = self.configuration.model
        self.data = self.configuration.data
        # Separate render buffer to avoid viewer touching live data directly
        self._render_data = mujoco.MjData(self.model)
        # Concurrent-access protection: prevent viewer and control loop from modifying/copying mjData at the same time,
        # which can trigger mj_copyDataVisual errors. Use a reentrant lock to avoid deadlocks in nested locked calls.
        self._mj_lock = threading.RLock()
        # Viewer lifecycle state. These handles are kept explicitly so the
        # replay shutdown path can stop the native GLFW/MuJoCo context before
        # Python starts destroying the model and data objects.
        self._viewer_handle = None
        self._viewer_thread = None
        self._viewer_stop = None
        self._viewer_image_provider = None
        self._viewer_images_visible = False
        self._viewer_images_logged = False
        self.solver = "daqp"

        # Marvin Pro provides a named keyframe. The imported Gento models do
        # not, so initialize their shared 14 arm joints by name and leave the
        # leg/base/waist/head joints at their XML defaults.
        if self.robot_name == "marvin_pro":
            self.configuration.update_from_keyframe("home")
        else:
            for joint_name, value in zip(
                _ARM_JOINT_NAMES, _HOME_ARM_QPOS[self.robot_name], strict=True
            ):
                joint_id = self.model.joint(joint_name).id
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[qpos_adr] = float(value)
            for joint_name, value in _EXTRA_HOME_QPOS.get(
                self.robot_name, {}
            ).items():
                joint_id = self.model.joint(joint_name).id
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[qpos_adr] = float(value)
            mujoco.mj_forward(self.model, self.data)
        self.posture_task.set_target_from_configuration(self.configuration)
        # Initialize mocap bodies at their respective sites.
        for hand in self.hands:
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")

        # Record baseline mocap positions (hands + COM) and (quaternions, MuJoCo order [w,x,y,z]).
        self.hands_base = [self.data.mocap_pos[mid].copy() for mid in self.hands_mid]
        self.hands_base_quat = [self.data.mocap_quat[mid].copy() for mid in self.hands_mid]
        # The replay layer represents each gripper as 0..100 percent, where
        # 100 means open.  In the MuJoCo model q=0 is open and q=0.05 is
        # closed (the two finger bodies move toward one another as q grows).
        self.gripper_joint_max = 0.05
        self._sync_gripper_from_current_locked()

    def _gripper_pct_to_joint(self, value_pct: float) -> float:
        pct = float(np.clip(value_pct, 0.0, 100.0))
        return (100.0 - pct) / 100.0 * self.gripper_joint_max

    def _gripper_joint_to_pct(self, joint_value: float) -> float:
        if self.gripper_joint_max <= 1e-9:
            return 0.0
        pct = 100.0 - float(joint_value) / self.gripper_joint_max * 100.0
        return float(np.clip(pct, 0.0, 100.0))

    def _set_gripper_side_locked(self, side: str, value_pct: float) -> None:
        pct = float(np.clip(value_pct, 0.0, 100.0))
        joint_value = self._gripper_pct_to_joint(pct)
        for joint_name in _GRIPPER_JOINT_NAMES[side]["fingers"]:
            joint_id = self.model.joint(joint_name).id
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_adr] = joint_value

    def _get_gripper_pct_locked(self, side: str) -> float:
        driver_name = _GRIPPER_JOINT_NAMES[side]["driver"]
        joint_id = self.model.joint(driver_name).id
        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        joint_value = float(self.data.qpos[qpos_adr])
        return self._gripper_joint_to_pct(joint_value)

    def _sync_gripper_from_current_locked(self) -> None:
        self._set_gripper_side_locked("left", self._get_gripper_pct_locked("left"))
        self._set_gripper_side_locked("right", self._get_gripper_pct_locked("right"))

    def _minimum_collision_distance_locked(self) -> float:
        if not self.avoid_collisions:
            return float("inf")
        fromto = np.empty(6, dtype=float)
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom1,
                    geom2,
                    _COLLISION_DETECTION_DISTANCE_M,
                    fromto,
                )
            )
            for limit in (
                self.base_collision_avoidance_limit,
                self.inter_arm_collision_avoidance_limit,
            )
            for geom1, geom2 in limit.geom_id_pairs
        )

    def _enforce_collision_clearance_locked(
        self, previous_qpos: np.ndarray, candidate_qpos: np.ndarray
    ) -> None:
        """Backtrack only an IK candidate that crosses the 10 mm clearance."""
        candidate_distance = self._minimum_collision_distance_locked()
        if candidate_distance >= _COLLISION_MIN_DISTANCE_M:
            return

        self.data.qpos[:] = previous_qpos
        mujoco.mj_fwdPosition(self.model, self.data)
        previous_distance = self._minimum_collision_distance_locked()
        if previous_distance < _COLLISION_MIN_DISTANCE_M:
            # Allow an already unsafe externally supplied state to move only
            # when the new IK candidate increases its clearance.
            if candidate_distance > previous_distance:
                self.data.qpos[:] = candidate_qpos
                mujoco.mj_fwdPosition(self.model, self.data)
            return

        safe_fraction = 0.0
        unsafe_fraction = 1.0
        for _ in range(_COLLISION_BACKTRACK_ITERATIONS):
            fraction = 0.5 * (safe_fraction + unsafe_fraction)
            self.data.qpos[:] = previous_qpos + fraction * (
                candidate_qpos - previous_qpos
            )
            mujoco.mj_fwdPosition(self.model, self.data)
            if (
                self._minimum_collision_distance_locked()
                >= _COLLISION_MIN_DISTANCE_M
            ):
                safe_fraction = fraction
            else:
                unsafe_fraction = fraction

        self.data.qpos[:] = previous_qpos + safe_fraction * (
            candidate_qpos - previous_qpos
        )
        mujoco.mj_fwdPosition(self.model, self.data)

    def solve_ik(self, tgt_hand_pose7):
        """Solve IK for given hand mocap targets (position + optional orientation).

        Args:
            tgt_hand_pos: list/tuple of two 3D position arrays.
            tgt_hand_quat: optional list of two quaternions [w,x,y,z]; if provided will set mocap orientations.
        Returns:
            numpy.ndarray: Updated joint positions `qpos` after integration (shape: (nq,)).
        """
        with self._mj_lock:
            left_gripper_pct = self._get_gripper_pct_locked("left")
            right_gripper_pct = self._get_gripper_pct_locked("right")
            self.data.mocap_pos[self.hands_mid[0]] = np.array(tgt_hand_pose7[0][0:3])
            self.data.mocap_pos[self.hands_mid[1]] = np.array(tgt_hand_pose7[1][0:3])
            self.data.mocap_quat[self.hands_mid[0]] = np.array(tgt_hand_pose7[0][3:7])
            self.data.mocap_quat[self.hands_mid[1]] = np.array(tgt_hand_pose7[1][3:7])

            # Dynamically adjust hands and COM costs (based on whether COM is inside/outside support triangle)
            for i, hand_task in enumerate(self.hand_tasks):
                hand_task.set_target(mink.SE3.from_mocap_id(self.data, self.hands_mid[i]))
            substep_count = _IK_SUBSTEPS if self.avoid_collisions else 1
            substep_dt = (
                _IK_DT_SEC / _IK_SUBSTEPS
                if self.avoid_collisions
                else 0.005
            )
            for _ in range(substep_count):
                previous_qpos = self.data.qpos.copy()
                vel = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    substep_dt,
                    self.solver,
                    1e-1,
                    limits=self.limits,
                )
                self.configuration.integrate_inplace(vel, substep_dt)
                # Frame and posture tasks solve over the full configuration.
                # Gripper finger joints do not contribute to either tool-site
                # Jacobian, so the posture task otherwise pulls them toward
                # their home values on every IK substep. Preserve the replayed
                # gripper command and let IK change arm joints only.
                self._set_gripper_side_locked("left", left_gripper_pct)
                self._set_gripper_side_locked("right", right_gripper_pct)
                candidate_qpos = self.data.qpos.copy()
                if self.avoid_collisions:
                    self._enforce_collision_clearance_locked(
                        previous_qpos,
                        candidate_qpos,
                    )
            _camlight = getattr(mujoco, "mj_camlight", None)
            if _camlight is not None:
                _camlight(self.model, self.data)
            # Collect mapped qpos for return while still holding the lock
            arm_qpos = [
                float(self.data.qpos[self.model.joint(joint_name).id])
                for joint_name in _ARM_JOINT_NAMES
            ]
            qret = np.array(
                arm_qpos
                + [
                    self._get_gripper_pct_locked("left"),
                    self._get_gripper_pct_locked("right"),
                ],
                dtype=float,
            )
        return qret

    def forward_kinematics(self, frame_name: str, frame_type: str = "site", sync_mocap: bool = False):
        """Return current world pose of a site/body.

        Args:
            frame_name: name of the site or body in the model.
            frame_type: "site" or "body".

        Returns:
            (pos, quat): tuple of numpy arrays. pos shape (3,), quat shape (4,w,x,y,z).
        """
        # Ensure all MuJoCo access happens under the lock to avoid stack-in-use errors
        with self._mj_lock:
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()

            if frame_type == "site":
                sid = self.model.site(frame_name).id
                pos = self.data.site_xpos[sid].copy()
                if hasattr(self.data, "site_xquat"):
                    quat = self.data.site_xquat[sid].copy()
                elif hasattr(self.data, "site_xmat"):
                    mat9 = self.data.site_xmat[sid].copy()
                    quat = np.empty(4, dtype=float)
                    try:
                        mujoco.mju_mat2Quat(quat, mat9)
                    except Exception:
                        # Fallback: identity if conversion utility is unavailable
                        quat = None
                else:
                    quat = None
                # print("*********************",pos,quat)
                return pos, quat
            elif frame_type == "body":
                bid = self.model.body(frame_name).id
                pos = self.data.xpos[bid].copy()
                quat = self.data.xquat[bid].copy()
                return pos, quat
            else:
                raise ValueError(f"Unsupported frame_type: {frame_type}. Use 'site' or 'body'.")

    def solve_fk(self):
        """Return current 7D poses of both hands as [x,y,z,qw,qx,qy,qz].

        Uses the actual hand sites (not the mocap target bodies) so this reflects
        the robot's current end-effector pose. Safe for seed/home capture.
        """
        poses7 = []
        for hand in self.hands:
            # The site frame (not *_target) is the midpoint of the finger tips.
            pos, quat = self.forward_kinematics(hand, "site", sync_mocap=False)# set_real_qpos has already synchronized this
            if quat is None:
                # Fallback: identity quaternion if missing
                quat = np.array([1.0,0.0,0.0,0.0], dtype=float)
            # MuJoCo site_xquat ordering is [w,x,y,z]; maintain same ordering used elsewhere
            pose7 = np.concatenate([pos.astype(float), quat.astype(float)])
            poses7.append(pose7)
        return poses7

    def set_real_qpos(self, q_real, sync_mocap: bool = False):
        """Set MuJoCo qpos from real robot encoders using a mapping.

        Args:
            q_real: sequence or numpy array of real robot joint positions (encoder readings).
            mapping: list of dicts, each with keys:
                - 'mj_name': MuJoCo joint name
                - 'robot_index': index in q_real
                - 'scale': unit scale (e.g., deg->rad). Default 1.0
                - 'offset': additive offset. Default 0.0
                - 'sign': +1 or -1 for direction. Default +1

        Writes mapped values into self.data.qpos in MuJoCo joint order, then runs forward update.
        """
        # Build name->id map once; perform full update in one locked block
        with self._mj_lock:
            q_mj = self.data.qpos.copy()
            for m in _MAPPING[:14]:
                mj_name = m['mj_name']
                ridx = m['robot_index']
                scale = m.get('scale', 1.0)
                offset = m.get('offset', 0.0)
                sign = m.get('sign', 1.0)
                jid = self.model.joint(mj_name).id
                raw = q_real[ridx]
                val = sign * (raw * scale + offset)
                q_mj[jid] = val
            self.data.qpos[:] = q_mj
            self._set_gripper_side_locked("left", float(q_real[14]) if len(q_real) > 14 else 0.0)
            self._set_gripper_side_locked("right", float(q_real[15]) if len(q_real) > 15 else 0.0)
            has_range = getattr(self.model, 'jnt_range', None) is not None
            if has_range:
                for j in range(self.model.njnt):
                    lo, hi = self.model.jnt_range[j]
                    if lo < hi:
                        self.data.qpos[j] = float(np.clip(self.data.qpos[j], lo, hi))
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()

    def update_configureation_to_home(self):
        """Update configuration from a named keyframe."""
        # with self._mj_lock:
        #     # 1. Apply keyframe
        #     self.configuration.update_from_keyframe("homepos")
        #     self._sync_gripper_from_current_locked()
        #     # 2. Forward update
        #     _fwd = getattr(mujoco, "mj_forward", None)
        #     if _fwd is not None:
        #         _fwd(self.model, self.data)
        #     else:
        #         _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
        #         if _fwd_pos is not None:
        #             _fwd_pos(self.model, self.data)
        #     # 3. Posture & torso orientation
        #     self.posture_task.set_target_from_configuration(self.configuration)
        pass

    def set_lock_flag(self, enabled: bool):
        pass

    def update_viewer(self, viewer=None):
        """Update MuJoCo derived data and optionally sync the viewer."""
        with self._mj_lock:
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            _sensor_pos = getattr(mujoco, "mj_sensorPos", None)
            if _sensor_pos is not None:
                _sensor_pos(self.model, self.data)
            if viewer is not None:
                viewer.sync()

    def launch_viewer(self, hand_amp=0.5, hand_freq=0.5):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            _free_cam = getattr(mujoco, "mjv_defaultFreeCamera", None)
            if _free_cam is not None:
                _free_cam(self.model, viewer.cam)
            t = 0.0
            rate = RateLimiter(frequency=200.0, warn=False)
            tgt_hand_pose7 =[np.hstack([self.hands_base[0].copy(),self.hands_base_quat[0].copy()]),
                             np.hstack([self.hands_base[1].copy(),self.hands_base_quat[1].copy()])]
            while viewer.is_running():
                # Drive simple periodic mocap motions so IK produces movement.
                t += rate.dt
                # meters, Hz (COM trajectory optional)
                # com_amp, com_freq = 0.2, 0.25

                w_hand = 2.0 * math.pi * hand_freq
                for i in range(2):
                    phase = math.pi if (i % 2 == 1) else 0.0  # alternate left/right
                    offset_x = hand_amp * math.sin(w_hand * t + phase)
                    new_pos = self.hands_base[i].copy()
                    new_pos[0] = new_pos[0] + offset_x
                    tgt_hand_pose7[i][0:3] = new_pos

                # Solve IK for current targets and update viewer
                self.solve_ik(tgt_hand_pose7)
                self.update_viewer(viewer)
                rate.sleep()

    # Test function for this class
    def test_traj(self, duration=5.0, viewer=False, hand_amp=0.5, hand_freq=0.5):
        """Run a simple sinusoidal hand trajectory.

        If viewer=False, only perform kinematic IK updates without rendering.
        """
        if viewer:
            return self.launch_viewer(hand_amp=hand_amp, hand_freq=hand_freq)

        # No viewer: run purely kinematic loop
        t = 0.0
        rate = RateLimiter(frequency=200.0, warn=False)
        tgt_hand_pose7 =[np.hstack([self.hands_base[0].copy(),self.hands_base_quat[0].copy()]),
                         np.hstack([self.hands_base[1].copy(),self.hands_base_quat[1].copy()])]
        steps = int(duration * 200)
        for _ in range(steps):
            t += rate.dt
            w_hand = 2.0 * math.pi * hand_freq
            for i in range(2):
                phase = math.pi if (i % 2 == 1) else 0.0
                offset_x = hand_amp * math.sin(w_hand * t + phase)
                new_pos = self.hands_base[i].copy()
                new_pos[0] = new_pos[0] + offset_x
                tgt_hand_pose7[i][0:3] = new_pos
                # Example: keep orientation constant; could add slow yaw rotation here if desired
            self.solve_ik(tgt_hand_pose7)
            self.update_viewer(viewer=None)
            rate.sleep()
        return True

    # Teleoperation and simulation interaction function
    def apply_mapped_qpos(self, q_mapped, sync_mocap: bool = False):
        """Apply a mapped qpos array (ordered like `_MAPPING`) into the MuJoCo sim.

        Args:
            q_mapped: iterable of floats with length == len(_MAPPING).
            sync_mocap: if True, move mocap targets (hands, COM) to match current frames
                        after updating qpos; useful for passive visualization without IK.
        """
        q = np.asarray(q_mapped, dtype=float).reshape(-1)
        if q.shape[0] < 1:
            return
        # Structure:
        # First 14 values: 7 (left arm) + 7 (right arm)
        # Optional 15th/16th values: two gripper scalars (left, right) in mm.
        # If provided, they are copied into 4 gripper joints in the model with mm->m conversion
        # (and the same /2 scaling convention as set_real_qpos).
        with self._mj_lock:
            # Write torso and arm values (no unit conversion involved)
            base_count = min(14, q.shape[0])
            for i in range(base_count):
                m = _MAPPING[i]
                jid = self.model.joint(m['mj_name']).id
                self.data.qpos[jid] = float(q[i])
            # Handle gripper (if 2 parameters are provided, [14]=left, [15]=right percentage)
            if q.shape[0] >= 16:
                left_grip_pct = float(q[14])
                right_grip_pct = float(q[15])
                self._set_gripper_side_locked("left", left_grip_pct)
                self._set_gripper_side_locked("right", right_grip_pct)
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            if sync_mocap:
                self._sync_mocap_targets_locked()

    def get_sim_feedback(self):
        """Return a minimal feedback dict from the current MuJoCo state.

        The dict contains qpos/qvel split into `torso`, `left_arm`, `right_arm`,
        plus physical gripper opening values. All joint lists are numpy arrays.
        """
        # Build list of q values in the same order as _MAPPING
        with self._mj_lock:
            left_arm = np.array(
                [float(self.data.qpos[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[:7]],
                dtype=float,
            )
            right_arm = np.array(
                [float(self.data.qpos[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[7:14]],
                dtype=float,
            )
            left_arm_vel = np.array(
                [float(self.data.qvel[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[:7]],
                dtype=float,
            )
            right_arm_vel = np.array(
                [float(self.data.qvel[self.model.joint(name).id]) for name in _ARM_JOINT_NAMES[7:14]],
                dtype=float,
            )
            left_grip_fb = self._get_gripper_pct_locked("left")
            right_grip_fb = self._get_gripper_pct_locked("right")
        feedback = {
            'qpos': {
                'left_arm': left_arm,
                'right_arm': right_arm,
                'left_gripper': [left_grip_fb],
                'right_gripper': [right_grip_fb],
            },
            'qvel': {
                'left_arm': left_arm_vel,
                'right_arm': right_arm_vel,
            }
        }
        return feedback

    def _sync_mocap_targets_locked(self):
        """Assumes caller holds `self._mj_lock`.

        Move mocap bodies (hands targets and COM target) to the current frames so that
        visual markers match the robot state. This is helpful when not running IK.
        """
        for hand in self.hands:
            # Move the corresponding mocap body to the site's current pose
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")

    def set_viewer_image_provider(self, provider) -> None:
        """Set a non-blocking callback returning ``(left_rgb, right_rgb)``."""
        self._viewer_image_provider = provider

    def _update_viewer_images(self, viewer) -> None:
        provider = self._viewer_image_provider
        frames = provider() if callable(provider) else None
        if frames is None:
            if self._viewer_images_visible:
                viewer.clear_images()
                self._viewer_images_visible = False
            return
        left, right = frames
        height, width = left.shape[:2]
        viewport = viewer.viewport
        if (
            viewport is None
            or viewport.width < width
            or viewport.height < height * 2
            or right.shape[:2] != (height, width)
        ):
            if self._viewer_images_visible:
                viewer.clear_images()
                self._viewer_images_visible = False
            return
        x = int(viewport.width - width)
        left_rect = mujoco.MjrRect(x, int(viewport.height - height), width, height)
        right_rect = mujoco.MjrRect(x, int(viewport.height - 2 * height), width, height)
        viewer.set_images([(left_rect, left), (right_rect, right)])
        if not self._viewer_images_logged:
            print(
                f"[VIEWER] Embedded camera images: {width}x{height} "
                f"at right side of {viewport.width}x{viewport.height} viewport"
            )
            self._viewer_images_logged = True
        self._viewer_images_visible = True

    def start_viewer_thread(self, hand_amp=0.0, hand_freq=0.0, drive_mocap: bool = False, viewer_hz: float = 60.0, compute_hz: float = 200.0):
        """Start a background thread that runs a passive viewer.

        Call `stop_viewer_thread()` to stop it. This is a convenience wrapper
        useful when integrating simulation with an external control loop.
        """
        if self._viewer_thread is not None:
            return
        import threading
        self._viewer_stop = threading.Event()

        def _remove_mujoco_glfw_atexit_handler():
            """Prevent GLFW TLS teardown from running on the wrong thread.

            MuJoCo's Linux ``launch_passive`` initializes GLFW in its internal
            GUI thread but registers ``glfw.terminate`` with Python's process-
            wide atexit stack. atexit callbacks run on the main thread, where
            this pyGLFW build has no allocated TLS slot and aborts in
            ``_glfwPlatformSetTls``. The viewer is closed explicitly below,
            so the mismatched atexit callback is unsafe and unnecessary.
            """
            try:
                import glfw
                atexit.unregister(glfw.terminate)
            except Exception:
                pass

        def _run():
            internal_viewer_thread = None
            try:
                # Use a separate render buffer so viewer never reads live self.data
                threads_before_launch = set(threading.enumerate())
                with mujoco.viewer.launch_passive(model=self.model, data=self._render_data, show_left_ui=False, show_right_ui=False) as viewer:
                    # launch_passive creates another Python thread running
                    # viewer._launch_internal. Keep that thread explicitly: a
                    # Handle.close() only requests exit and does not join it.
                    new_threads = [
                        thread
                        for thread in threading.enumerate()
                        if thread not in threads_before_launch
                        and thread is not threading.current_thread()
                        and "_launch_internal" in thread.name
                    ]
                    if new_threads:
                        internal_viewer_thread = new_threads[0]
                    _remove_mujoco_glfw_atexit_handler()
                    # Keep a temporary handle so shutdown can wake GLFW before
                    # joining this thread.
                    self._viewer_handle = viewer
                    self.set_viewer_status("COLLISION MONITOR\nwaiting for replay")
                    _free_cam = getattr(mujoco, "mjv_defaultFreeCamera", None)
                    if _free_cam is not None:
                        _free_cam(self.model, viewer.cam)
                    rate = RateLimiter(frequency=max(1.0, float(compute_hz)), warn=False)
                    # viewer sync limiter (lower rate than compute)
                    viewer_interval = 1.0 / max(1.0, float(viewer_hz))
                    last_view_sync_t = time.time()
                    while viewer.is_running() and not self._viewer_stop.is_set():
                        # simple periodic mocap motion if requested
                        t = time.time()
                        w_hand = 2.0 * math.pi * hand_freq
                        # Update live data and copy to render buffer under lock
                        with self._mj_lock:
                            if drive_mocap and hand_amp != 0.0:
                                for i in range(2):
                                    phase = math.pi if (i % 2 == 1) else 0.0
                                    offset_x = hand_amp * math.sin(w_hand * t + phase)
                                    new_pos = self.hands_base[i].copy()
                                    new_pos[0] = new_pos[0] + offset_x
                                    self.data.mocap_pos[self.hands_mid[i]] = new_pos
                            # Compute derived quantities on live data (fast loop)
                            _fwd = getattr(mujoco, "mj_forward", None)
                            if _fwd is not None:
                                _fwd(self.model, self.data)
                            else:
                                _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
                                if _fwd_pos is not None:
                                    _fwd_pos(self.model, self.data)
                                _sensor_pos = getattr(mujoco, "mj_sensorPos", None)
                                if _sensor_pos is not None:
                                    _sensor_pos(self.model, self.data)
                            # Copy live data to render buffer safely
                            _copy = getattr(mujoco, "mj_copyData", None)
                            if _copy is not None:
                                _copy(self._render_data, self.model, self.data)
                            else:
                                # Fallback: minimal fields
                                self._render_data.qpos[:] = self.data.qpos
                                self._render_data.qvel[:] = self.data.qvel
                                self._render_data.mocap_pos[:] = self.data.mocap_pos
                                self._render_data.mocap_quat[:] = self.data.mocap_quat
                                _fwd2 = getattr(mujoco, "mj_forward", None)
                                if _fwd2 is not None:
                                    _fwd2(self.model, self._render_data)
                        # Sync the viewer at throttled rate, reading only render buffer
                        now_t = time.time()
                        if (now_t - last_view_sync_t) >= viewer_interval:
                            try:
                                self._update_viewer_images(viewer)
                            except Exception:
                                # Video overlay failures must never stop the
                                # MuJoCo render/control loop.
                                if self._viewer_images_visible:
                                    try:
                                        viewer.clear_images()
                                    except Exception:
                                        pass
                                    self._viewer_images_visible = False
                            viewer.sync()
                            last_view_sync_t = now_t
                        rate.sleep()
                # The context manager has requested native viewer shutdown.
                # Wait for MuJoCo's actual GUI thread before allowing Python
                # module finalization to reclaim its callback objects.
                if internal_viewer_thread is not None:
                    internal_viewer_thread.join()
            except Exception:
                pass
            finally:
                # Also remove handlers from failed/retried viewer startups.
                _remove_mujoco_glfw_atexit_handler()
                self._viewer_handle = None
                self._viewer_images_visible = False
                self._viewer_images_logged = False

        self._viewer_thread = threading.Thread(target=_run, daemon=True)
        self._viewer_thread.start()

    def stop_viewer_thread(self):
        if self._viewer_stop is not None:
            self._viewer_stop.set()

        thread = self._viewer_thread
        if thread is not None:
            # Never return while native MuJoCo/GLFW code may still reference
            # self.model or self._render_data. A daemon thread surviving this
            # point can cause a segmentation fault during interpreter exit.
            thread.join(timeout=5.0)
            if thread.is_alive():
                # Best-effort close only after the normal event-driven exit
                # failed, avoiding concurrent close/sync in the usual path.
                viewer = self._viewer_handle
                close = getattr(viewer, "close", None) if viewer is not None else None
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                thread.join()
            self._viewer_thread = None
            self._viewer_stop = None

    def set_viewer_status(self, text: str) -> None:
        """Display a short status message in the MuJoCo viewer overlay."""
        viewer = self._viewer_handle
        if viewer is None:
            return
        try:
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_200,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                str(text),
                None,
            ))
        except Exception:
            # The window may be closing while the replay loop publishes its
            # final status; overlay failure must not affect scoring/replay.
            pass

    def check_collision_pairs(self):
        """Score non-floor contacts reported by the Marvin MuJoCo model.

        Adjacent-link contacts are excluded in ``marvin_pro.xml``.  Therefore
        remaining robot-to-robot contacts represent relevant self-collisions.
        """
        with self._mj_lock:
            _fwd = getattr(mujoco, "mj_forward", None)
            if _fwd is not None:
                _fwd(self.model, self.data)
            try:
                floor_id = int(self.model.geom("floor").id)
            except Exception:
                floor_id = -1
            contacts = []
            for contact_index in range(int(self.data.ncon)):
                contact = self.data.contact[contact_index]
                if floor_id in (int(contact.geom1), int(contact.geom2)):
                    continue
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                body1 = int(self.model.geom_bodyid[geom1])
                body2 = int(self.model.geom_bodyid[geom2])
                contacts.append(
                    {
                        "geom1": self.model.geom(geom1).name or f"geom#{geom1}",
                        "geom2": self.model.geom(geom2).name or f"geom#{geom2}",
                        "body1": self.model.body(body1).name,
                        "body2": self.model.body(body2).name,
                        "distance": float(contact.dist),
                    }
                )
            if not contacts:
                return {
                    "colliding": False,
                    "min_dist": [],
                    "score": 100.0,
                    "contacts": [],
                }
            contacts.sort(key=lambda item: item["distance"])
            min_dist = float(contacts[0]["distance"])
            # Keep the UMI-Vista distance-based scoring rule:
            # <=5 mm is a collision (0 points), 5--45 mm rises linearly,
            # and >=45 mm receives full credit.
            if min_dist >= 0.045:
                score = 100.0
            elif min_dist <= 0.005:
                score = 0.0
            else:
                score = 100.0 * ((min_dist - 0.005) / 0.04)
            return {
                "colliding": bool(min_dist <= 0.005),
                "min_dist": [min_dist],
                "score": float(score),
                "contacts": contacts,
            }

    def get_hand_target_pose_error(self):
        """Compute 6D error between hand target mocaps and actual hand sites.

        Returns a list with two dicts (left, right):
        - pos_error: Euclidean distance between mocap pos and site pos (meters)
        - ori_error_rad: rotation angle between mocap quat and site quat (radians)
        - pos_delta: 3-vector difference (target - actual)
        - ori_axis_angle: 3-vector axis * angle (angle-axis) representing orientation error
        """
        errs = []
        with self._mj_lock:
            # Make sure derived quantities are current
            _fwd_pos = getattr(mujoco, "mj_fwdPosition", None)
            if _fwd_pos is not None:
                _fwd_pos(self.model, self.data)
            for i, hand in enumerate(self.hands):
                # Target from mocap
                tpos = self.data.mocap_pos[self.hands_mid[i]].copy()
                tquat = self.data.mocap_quat[self.hands_mid[i]].copy()
                # Actual site pose
                spos, squat = self.forward_kinematics(hand, "site", sync_mocap=False)
                if squat is None:
                    squat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                # Position error
                dp = tpos - spos
                pos_err = float(np.linalg.norm(dp))
                # Orientation error: q_err = q_target * inv(q_actual)
                qw_t, qx_t, qy_t, qz_t = tquat
                qw_a, qx_a, qy_a, qz_a = squat
                # inverse actual
                q_inv_a = np.array([qw_a, -qx_a, -qy_a, -qz_a], dtype=float)
                q_err = smath.quat_multiply(np.array([qw_t, qx_t, qy_t, qz_t], dtype=float), q_inv_a)
                # angle from quaternion using shortest rotation (q and -q are equivalent)
                w = float(q_err[0])
                w = max(-1.0, min(1.0, w))
                ang = 2.0 * math.acos(abs(w))
                # normalize axis from vector part; handle near-zero rotation robustly
                v = np.array([q_err[1], q_err[2], q_err[3]], dtype=float)
                v_norm = float(np.linalg.norm(v))
                if v_norm > 1e-12:
                    axis = v / v_norm
                else:
                    axis = np.array([0.0, 0.0, 0.0], dtype=float)
                ori_axis_angle = axis * ang
                errs.append({
                    "pos_error": pos_err,
                    "ori_error_rad": float(ang),
                    "pos_delta": dp,
                    "ori_axis_angle": ori_axis_angle,
                })
        return errs

    def check_out_workspace(self):
        """Determine whether workspace limits are exceeded based on 6D error between target and actual hand pose.

        Rule: if either hand has `pos_error` or `ori_error_rad` > 0.03, mark as out-of-workspace.
        Returns: {"any_out": bool, "left": {...}, "right": {...}}
        left/right include: pos_error, ori_error_rad, pos_delta, ori_axis_angle, out.
        """
        errs = self.get_hand_target_pose_error()
        result_per_hand = []
        any_out = False
        overall_score = 100.0
        for e in errs:
            out = (float(e.get("pos_error", 0.0)) > 0.01) or (float(e.get("ori_error_rad", 0.0)) > 2.0/180.0*math.pi) # and not self._is_com_inside_support() ?
            # Scoring rules:
            # - Position error p: 0 -> 100, 0.045 -> 60; above threshold, use exponential decay asymptotically toward 0
            # - Angular error a_deg: 0 -> 100, 9 deg -> 60; above threshold, use exponential decay asymptotically toward 0
            p = float(e.get("pos_error", 0.0))
            a_rad = float(e.get("ori_error_rad", 0.0))
            a_deg = a_rad * 180.0 / math.pi
            if p<=0.005:
                score_pos = 100.0
            elif p <= 0.045:
                score_pos = 100.0 - 40.0 * ((p-0.005) / 0.04)
            else:
                # Progressive decay: continuous at p=0.045 with score 60, then exponentially decays toward 0
                delta_p = p - 0.045
                decay_scale_p = 0.1  # Decay scale of about 10 cm
                score_pos = 60.0 * math.exp(-max(0.0, delta_p) / max(1e-6, decay_scale_p))
            if a_deg <= 1.0:
                score_ori = 100.0
            elif a_deg <= 9.0:
                score_ori = 100.0 - 40.0 * (a_deg / 8.0)
            else:
                # Progressive decay: continuous at a=9 degrees with score 60, then exponentially decays toward 0
                delta_a = a_deg - 9.0
                decay_scale_a = 20.0  # Decay scale of about 20 degrees
                score_ori = 60.0 * math.exp(-max(0.0, delta_a) / max(1e-6, decay_scale_a))
            score = min(score_pos, score_ori)
            res = {
                "pos_error": float(e.get("pos_error", 0.0)),
                "ori_error_rad": float(e.get("ori_error_rad", 0.0)),
                "pos_delta": np.asarray(e.get("pos_delta", np.zeros(3, dtype=float)), dtype=float),
                "ori_axis_angle": np.asarray(e.get("ori_axis_angle", np.zeros(3, dtype=float)), dtype=float),
                "out": bool(out),
                "score": float(max(0.0, score)),
            }
            result_per_hand.append(res)
            any_out = any_out or out
            overall_score = min(overall_score, res["score"]) if result_per_hand else res["score"]
        left = result_per_hand[0] if len(result_per_hand) > 0 else None
        right = result_per_hand[1] if len(result_per_hand) > 1 else None
        return {"any_out": bool(any_out), "left": left, "right": right, "score": float(overall_score if result_per_hand else 100.0)}
