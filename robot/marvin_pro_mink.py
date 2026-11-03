"""Marvin Pro simulation adapter."""

from .dual_arm_mink import (
    DualArmMink,
    _ARM_MAX_VELOCITIES_RAD_S,
    _COLLISION_DETECTION_DISTANCE_M,
    _COLLISION_MIN_DISTANCE_M,
    _HAND_ORIENTATION_COST,
    _HAND_POSITION_COST,
    _IK_DT_SEC,
    _IK_SUBSTEPS,
)


class MarvinProMink(DualArmMink):
    """Bind the dual-arm Mink implementation to the Marvin Pro model."""

    def __init__(self, *args, **kwargs):
        kwargs["robot_name"] = "marvin_pro"
        super().__init__(*args, **kwargs)
