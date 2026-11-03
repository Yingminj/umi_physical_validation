"""Dual-arm platform B simulation adapter."""

from .dual_arm_mink import DualArmMink


class RobotBMink(DualArmMink):
    """Bind the shared dual-arm Mink adapter to the platform B model."""

    def __init__(self, *args, **kwargs):
        kwargs["robot_name"] = "robot_b"
        super().__init__(*args, **kwargs)
