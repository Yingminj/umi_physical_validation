"""Dual-arm platform C simulation adapter."""

from .dual_arm_mink import DualArmMink


class RobotCMink(DualArmMink):
    """Bind the shared dual-arm Mink adapter to the platform C model."""

    def __init__(self, *args, **kwargs):
        kwargs["robot_name"] = "robot_c"
        super().__init__(*args, **kwargs)
