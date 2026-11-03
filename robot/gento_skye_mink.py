"""Gento Skye simulation adapter."""

from .dual_arm_mink import DualArmMink


class GentoSkyeMink(DualArmMink):
    """Bind the shared dual-arm Mink adapter to the Gento Skye model."""

    def __init__(self, *args, **kwargs):
        kwargs["robot_name"] = "gento_skye"
        super().__init__(*args, **kwargs)
