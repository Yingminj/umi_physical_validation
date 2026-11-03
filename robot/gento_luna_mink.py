"""Gento Luna simulation adapter."""

from .dual_arm_mink import DualArmMink


class GentoLunaMink(DualArmMink):
    """Bind the shared dual-arm Mink adapter to the Gento Luna model."""

    def __init__(self, *args, **kwargs):
        kwargs["robot_name"] = "gento_luna"
        super().__init__(*args, **kwargs)
