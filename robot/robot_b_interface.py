"""Real-hardware placeholder for platform B."""


class Dual_arm_controller:
    """Reject real-hardware mode until a platform B controller is provided."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "platform B real-robot control is not configured; use RUN_MODE_REAL=0"
        )

