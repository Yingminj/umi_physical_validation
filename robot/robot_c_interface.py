"""Real-hardware placeholder for platform C."""


class Dual_arm_controller:
    """Reject real-hardware mode until a platform C controller is provided."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "platform C real-robot control is not configured; use RUN_MODE_REAL=0"
        )

