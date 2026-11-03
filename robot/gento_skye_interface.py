"""Real-hardware placeholder for Gento Skye."""


class Dual_arm_controller:
    """Reject real-hardware mode until a Gento Skye controller is provided."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Gento Skye real-robot control is not configured; use RUN_MODE_REAL=0"
        )

