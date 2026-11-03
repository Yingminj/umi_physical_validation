"""Real-hardware placeholder for Gento Luna."""


class Dual_arm_controller:
    """Reject real-hardware mode until a Gento Luna controller is provided."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Gento Luna real-robot control is not configured; use RUN_MODE_REAL=0"
        )

