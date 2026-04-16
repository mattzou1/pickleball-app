"""Ball detection, interpolation, bounce detection, and per-side state machine.

Ball position is tracked in pixel space. Side (left/right of net) is derived
from the ball's pixel x relative to `net_x_pixel` (net-center camera makes the
threshold effective regardless of court perspective).
"""

import numpy as np

from pickleball.constants import (
    BALL_INTERPOLATION_MAX_GAP,
    BALL_UNKNOWN_GAP_FRAMES,
    BOUNCE_SMOOTHING_WINDOW,
)


# Ball states
LIVE = "LIVE"
BOUNCED = "BOUNCED"
UNKNOWN = "UNKNOWN"


def interpolate_positions(
    detections: list[dict | None],
    max_gap: int = BALL_INTERPOLATION_MAX_GAP,
) -> list[dict | None]:
    """Fill gaps in ball detections with linear interpolation.

    Args:
        detections: list indexed by frame number. Each element is either
            a dict with at least "x" and "y" keys, or None (no detection).
        max_gap: maximum gap length to interpolate across.

    Returns:
        New list with gaps <= max_gap filled via linear interpolation.
        Original detections are not modified. Interpolated entries have
        "interpolated": True added.
    """
    result = list(detections)
    n = len(result)
    i = 0

    while i < n:
        if result[i] is not None:
            i += 1
            continue

        # Find gap boundaries
        gap_start = i
        while i < n and result[i] is None:
            i += 1
        gap_end = i  # first non-None after gap, or n

        gap_len = gap_end - gap_start

        # Need anchors on both sides and gap within limit
        if gap_start == 0 or gap_end >= n:
            continue
        if gap_len > max_gap:
            continue

        before = result[gap_start - 1]
        after = result[gap_end]

        for j in range(gap_start, gap_end):
            t = (j - gap_start + 1) / (gap_len + 1)
            result[j] = {
                "x": before["x"] + t * (after["x"] - before["x"]),
                "y": before["y"] + t * (after["y"] - before["y"]),
                "interpolated": True,
            }

    return result


def compute_vertical_velocity(
    detections: list[dict | None],
    window: int = BOUNCE_SMOOTHING_WINDOW,
) -> list[float | None]:
    """Compute smoothed vertical velocity in pixel coordinates (pixels/frame).

    Positive velocity = y increasing (moving downward in image).

    Args:
        detections: list of ball detections (may contain None).
        window: moving average window for smoothing.

    Returns:
        List of velocities per frame (None where not computable).
    """
    n = len(detections)
    raw_vel = [None] * n

    # Raw velocity: difference between consecutive frames (pixel y)
    for i in range(1, n):
        if detections[i] is not None and detections[i - 1] is not None:
            raw_vel[i] = detections[i]["y"] - detections[i - 1]["y"]

    # Smooth with moving average
    smoothed = [None] * n
    for i in range(n):
        vals = []
        for j in range(max(0, i - window + 1), i + 1):
            if raw_vel[j] is not None:
                vals.append(raw_vel[j])
        if vals:
            smoothed[i] = sum(vals) / len(vals)

    return smoothed


def detect_bounces(
    detections: list[dict | None],
    velocities: list[float | None],
) -> list[int]:
    """Detect bounce frames from pixel-y velocity sign reversals.

    A bounce is a vertical direction change — velocity sign flips between
    consecutive frames. Requires non-None velocity on both sides.

    Args:
        detections: ball detections per frame.
        velocities: smoothed pixel-y velocities.

    Returns:
        List of frame indices where bounces were detected.
    """
    bounces = []
    n = len(velocities)

    for i in range(2, n):
        v_prev = velocities[i - 1]
        v_curr = velocities[i]

        if v_prev is None or v_curr is None:
            continue

        # Sign reversal: velocity was positive (moving away), now negative (coming back)
        # OR was negative, now positive. Either direction change = potential bounce.
        if v_prev > 0 and v_curr < 0:
            bounces.append(i)
        elif v_prev < 0 and v_curr > 0:
            bounces.append(i)

    return bounces


def classify_bounce_side(
    frame_idx: int,
    detections: list[dict | None],
    net_x_pixel: float,
) -> str | None:
    """Determine which side of the net a bounce occurred on (pixel space).

    Args:
        frame_idx: frame where bounce was detected.
        detections: ball detections (pixel coords).
        net_x_pixel: x-pixel of the net center.

    Returns:
        "left" if bounce on left side of net, "right" if on right, None if no detection.
    """
    det = detections[frame_idx]
    if det is None:
        return None
    return "left" if det["x"] < net_x_pixel else "right"


class BallStateMachine:
    """Per-side ball state machine tracking LIVE/BOUNCED/UNKNOWN.

    States:
        left_state: state for the left side
        right_state: state for the right side

    Transitions:
        Bounce on left → left=BOUNCED, right=LIVE
        Bounce on right → right=BOUNCED, left=LIVE
        Ball crosses net heading left → left=LIVE
        Ball crosses net heading right → right=LIVE
        Ball undetected >N frames → both=UNKNOWN
    """

    def __init__(self, unknown_gap_frames: int = BALL_UNKNOWN_GAP_FRAMES):
        self.left_state = LIVE
        self.right_state = LIVE
        self.unknown_gap_frames = unknown_gap_frames
        self.frames_without_detection = 0
        self._prev_side: str | None = None  # track which side ball was last on

    def update_bounce(self, side: str) -> None:
        """Record a bounce on the given side."""
        if side == "left":
            self.left_state = BOUNCED
            self.right_state = LIVE
        elif side == "right":
            self.right_state = BOUNCED
            self.left_state = LIVE

    def update_net_crossing(self, heading_toward: str) -> None:
        """Record ball crossing the net.

        Args:
            heading_toward: "left" or "right" indicating destination side.
        """
        if heading_toward == "left":
            self.left_state = LIVE
        elif heading_toward == "right":
            self.right_state = LIVE

    def update_detection(self, detected: bool, ball_side: str | None = None, net_x_pixel: float | None = None, ball_x_pixel: float | None = None) -> None:
        """Update state based on whether ball was detected this frame.

        Args:
            detected: True if ball was detected.
            ball_side: "left" or "right" based on ball court position.
            net_x_pixel: x-pixel of the net center (for crossing detection).
            ball_x_pixel: x-pixel of the ball (for crossing detection).
        """
        if detected:
            self.frames_without_detection = 0

            # Check for net crossing
            if ball_side is not None and self._prev_side is not None:
                if ball_side != self._prev_side:
                    self.update_net_crossing(heading_toward=ball_side)
            self._prev_side = ball_side
        else:
            self.frames_without_detection += 1
            if self.frames_without_detection > self.unknown_gap_frames:
                self.left_state = UNKNOWN
                self.right_state = UNKNOWN

    def get_state(self, side: str) -> str:
        """Get current state for the given side."""
        if side == "left":
            return self.left_state
        elif side == "right":
            return self.right_state
        return UNKNOWN
