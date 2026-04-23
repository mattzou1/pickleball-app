"""Ball detection, interpolation, bounce detection, and per-side state machine.

Ball position is tracked in pixel space. Side (left/right of net) is derived
from the ball's pixel x relative to `net_x_pixel` (net-center camera makes the
threshold effective regardless of court perspective).
"""

import math

import numpy as np

from pickleball.constants import (
    BALL_INTERPOLATION_MAX_GAP,
    BALL_UNKNOWN_GAP_FRAMES,
    BOUNCE_SMOOTHING_WINDOW,
    COCO_LEFT_WRIST,
    COCO_RIGHT_WRIST,
    PADDLE_CONTACT_ANGLE_THRESHOLD_DEG,
    PADDLE_CONTACT_DISTANCE_PX,
    PADDLE_CONTACT_MIN_HORIZONTAL_SPEED_PX,
    PADDLE_CONTACT_SPEED_RATIO,
    PADDLE_CONTACT_WRIST_CONF_THRESHOLD,
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


# ── Paddle-contact detection ────────────────────────────────────────────────
# Trajectory-inflection + wrist-proximity heuristic for the instant of paddle
# contact. Pure pixel-space, reuses existing interpolated ball positions.


def compute_velocity_vectors(
    detections: list[dict | None],
    window: int = BOUNCE_SMOOTHING_WINDOW,
) -> list[tuple[float, float] | None]:
    """Compute smoothed 2D velocity vectors in pixel coords per frame.

    Mirrors `compute_vertical_velocity` but keeps both axes, so direction
    changes (paddle hits) — not just vertical ones (floor bounces) — are
    visible downstream.

    Args:
        detections: list of ball detections (may contain None).
        window: moving average window for smoothing.

    Returns:
        Per-frame list of (vx, vy) tuples or None where not computable.
    """
    n = len(detections)
    raw_vx: list[float | None] = [None] * n
    raw_vy: list[float | None] = [None] * n

    for i in range(1, n):
        if detections[i] is not None and detections[i - 1] is not None:
            raw_vx[i] = detections[i]["x"] - detections[i - 1]["x"]
            raw_vy[i] = detections[i]["y"] - detections[i - 1]["y"]

    smoothed: list[tuple[float, float] | None] = [None] * n
    for i in range(n):
        xs, ys = [], []
        for j in range(max(0, i - window + 1), i + 1):
            if raw_vx[j] is not None:
                xs.append(raw_vx[j])
                ys.append(raw_vy[j])
        if xs:
            smoothed[i] = (sum(xs) / len(xs), sum(ys) / len(ys))

    return smoothed


def _closest_wrist(
    ball_xy: tuple[float, float],
    pose_frame: dict,
    max_distance_px: float,
    conf_threshold: float,
) -> dict | None:
    """Find the wrist keypoint closest to the ball within a max pixel radius.

    Args:
        ball_xy: (x, y) ball pixel position.
        pose_frame: {track_id: keypoints ndarray (K,3)} for this frame.
        max_distance_px: reject candidates beyond this radius.
        conf_threshold: minimum wrist keypoint confidence.

    Returns:
        Dict with {track_id, wrist_side, wrist_xy, distance_px, conf} or None.
    """
    bx, by = ball_xy
    best: dict | None = None

    for track_id, kps in pose_frame.items():
        if kps is None or kps.shape[0] <= COCO_RIGHT_WRIST:
            continue
        for idx, side in ((COCO_LEFT_WRIST, "left"), (COCO_RIGHT_WRIST, "right")):
            wx, wy, wc = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
            if wc < conf_threshold:
                continue
            dist = math.hypot(wx - bx, wy - by)
            if dist > max_distance_px:
                continue
            if best is None or dist < best["distance_px"]:
                best = {
                    "track_id": int(track_id),
                    "wrist_side": side,
                    "wrist_xy": (wx, wy),
                    "distance_px": dist,
                    "conf": wc,
                }
    return best


def detect_paddle_contacts(
    detections: list[dict | None],
    velocities_xy: list[tuple[float, float] | None],
    pose_data: list[dict],
    bounce_frames: list[int],
    angle_threshold_deg: float = PADDLE_CONTACT_ANGLE_THRESHOLD_DEG,
    speed_ratio: float = PADDLE_CONTACT_SPEED_RATIO,
    max_distance_px: float = PADDLE_CONTACT_DISTANCE_PX,
    wrist_conf_threshold: float = PADDLE_CONTACT_WRIST_CONF_THRESHOLD,
    min_horizontal_speed_px: float = PADDLE_CONTACT_MIN_HORIZONTAL_SPEED_PX,
) -> list[dict]:
    """Detect paddle-ball contact events from trajectory inflection + wrist proximity.

    Inflection is measured by comparing the averaged *pre-contact* velocity
    (frames i-2..i-1) against the averaged *post-contact* velocity
    (frames i+1..i+2). Using before/after windows around the contact frame —
    rather than a single smoothed velocity — keeps the direction change sharp
    without amplifying single-frame detection jitter. The `velocities_xy`
    argument is kept for API stability and tests, but is no longer used for
    the inflection check.

    Floor bounces (frames in `bounce_frames`, ±1) are excluded. Each surviving
    inflection is associated with the closest qualifying wrist keypoint at the
    contact frame.

    Args:
        detections: ball detections per frame (post-interpolation).
        velocities_xy: unused (kept for back-compat).
        pose_data: per-frame {track_id: (K,3) keypoints ndarray}.
        bounce_frames: frame indices of detected floor bounces (to exclude).
        angle_threshold_deg: direction change (degrees) to flag an inflection.
        speed_ratio: max(|v_pre|,|v_post|) / min(...) to flag an inflection.
        max_distance_px: max ball-to-wrist distance for association.
        wrist_conf_threshold: min wrist keypoint confidence.

    Returns:
        List of dicts with {frame, track_id, wrist_side, ball_xy, wrist_xy,
        distance_px, angle_change_deg, speed_ratio}, sorted by frame.
    """
    del velocities_xy  # unused; see docstring
    n = len(detections)
    contacts: list[dict] = []
    bounce_set = set(bounce_frames)
    bounce_guard = bounce_set | {b - 1 for b in bounce_set} | {b + 1 for b in bounce_set}

    # Raw frame-to-frame velocity; raw_vel[k] = pos[k] - pos[k-1].
    raw_vel: list[tuple[float, float] | None] = [None] * n
    for k in range(1, n):
        if detections[k] is not None and detections[k - 1] is not None:
            raw_vel[k] = (
                detections[k]["x"] - detections[k - 1]["x"],
                detections[k]["y"] - detections[k - 1]["y"],
            )

    def _avg(vs: list[tuple[float, float] | None]) -> tuple[float, float] | None:
        vals = [v for v in vs if v is not None]
        if not vals:
            return None
        return (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))

    # Score every frame; keep only local maxima (temporal NMS) above threshold.
    # This prevents double-firing when the pre/post window straddles the contact
    # and two adjacent frames both exceed the threshold.
    scores: list[float] = [0.0] * n
    meta: list[tuple[float, float] | None] = [None] * n  # (angle, speed_ratio)
    for i in range(3, n - 2):
        if i in bounce_guard or detections[i] is None:
            continue
        # Non-overlapping pre/post windows — skip raw_vel[i] itself, which
        # spans across the contact moment and would dilute both averages.
        v_pre = _avg([raw_vel[i - 2], raw_vel[i - 1]])
        v_post = _avg([raw_vel[i + 1], raw_vel[i + 2]])
        if v_pre is None or v_post is None:
            continue

        # Hard gate #1: both pre- and post-contact motion must carry
        # meaningful horizontal speed. Rejects ball pickups (~stationary ball).
        if abs(v_pre[0]) < min_horizontal_speed_px or abs(v_post[0]) < min_horizontal_speed_px:
            continue
        # Hard gate #2: average horizontal direction must flip across the
        # contact. Rejects floor bounces (vertical flip only, vx preserved).
        if (v_pre[0] > 0) == (v_post[0] > 0):
            continue
        # Hard gate #3: pin contact to the exact transition frame — the frame
        # where raw horizontal velocity first reverses sign. Without this,
        # adjacent frames surrounding the contact also pass gates #1 and #2.
        r_prev = raw_vel[i - 1]
        r_curr = raw_vel[i]
        if r_prev is None or r_curr is None:
            continue
        if (r_prev[0] > 0) == (r_curr[0] > 0):
            continue

        pre_speed = math.hypot(v_pre[0], v_pre[1])
        post_speed = math.hypot(v_post[0], v_post[1])
        if pre_speed < 1e-6 and post_speed < 1e-6:
            continue

        if pre_speed < 1e-6 or post_speed < 1e-6:
            angle_change = 180.0
        else:
            cos_theta = (v_pre[0] * v_post[0] + v_pre[1] * v_post[1]) / (pre_speed * post_speed)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle_change = math.degrees(math.acos(cos_theta))

        hi, lo = max(pre_speed, post_speed), min(pre_speed, post_speed)
        ratio = hi / lo if lo > 1e-6 else float("inf")

        if angle_change <= angle_threshold_deg and ratio <= speed_ratio:
            continue

        scores[i] = angle_change
        meta[i] = (angle_change, ratio)

    # Temporal NMS: within ±2 frames, keep only the peak.
    for i in range(n):
        if scores[i] <= 0.0:
            continue
        lo_i, hi_i = max(0, i - 2), min(n - 1, i + 2)
        if any(scores[j] > scores[i] for j in range(lo_i, hi_i + 1) if j != i):
            continue
        det = detections[i]
        if det is None or i >= len(pose_data):
            continue
        pose_frame = pose_data[i] or {}
        wrist = _closest_wrist(
            (float(det["x"]), float(det["y"])),
            pose_frame,
            max_distance_px=max_distance_px,
            conf_threshold=wrist_conf_threshold,
        )
        if wrist is None:
            continue

        angle_change, ratio = meta[i]
        contacts.append({
            "frame": i,
            "track_id": wrist["track_id"],
            "wrist_side": wrist["wrist_side"],
            "ball_xy": (float(det["x"]), float(det["y"])),
            "wrist_xy": wrist["wrist_xy"],
            "distance_px": round(wrist["distance_px"], 2),
            "angle_change_deg": round(angle_change, 2),
            "speed_ratio": round(ratio, 3) if ratio != float("inf") else None,
        })

    return contacts


def paddle_contact_near(
    contacts: list[dict],
    frame_idx: int,
    window_frames: int,
    track_id: int | None = None,
) -> dict | None:
    """Return the paddle contact closest (in frames) to `frame_idx` within window.

    If `track_id` is given, only contacts from that track are considered.
    Contacts are assumed sorted by frame (as emitted by `detect_paddle_contacts`).

    Args:
        contacts: list of contact dicts from `detect_paddle_contacts`.
        frame_idx: frame to query around.
        window_frames: max frame distance to consider.
        track_id: optional track filter.

    Returns:
        Nearest contact dict or None.
    """
    best: dict | None = None
    best_dist = window_frames + 1
    for c in contacts:
        if track_id is not None and c["track_id"] != track_id:
            continue
        d = abs(c["frame"] - frame_idx)
        if d > window_frames:
            continue
        if d < best_dist:
            best = c
            best_dist = d
    return best
