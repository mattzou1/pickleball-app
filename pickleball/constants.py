"""Shared constants for the pickleball kitchen fault detector.

All thresholds, keypoint indices, and frame tolerances live here.
Tuning is a one-file change.
"""

# ── Keypoint indices ─────────────────────────────────────────────────────────
# COCO 17-keypoint (standard YOLOv8-pose)
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16
# Wrists share the same indices in COCO 17 and COCO-WholeBody body block.
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10

# COCO-WholeBody foot keypoints (indices 17-22)
# Available when using a WholeBody model (ViTPose, RTMPose-WholeBody)
WHOLEBODY_FOOT_KEYPOINTS = {
    "left_big_toe": 17,
    "left_small_toe": 18,
    "left_heel": 19,
    "right_big_toe": 20,
    "right_small_toe": 21,
    "right_heel": 22,
}
WHOLEBODY_FOOT_INDICES = list(WHOLEBODY_FOOT_KEYPOINTS.values())

# ── Confidence thresholds ────────────────────────────────────────────────────
# Foot keypoints (WholeBody) are noisier; lower threshold for zone check
FOOT_KP_CONF_THRESHOLD = 0.2
# Ankle keypoints used for confidence scoring; higher threshold
ANKLE_CONF_THRESHOLD = 0.5

# ── Zone detection ───────────────────────────────────────────────────────────
# Outward tolerance for the kitchen-polygon hit test (pixels). A foot keypoint
# outside the polygon by up to this many pixels still counts as inside.
# Compensates for pose-keypoint pixel noise so a toe on or just outside the
# kitchen line still registers as a fault. Pixel-based (not physical) — the
# effective physical slack varies slightly with camera perspective, acceptable
# given the camera is net-center.
KITCHEN_BOUNDARY_TOLERANCE_PX = 5.0

# ── Fault detection thresholds ───────────────────────────────────────────────
# Minimum consecutive frames with foot in zone to trigger fault
CONSECUTIVE_FRAMES_MIN = 3
# Consecutive frames at which the consecutive component saturates in confidence
CONSECUTIVE_FRAMES_SATURATE = 10
# Tolerate this many consecutive inactive frames before resetting the consecutive
# counter. Prevents single-frame tracker flickers/occlusions from erasing a
# legitimate streak. FPS-scaled via scale_frame_threshold.
CONSECUTIVE_GAP_TOLERANCE = 1

# Confidence tiers (multiplicative formula)
TIER_AUTO_FAULT = 0.5
TIER_REVIEW_NEEDED = 0.15

# ── Player selection ─────────────────────────────────────────────────────────
# With the camera at net-center, the on-court players are the people closest
# to the camera and therefore have the largest bboxes in pixel space. Keep the
# N largest detections and drop everything else (adjacent-court players,
# spectators). 4 covers singles + doubles.
MAX_PLAYERS_KEPT = 4

# ── Ball detection ───────────────────────────────────────────────────────────
# Max frames to interpolate ball position across gaps
BALL_INTERPOLATION_MAX_GAP = 5
# Max frames with no ball detection before marking state UNKNOWN
BALL_UNKNOWN_GAP_FRAMES = 10
# Moving average window for bounce velocity smoothing (frames at 30fps base)
BOUNCE_SMOOTHING_WINDOW = 3

# ── Paddle-contact detection ─────────────────────────────────────────────────
# Trajectory inflection (ball velocity vector changes sharply) co-located with
# a player's wrist keypoint. Used to pin the *instant* of paddle-ball contact
# so kitchen entries can be tiered as true volleys vs. loitering.
#
# An inflection fires if the velocity direction changes by more than
# PADDLE_CONTACT_ANGLE_THRESHOLD_DEG OR the speed ratio exceeds
# PADDLE_CONTACT_SPEED_RATIO (ratio is max(|v_i|,|v_{i-1}|) / min(...)).
PADDLE_CONTACT_ANGLE_THRESHOLD_DEG = 35.0
PADDLE_CONTACT_SPEED_RATIO = 1.8
# Minimum horizontal speed (pixels/frame) on BOTH sides of the contact.
# A true paddle hit reverses horizontal direction — both pre- and post-contact
# must carry meaningful horizontal motion. Floor bounces (pure vertical flip)
# and ball pickups (near-stationary ball) fail this gate.
PADDLE_CONTACT_MIN_HORIZONTAL_SPEED_PX = 2.0
# Max pixel distance from ball center to wrist to associate the inflection
# with a player. The ball is at the paddle face at contact, not the wrist,
# so this needs to be generous enough to cover paddle length + arm extension.
PADDLE_CONTACT_DISTANCE_PX = 150.0
# Minimum wrist keypoint confidence to consider it a valid candidate.
PADDLE_CONTACT_WRIST_CONF_THRESHOLD = 0.3
# Window (in frames at 30fps base, fps-scaled) around a kitchen-entry frame
# in which a paddle-contact event is considered "co-occurring" with the entry.
PADDLE_CONTACT_WINDOW_FRAMES = 5
# Confidence multiplier applied to composite when a paddle contact by the same
# track_id co-occurs with a LIVE kitchen entry (capped at 1.0).
PADDLE_CONTACT_CONFIDENCE_BOOST = 1.35
# Confidence multiplier applied when no paddle contact co-occurs with a LIVE
# kitchen entry — demotes ambiguous "standing in kitchen" cases.
PADDLE_CONTACT_MISSING_PENALTY = 0.6

# ── FPS ──────────────────────────────────────────────────────────────────────
BASE_FPS = 30.0


def scale_frame_threshold(threshold_frames: int, actual_fps: float) -> int:
    """Scale a frame-count threshold from 30fps base to actual video fps.

    Example: 3 frames at 30fps = 100ms. At 60fps that's 6 frames for 100ms.
    """
    return max(1, round(threshold_frames * actual_fps / BASE_FPS))
