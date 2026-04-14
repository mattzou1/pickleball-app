"""Shared constants for the pickleball kitchen fault detector.

All thresholds, zone bounds, keypoint indices, and frame tolerances live here.
Tuning is a one-file change.
"""

# ── Court dimensions (feet) ──────────────────────────────────────────────────
# Coordinate system: origin at near-left corner of near-side kitchen
# x-axis along baseline (0-20 ft), y-axis toward far baseline (0-27 ft)
COURT_WIDTH_FT = 20.0
COURT_LENGTH_FT = 27.0  # near baseline to far baseline (kitchen to kitchen)
KITCHEN_DEPTH_FT = 7.0

# Kitchen zone boundaries in court coordinates
NEAR_KITCHEN_Y_MIN = 0.0
NEAR_KITCHEN_Y_MAX = 7.0
FAR_KITCHEN_Y_MIN = 20.0
FAR_KITCHEN_Y_MAX = 27.0

# World coordinates for the 8 calibration corners (feet)
# Order: near kitchen (BL, BR, TR, TL), far kitchen (BL, BR, TR, TL)
# "Bottom" = closer to near baseline, "Top" = closer to net
WORLD_CORNERS = [
    # Near kitchen
    [0.0, 0.0],    # near-left baseline corner
    [20.0, 0.0],   # near-right baseline corner
    [20.0, 7.0],   # near-right kitchen line
    [0.0, 7.0],    # near-left kitchen line
    # Far kitchen
    [0.0, 20.0],   # far-left kitchen line
    [20.0, 20.0],  # far-right kitchen line
    [20.0, 27.0],  # far-left baseline corner
    [0.0, 27.0],   # far-right baseline corner
]

# ── Keypoint indices ─────────────────────────────────────────────────────────
# COCO 17-keypoint (standard YOLOv8-pose)
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

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

# ── Zone check: ankle buffer (when WholeBody foot keypoints unavailable) ─────
# Ankle is ~3-4 inches above ground. Buffer expands zone to catch toe/heel.
ANKLE_BUFFER_FT = 0.5

# ── Fault detection thresholds ───────────────────────────────────────────────
# Minimum consecutive frames with foot in zone to trigger fault
CONSECUTIVE_FRAMES_MIN = 3
# Consecutive frames at which the consecutive component saturates in confidence
CONSECUTIVE_FRAMES_SATURATE = 10

# Confidence tiers (multiplicative formula)
TIER_AUTO_FAULT = 0.5
TIER_REVIEW_NEEDED = 0.15

# ── Ball detection ───────────────────────────────────────────────────────────
# Max frames to interpolate ball position across gaps
BALL_INTERPOLATION_MAX_GAP = 5
# Max frames with no ball detection before marking state UNKNOWN
BALL_UNKNOWN_GAP_FRAMES = 10
# Moving average window for bounce velocity smoothing (frames at 30fps base)
BOUNCE_SMOOTHING_WINDOW = 3

# ── FPS ──────────────────────────────────────────────────────────────────────
BASE_FPS = 30.0


def scale_frame_threshold(threshold_frames: int, actual_fps: float) -> int:
    """Scale a frame-count threshold from 30fps base to actual video fps.

    Example: 3 frames at 30fps = 100ms. At 60fps that's 6 frames for 100ms.
    """
    return max(1, round(threshold_frames * actual_fps / BASE_FPS))
