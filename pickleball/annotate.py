"""Frame annotation helpers for debug video output.

Pure logic functions (testable without OpenCV):
    get_foot_keypoint_color  -- color based on in-zone status
    build_status_text        -- status bar text from ball states
    get_bounce_flash         -- flash text if bounce occurred recently
    is_fault_frame           -- True if this frame has a fault

OpenCV drawing functions (called by annotate_frame):
    draw_kitchen_zones       -- filled zone overlays from pixel_corners
    draw_net_line            -- net line between left/right net corners
    draw_foot_keypoints      -- colored dots per player foot keypoint
    draw_ball_trail          -- ball dot + position trail
    draw_status_bar          -- text overlay at top of frame
    draw_bounce_flash        -- large flash text when bounce occurred
    draw_fault_border        -- red frame border on fault frames

annotate_frame               -- composes all of the above onto one frame
"""

import cv2
import numpy as np

from pickleball.pose import (
    check_kitchen,
    extract_ankle_keypoints,
    extract_foot_keypoints,
    is_wholebody_model,
    transform_to_court,
)

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
COLOR_FOOT_SAFE = (0, 220, 0)        # green  — outside kitchen
COLOR_FOOT_FAULT = (0, 0, 220)       # red    — inside kitchen
COLOR_BALL = (0, 220, 220)           # yellow — ball dot
COLOR_BALL_TRAIL = (0, 140, 140)     # dim yellow — trail
COLOR_LEFT_ZONE = (0, 200, 0)        # green overlay — left kitchen
COLOR_RIGHT_ZONE = (200, 80, 0)      # blue overlay  — right kitchen
COLOR_NET = (0, 220, 220)            # yellow — net line
COLOR_FAULT_BORDER = (0, 0, 255)     # red — fault frame border
COLOR_TEXT = (255, 255, 255)         # white text
COLOR_TEXT_BG = (0, 0, 0)           # black text background

BOUNCE_FLASH_WINDOW = 15            # frames to show bounce flash text
BALL_TRAIL_LEN = 10                 # frames to show in ball trail


# ── Pure logic (testable) ─────────────────────────────────────────────────────

def get_foot_keypoint_color(in_zone: bool) -> tuple[int, int, int]:
    """Return BGR color for a foot keypoint dot based on zone status."""
    return COLOR_FOOT_FAULT if in_zone else COLOR_FOOT_SAFE


def build_status_text(left_state: str, right_state: str, fallback_mode: bool) -> str:
    """Build the status bar text string.

    Args:
        left_state: "LIVE", "BOUNCED", or "UNKNOWN"
        right_state: "LIVE", "BOUNCED", or "UNKNOWN"
        fallback_mode: True if ball model is not running

    Returns:
        Single-line status string for display.
    """
    if fallback_mode:
        return "FALLBACK MODE (no ball detection) | Left: ? | Right: ?"
    return f"Left: {left_state}  |  Right: {right_state}"


def get_bounce_flash(
    frame_idx: int,
    bounce_events: list[dict],
    window: int = BOUNCE_FLASH_WINDOW,
) -> str | None:
    """Return flash text if a bounce occurred within the recent window.

    Args:
        frame_idx: current frame number.
        bounce_events: list of dicts with keys "frame" and "side".
        window: number of frames to keep the flash visible.

    Returns:
        Flash string like "BOUNCE: NEAR" or None if no recent bounce.
    """
    # Find the most recent bounce within the window
    best = None
    for event in bounce_events:
        frames_since = frame_idx - event["frame"]
        if 0 <= frames_since <= window:
            if best is None or event["frame"] > best["frame"]:
                best = event
    if best is not None:
        side = best.get("side") or "?"
        return f"BOUNCE: {side.upper()}"
    return None


def is_fault_frame(frame_idx: int, fault_frame_set: set[int]) -> bool:
    """True if this frame has a detected fault."""
    return frame_idx in fault_frame_set


# ── OpenCV drawing ────────────────────────────────────────────────────────────

def draw_kitchen_zones(frame: np.ndarray, config: dict) -> np.ndarray:
    """Draw semi-transparent kitchen zone overlays using stored pixel_corners."""
    overlay = frame.copy()
    corners = config.get("pixel_corners", [])
    if len(corners) < 8:
        return frame

    left_pts = np.array(corners[:4], dtype=np.int32)
    right_pts = np.array(corners[4:8], dtype=np.int32)

    cv2.fillPoly(overlay, [left_pts], COLOR_LEFT_ZONE)
    cv2.fillPoly(overlay, [right_pts], COLOR_RIGHT_ZONE)

    result = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

    # Draw zone outlines
    cv2.polylines(result, [left_pts], True, COLOR_LEFT_ZONE, 2)
    cv2.polylines(result, [right_pts], True, COLOR_RIGHT_ZONE, 2)

    return result


def draw_net_line(frame: np.ndarray, config: dict) -> np.ndarray:
    """Draw the net line between the two net corner points."""
    left = config.get("net_left_pixel")
    right = config.get("net_right_pixel")

    if left and right:
        pt1 = (int(left[0]), int(left[1]))
        pt2 = (int(right[0]), int(right[1]))
        cv2.line(frame, pt1, pt2, COLOR_NET, 2)
        cv2.circle(frame, pt1, 5, COLOR_NET, -1)
        cv2.circle(frame, pt2, 5, COLOR_NET, -1)
    elif config.get("net_x_pixel"):
        # Fallback: old calibration with single net point
        x = int(config["net_x_pixel"])
        h = frame.shape[0]
        cv2.line(frame, (x, 0), (x, h), COLOR_NET, 1)

    return frame


def draw_foot_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Draw foot/ankle keypoint dots, colored by zone status."""
    wholebody = is_wholebody_model(keypoints)

    if wholebody:
        kps = extract_foot_keypoints(keypoints)
    else:
        kps = extract_ankle_keypoints(keypoints)

    for kp in kps:
        px, py = int(kp["x"]), int(kp["y"])
        court_x, court_y = transform_to_court((kp["x"], kp["y"]), H)
        in_zone = check_kitchen(court_x, court_y) is not None
        color = get_foot_keypoint_color(in_zone)
        cv2.circle(frame, (px, py), 8, color, -1)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)  # white outline

    return frame


def draw_ball_trail(
    frame: np.ndarray,
    ball_trail: list[tuple[int, int]],
) -> np.ndarray:
    """Draw ball position trail and current dot."""
    if not ball_trail:
        return frame

    # Draw trail (older positions, fading)
    for i, pt in enumerate(ball_trail[:-1]):
        alpha = (i + 1) / len(ball_trail)
        radius = max(2, int(6 * alpha))
        color = COLOR_BALL_TRAIL
        cv2.circle(frame, pt, radius, color, -1)

    # Draw current position
    cv2.circle(frame, ball_trail[-1], 8, COLOR_BALL, -1)
    cv2.circle(frame, ball_trail[-1], 8, (255, 255, 255), 1)

    return frame


def draw_status_bar(frame: np.ndarray, status_text: str) -> np.ndarray:
    """Draw status text with background at top of frame."""
    padding = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(status_text, font, scale, thickness)
    bar_h = text_h + baseline + padding * 2

    cv2.rectangle(frame, (0, 0), (text_w + padding * 2, bar_h), COLOR_TEXT_BG, -1)
    cv2.putText(
        frame, status_text,
        (padding, text_h + padding),
        font, scale, COLOR_TEXT, thickness,
    )
    return frame


def draw_bounce_flash(frame: np.ndarray, bounce_text: str) -> np.ndarray:
    """Draw large centered bounce flash text."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.8
    thickness = 3

    (text_w, text_h), _ = cv2.getTextSize(bounce_text, font, scale, thickness)
    cx = (w - text_w) // 2
    cy = h // 2 + text_h // 2

    # Shadow
    cv2.putText(frame, bounce_text, (cx + 2, cy + 2), font, scale, (0, 0, 0), thickness + 2)
    # Text
    cv2.putText(frame, bounce_text, (cx, cy), font, scale, COLOR_BALL, thickness)
    return frame


def draw_fault_border(frame: np.ndarray) -> np.ndarray:
    """Draw a red border around the frame to indicate active fault."""
    h, w = frame.shape[:2]
    border = 8
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_FAULT_BORDER, border)
    return frame


def draw_player_id(frame: np.ndarray, track_id: int, keypoints: np.ndarray) -> np.ndarray:
    """Draw player track ID above the player's highest visible keypoint."""
    # Find topmost (lowest y-value) keypoint with reasonable confidence
    min_y = float("inf")
    min_x = None
    for idx in range(min(17, keypoints.shape[0])):
        conf = float(keypoints[idx, 2])
        if conf > 0.3:
            y = float(keypoints[idx, 1])
            if y < min_y:
                min_y = y
                min_x = float(keypoints[idx, 0])

    if min_x is not None:
        label = f"P{track_id}"
        cv2.putText(
            frame, label,
            (int(min_x) - 12, int(min_y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2,
        )
    return frame


# ── Main annotation function ──────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    config: dict,
    pose_data_frame: dict,
    ball_trail: list[tuple[int, int]],
    ball_states: dict,
    bounce_events: list[dict],
    fault_frame_set: set[int],
    H: np.ndarray,
    fallback_mode: bool,
) -> np.ndarray:
    """Compose all annotations onto a single frame.

    Args:
        frame: original BGR frame.
        frame_idx: current frame number.
        config: calibration config dict.
        pose_data_frame: {track_id: keypoints} for this frame.
        ball_trail: list of (x, y) pixel positions (newest last).
        ball_states: {"near": state, "far": state} for this frame.
        bounce_events: list of {"frame": int, "side": str} dicts.
        fault_frame_set: set of frame numbers with faults.
        H: homography matrix.
        fallback_mode: True if ball model not running.

    Returns:
        Annotated frame copy.
    """
    result = frame.copy()

    # Kitchen zones + net line
    result = draw_kitchen_zones(result, config)
    result = draw_net_line(result, config)

    # Player keypoints + IDs
    for track_id, keypoints in pose_data_frame.items():
        result = draw_foot_keypoints(result, keypoints, H)
        result = draw_player_id(result, track_id, keypoints)

    # Ball trail
    result = draw_ball_trail(result, ball_trail)

    # Fault border (drawn before text so text renders on top)
    if is_fault_frame(frame_idx, fault_frame_set):
        result = draw_fault_border(result)

    # Status bar
    left_state = ball_states.get("left", "UNKNOWN")
    right_state = ball_states.get("right", "UNKNOWN")
    status = build_status_text(left_state, right_state, fallback_mode)
    result = draw_status_bar(result, status)

    # Bounce flash
    bounce_text = get_bounce_flash(frame_idx, bounce_events)
    if bounce_text:
        result = draw_bounce_flash(result, bounce_text)

    # Frame counter (bottom-right)
    h, w = result.shape[:2]
    frame_label = f"#{frame_idx}"
    cv2.putText(
        result, frame_label,
        (w - 80, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
    )

    return result
