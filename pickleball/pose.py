"""Pose tracking, foot keypoint extraction, homography transform, and zone check.

Supports two model types:
- WholeBody (133+ keypoints): uses 6 foot keypoints (toe + heel per foot) for zone check
- COCO 17-keypoint: falls back to ankle keypoints + ANKLE_BUFFER_FT expansion
"""

import cv2
import numpy as np

from pickleball.constants import (
    ANKLE_BUFFER_FT,
    ANKLE_CONF_THRESHOLD,
    COCO_LEFT_ANKLE,
    COCO_RIGHT_ANKLE,
    LEFT_KITCHEN_Y_MAX,
    LEFT_KITCHEN_Y_MIN,
    FOOT_KP_CONF_THRESHOLD,
    RIGHT_KITCHEN_Y_MAX,
    RIGHT_KITCHEN_Y_MIN,
    WHOLEBODY_FOOT_INDICES,
    WORLD_CORNERS,
)


def compute_homography(pixel_corners: list[list[float]]) -> np.ndarray:
    """Compute homography from 8 pixel corners to court world coordinates.

    Args:
        pixel_corners: 8 pixel [x, y] points matching WORLD_CORNERS order.

    Returns:
        3x3 homography matrix.
    """
    src = np.array(pixel_corners, dtype=np.float64)
    dst = np.array(WORLD_CORNERS, dtype=np.float64)
    H, status = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        raise ValueError("Homography computation failed. Check corner points.")
    return H


def transform_to_court(pixel_point: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Transform a pixel coordinate to court coordinates using homography.

    Args:
        pixel_point: (x, y) in pixels.
        H: 3x3 homography matrix.

    Returns:
        (x_ft, y_ft) in court coordinates.
    """
    pt = np.array([[pixel_point[0], pixel_point[1]]], dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])


def check_kitchen(court_x: float, court_y: float, buffer: float = 0.0) -> str | None:
    """Check if a court coordinate is inside a kitchen zone.

    Args:
        court_x: x position in feet (0-20).
        court_y: y position in feet (0-27).
        buffer: expand zone boundaries by this many feet (for ankle fallback).

    Returns:
        "left" if in left kitchen, "right" if in right kitchen, None if outside both.
    """
    if (LEFT_KITCHEN_Y_MIN - buffer) <= court_y <= (LEFT_KITCHEN_Y_MAX + buffer):
        if -buffer <= court_x <= (20.0 + buffer):
            return "left"
    if (RIGHT_KITCHEN_Y_MIN - buffer) <= court_y <= (RIGHT_KITCHEN_Y_MAX + buffer):
        if -buffer <= court_x <= (20.0 + buffer):
            return "right"
    return None


def is_wholebody_model(keypoints: np.ndarray) -> bool:
    """Detect if model outputs WholeBody keypoints (133+) vs COCO 17."""
    if keypoints is None or len(keypoints.shape) < 1:
        return False
    # keypoints shape: (num_keypoints, 2 or 3) for a single person
    num_kp = keypoints.shape[0] if len(keypoints.shape) == 2 else keypoints.shape[-2]
    return num_kp >= 23  # at minimum has foot keypoints (indices 17-22)


def extract_foot_keypoints(keypoints: np.ndarray) -> list[dict]:
    """Extract foot keypoints from WholeBody model output.

    Args:
        keypoints: (num_keypoints, 3) array where columns are [x, y, confidence].

    Returns:
        List of dicts with keys: index, x, y, conf, side ("left" or "right").
    """
    results = []
    for idx in WHOLEBODY_FOOT_INDICES:
        if idx >= keypoints.shape[0]:
            continue
        x, y, conf = keypoints[idx]
        if conf < FOOT_KP_CONF_THRESHOLD:
            continue
        side = "left" if idx <= 19 else "right"
        results.append({"index": idx, "x": float(x), "y": float(y), "conf": float(conf), "side": side})
    return results


def extract_ankle_keypoints(keypoints: np.ndarray) -> list[dict]:
    """Extract ankle keypoints from COCO 17-keypoint model output.

    Args:
        keypoints: (num_keypoints, 3) array where columns are [x, y, confidence].

    Returns:
        List of dicts with keys: index, x, y, conf, side ("left" or "right").
    """
    results = []
    for idx, side in [(COCO_LEFT_ANKLE, "left"), (COCO_RIGHT_ANKLE, "right")]:
        if idx >= keypoints.shape[0]:
            continue
        x, y, conf = keypoints[idx]
        if conf < ANKLE_CONF_THRESHOLD:
            continue
        results.append({"index": idx, "x": float(x), "y": float(y), "conf": float(conf), "side": side})
    return results


def check_player_in_kitchen(
    keypoints: np.ndarray,
    H: np.ndarray,
) -> list[dict]:
    """Check if any of a player's foot/ankle keypoints are in a kitchen zone.

    Auto-detects model type (WholeBody vs COCO 17) and uses appropriate keypoints.
    WholeBody: foot keypoints with FOOT_KP_CONF_THRESHOLD for zone check.
    COCO 17: ankle keypoints with ANKLE_BUFFER_FT zone expansion.

    Args:
        keypoints: (num_keypoints, 3) array for one person.
        H: 3x3 homography matrix.

    Returns:
        List of zone hits, each a dict with:
            zone: "left" or "right" (kitchen zone)
            keypoint_side: "left" or "right" (body side — which foot)
            pixel: (x, y)
            court_coord: (x_ft, y_ft)
            conf: keypoint confidence
            source: "foot" or "ankle"
    """
    hits = []
    wholebody = is_wholebody_model(keypoints)

    if wholebody:
        foot_kps = extract_foot_keypoints(keypoints)
        for kp in foot_kps:
            court_x, court_y = transform_to_court((kp["x"], kp["y"]), H)
            zone = check_kitchen(court_x, court_y, buffer=0.0)
            if zone:
                hits.append({
                    "zone": zone,
                    "keypoint_side": kp["side"],
                    "pixel": (kp["x"], kp["y"]),
                    "court_coord": (court_x, court_y),
                    "conf": kp["conf"],
                    "source": "foot",
                })
    else:
        ankle_kps = extract_ankle_keypoints(keypoints)
        for kp in ankle_kps:
            court_x, court_y = transform_to_court((kp["x"], kp["y"]), H)
            zone = check_kitchen(court_x, court_y, buffer=ANKLE_BUFFER_FT)
            if zone:
                hits.append({
                    "zone": zone,
                    "keypoint_side": kp["side"],
                    "pixel": (kp["x"], kp["y"]),
                    "court_coord": (court_x, court_y),
                    "conf": kp["conf"],
                    "source": "ankle",
                })

    return hits


def get_ankle_confidence(keypoints: np.ndarray) -> float:
    """Get the best ankle confidence for composite scoring.

    Uses COCO ankle keypoints (15, 16) regardless of model type.
    Returns max confidence across both ankles if above threshold.

    Args:
        keypoints: (num_keypoints, 3) array for one person.

    Returns:
        Best ankle confidence (0.0 if both below threshold).
    """
    best = 0.0
    for idx in [COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE]:
        if idx >= keypoints.shape[0]:
            continue
        conf = float(keypoints[idx, 2])
        if conf >= ANKLE_CONF_THRESHOLD and conf > best:
            best = conf
    return best
