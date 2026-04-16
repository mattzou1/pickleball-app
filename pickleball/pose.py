"""Pose keypoint extraction and pixel-space kitchen-zone check.

Supports two model types:
- WholeBody (133+ keypoints): uses 6 foot keypoints (toe + heel per foot) for zone check
- COCO 17-keypoint: falls back to ankle keypoints

Zone detection is pure pixel-space: `point_in_kitchen` tests a foot pixel against
the 4-corner kitchen polygon via `cv2.pointPolygonTest`. No homography is used.
"""

import cv2
import numpy as np

from pickleball.constants import (
    ANKLE_CONF_THRESHOLD,
    COCO_LEFT_ANKLE,
    COCO_RIGHT_ANKLE,
    FOOT_KP_CONF_THRESHOLD,
    KITCHEN_BOUNDARY_TOLERANCE_PX,
    WHOLEBODY_FOOT_INDICES,
)


def point_in_kitchen(
    x: float,
    y: float,
    polygon: np.ndarray,
    tolerance_px: float = KITCHEN_BOUNDARY_TOLERANCE_PX,
) -> bool:
    """Test whether a pixel lies inside (or within `tolerance_px` of) the kitchen polygon.

    Points up to `tolerance_px` outside the polygon still count as inside,
    compensating for pose-keypoint pixel noise on near-line toe touches.

    Args:
        x, y: pixel coordinates.
        polygon: (N, 2) int32 array of polygon vertices in pixel space.
        tolerance_px: outward slack in pixels. Defaults to
            KITCHEN_BOUNDARY_TOLERANCE_PX. Pass 0.0 for strict-boundary behavior.

    Returns:
        True if the point is inside, on, or within `tolerance_px` of the polygon boundary.
    """
    result = cv2.pointPolygonTest(polygon, (float(x), float(y)), True)
    return result >= -tolerance_px


def is_wholebody_model(keypoints: np.ndarray) -> bool:
    """Detect if model outputs WholeBody keypoints (133+) vs COCO 17."""
    if keypoints is None or len(keypoints.shape) < 1:
        return False
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
    polygon: np.ndarray,
    net_x_pixel: float | None = None,
) -> list[dict]:
    """Check if any of a player's foot/ankle keypoints are inside the kitchen polygon.

    Auto-detects model type (WholeBody vs COCO 17) and uses appropriate keypoints.
    Zone test is pure pixel-space (`cv2.pointPolygonTest`), no homography.

    Args:
        keypoints: (num_keypoints, 3) array for one person.
        polygon: (N, 2) int32 array of kitchen polygon vertices in pixel space.
        net_x_pixel: x-pixel of the net center. When provided, each hit is tagged
            with foot_side ("left"/"right") indicating which side of the net the
            foot is on. Used downstream to pick the correct ball state.

    Returns:
        List of zone hits, each a dict with:
            zone: "kitchen" (single combined zone)
            keypoint_side: "left" or "right" (body side — which foot)
            foot_side: "left" or "right" (court-side based on net_x_pixel, or None)
            pixel: (x, y)
            conf: keypoint confidence
            source: "foot" or "ankle"
    """
    hits = []
    wholebody = is_wholebody_model(keypoints)

    if wholebody:
        kps = extract_foot_keypoints(keypoints)
        source = "foot"
    else:
        kps = extract_ankle_keypoints(keypoints)
        source = "ankle"

    for kp in kps:
        if not point_in_kitchen(kp["x"], kp["y"], polygon):
            continue
        foot_side = None
        if net_x_pixel is not None:
            foot_side = "left" if kp["x"] < net_x_pixel else "right"
        hits.append({
            "zone": "kitchen",
            "keypoint_side": kp["side"],
            "foot_side": foot_side,
            "pixel": (kp["x"], kp["y"]),
            "conf": kp["conf"],
            "source": source,
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


def get_pose_confidence(keypoints: np.ndarray) -> float:
    """Best available keypoint confidence for composite scoring.

    Prefers ankle conf if ≥ ANKLE_CONF_THRESHOLD. For WholeBody output,
    falls back to the best foot-keypoint conf ≥ FOOT_KP_CONF_THRESHOLD when
    the ankle is occluded/low-confidence. This prevents silent drops of
    valid faults in WholeBody mode where foot keypoints are reliable but
    the ankle isn't (composite × 0 → FILTERED).

    Args:
        keypoints: (num_keypoints, 3) array for one person.

    Returns:
        Best confidence (0.0 if nothing reliable).
    """
    ankle_conf = get_ankle_confidence(keypoints)
    if ankle_conf > 0.0:
        return ankle_conf
    if is_wholebody_model(keypoints):
        best = 0.0
        for idx in WHOLEBODY_FOOT_INDICES:
            if idx >= keypoints.shape[0]:
                continue
            conf = float(keypoints[idx, 2])
            if conf >= FOOT_KP_CONF_THRESHOLD and conf > best:
                best = conf
        return best
    return 0.0
