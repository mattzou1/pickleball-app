"""Confidence scoring, tier classification, and fault correlation.

Composite confidence formula:
    ankle_keypoint_confidence × min(consecutive_frames / 10, 1.0) × ball_confidence

Tiers:
    AUTO_FAULT:    confidence >= 0.5
    REVIEW_NEEDED: confidence >= 0.15
    FILTERED:      confidence < 0.15
"""

from pickleball.ball import BOUNCED, LIVE, UNKNOWN
from pickleball.constants import (
    CONSECUTIVE_FRAMES_MIN,
    CONSECUTIVE_FRAMES_SATURATE,
    PADDLE_CONTACT_CONFIDENCE_BOOST,
    PADDLE_CONTACT_MISSING_PENALTY,
    TIER_AUTO_FAULT,
    TIER_REVIEW_NEEDED,
)


def compute_confidence(
    ankle_conf: float,
    consecutive_frames: int,
    ball_conf: float,
) -> float:
    """Compute composite confidence score.

    Args:
        ankle_conf: keypoint confidence for the best ankle (0.0-1.0).
        consecutive_frames: number of consecutive frames foot in zone.
        ball_conf: ball detection confidence (0.0-1.0). 0.0 if UNKNOWN.

    Returns:
        Composite confidence (0.0-1.0).
    """
    consecutive_factor = min(consecutive_frames / CONSECUTIVE_FRAMES_SATURATE, 1.0)
    return ankle_conf * consecutive_factor * ball_conf


def classify_tier(confidence: float) -> str:
    """Classify confidence into a tier.

    Args:
        confidence: composite confidence score.

    Returns:
        "AUTO_FAULT", "REVIEW_NEEDED", or "FILTERED".
    """
    if confidence >= TIER_AUTO_FAULT:
        return "AUTO_FAULT"
    elif confidence >= TIER_REVIEW_NEEDED:
        return "REVIEW_NEEDED"
    else:
        return "FILTERED"


def should_trigger_fault(
    ball_state: str,
    consecutive_frames: int,
    min_consecutive: int = CONSECUTIVE_FRAMES_MIN,
) -> tuple[bool, str]:
    """Determine if a kitchen zone entry should be flagged as a fault.

    Args:
        ball_state: current ball state for the side ("LIVE", "BOUNCED", "UNKNOWN").
        consecutive_frames: frames the player has been in the zone.
        min_consecutive: minimum frames to trigger (scaled for fps).

    Returns:
        Tuple of (is_fault, reason).
        reason is one of: "fault", "bounced", "unknown", "below_consecutive".
    """
    if consecutive_frames < min_consecutive:
        return False, "below_consecutive"

    if ball_state == LIVE:
        return True, "fault"
    elif ball_state == BOUNCED:
        return False, "bounced"
    elif ball_state == UNKNOWN:
        # Flag for review but don't auto-fault
        return True, "unknown"
    else:
        return False, "below_consecutive"


def correlate_fault(
    zone_hit: dict,
    ball_state: str,
    consecutive_frames: int,
    ankle_conf: float,
    ball_conf: float,
    min_consecutive: int = CONSECUTIVE_FRAMES_MIN,
    paddle_contact: dict | None = None,
) -> dict | None:
    """Full fault correlation: check trigger, compute confidence, classify tier.

    Args:
        zone_hit: dict from pose.check_player_in_kitchen with zone, pixel, etc.
        ball_state: current ball state for the relevant side.
        consecutive_frames: frames player has been in zone.
        ankle_conf: best ankle keypoint confidence.
        ball_conf: ball detection confidence (0.0 if UNKNOWN).
        min_consecutive: minimum frames threshold (fps-scaled).
        paddle_contact: optional paddle-contact event (from
            `ball.paddle_contact_near`) co-occurring with this kitchen entry.
            When present and ball_state == LIVE, boosts composite confidence
            and marks the fault as a confirmed volley. When explicitly absent
            (None) on a LIVE entry, applies a penalty so ambiguous
            standing-in-kitchen cases don't auto-fault.

    Returns:
        Fault dict if triggered and not FILTERED, None otherwise.
    """
    is_fault, reason = should_trigger_fault(ball_state, consecutive_frames, min_consecutive)

    if not is_fault:
        return None

    confidence = compute_confidence(ankle_conf, consecutive_frames, ball_conf)

    # Paddle-contact only informs LIVE-state kitchen entries (the volley
    # question). BOUNCED is already filtered above; UNKNOWN is a separate
    # review tier and we don't want to demote it further.
    if ball_state == LIVE:
        if paddle_contact is not None:
            confidence = min(1.0, confidence * PADDLE_CONTACT_CONFIDENCE_BOOST)
            reason = "volley_confirmed"
        else:
            confidence = confidence * PADDLE_CONTACT_MISSING_PENALTY

    tier = classify_tier(confidence)

    if tier == "FILTERED":
        return None

    return {
        "zone": zone_hit["zone"],
        "keypoint_side": zone_hit["keypoint_side"],
        "foot_side": zone_hit.get("foot_side"),
        "pixel": zone_hit["pixel"],
        "keypoint_confidence": zone_hit["conf"],
        "source": zone_hit["source"],
        "ankle_keypoint_confidence": ankle_conf,
        "consecutive_frames_in_zone": consecutive_frames,
        "ball_state": ball_state,
        "ball_detection_confidence": ball_conf,
        "composite_confidence": round(confidence, 4),
        "confidence_tier": tier,
        "fault_reason": reason,
        "paddle_contact": paddle_contact,
    }
