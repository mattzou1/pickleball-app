"""Tests for pickleball.fault module."""

import pytest

from pickleball.fault import (
    classify_tier,
    compute_confidence,
    correlate_fault,
    should_trigger_fault,
)


# ── Confidence computation tests ────────────────────────────────────────────

def test_confidence_all_high():
    """(0.9 * 1.0 * 0.95) = 0.855 -> high confidence."""
    conf = compute_confidence(ankle_conf=0.9, consecutive_frames=10, ball_conf=0.95)
    assert conf == pytest.approx(0.855)


def test_confidence_low_ankle():
    """(0.3 * 1.0 * 0.9) = 0.27."""
    conf = compute_confidence(ankle_conf=0.3, consecutive_frames=10, ball_conf=0.9)
    assert conf == pytest.approx(0.27)


def test_confidence_one_frame():
    """(0.9 * 0.1 * 0.9) = 0.081 -> very low (1/10 consecutive factor)."""
    conf = compute_confidence(ankle_conf=0.9, consecutive_frames=1, ball_conf=0.9)
    assert conf == pytest.approx(0.081)


def test_confidence_unknown_ball():
    """ball_conf=0.0 -> composite=0.0."""
    conf = compute_confidence(ankle_conf=0.9, consecutive_frames=10, ball_conf=0.0)
    assert conf == 0.0


def test_confidence_saturates_at_10():
    """Consecutive factor caps at 1.0 for 10+ frames."""
    conf_10 = compute_confidence(ankle_conf=0.9, consecutive_frames=10, ball_conf=0.9)
    conf_20 = compute_confidence(ankle_conf=0.9, consecutive_frames=20, ball_conf=0.9)
    assert conf_10 == conf_20


# ── Tier classification tests ───────────────────────────────────────────────

def test_tier_auto_fault():
    """0.85 -> AUTO_FAULT (>= 0.5)."""
    assert classify_tier(0.85) == "AUTO_FAULT"


def test_tier_auto_fault_boundary():
    """0.5 -> AUTO_FAULT (exactly at threshold)."""
    assert classify_tier(0.5) == "AUTO_FAULT"


def test_tier_review_needed():
    """0.3 -> REVIEW_NEEDED (>= 0.15, < 0.5)."""
    assert classify_tier(0.3) == "REVIEW_NEEDED"


def test_tier_review_needed_boundary():
    """0.15 -> REVIEW_NEEDED (exactly at threshold)."""
    assert classify_tier(0.15) == "REVIEW_NEEDED"


def test_tier_filtered():
    """0.1 -> FILTERED (< 0.15)."""
    assert classify_tier(0.1) == "FILTERED"


# ── Fault trigger tests ─────────────────────────────────────────────────────

def test_correlate_live_in_zone():
    """Ankle in zone + LIVE + 3 frames = FAULT."""
    is_fault, reason = should_trigger_fault("LIVE", consecutive_frames=3, min_consecutive=3)
    assert is_fault is True
    assert reason == "fault"


def test_correlate_bounced_in_zone():
    """Ankle in zone + BOUNCED = no fault."""
    is_fault, reason = should_trigger_fault("BOUNCED", consecutive_frames=5, min_consecutive=3)
    assert is_fault is False
    assert reason == "bounced"


def test_correlate_unknown_in_zone():
    """Ankle in zone + UNKNOWN = flag for review."""
    is_fault, reason = should_trigger_fault("UNKNOWN", consecutive_frames=5, min_consecutive=3)
    assert is_fault is True
    assert reason == "unknown"


def test_correlate_below_consecutive():
    """Ankle in zone + LIVE + 2 frames = no fault (below min)."""
    is_fault, reason = should_trigger_fault("LIVE", consecutive_frames=2, min_consecutive=3)
    assert is_fault is False
    assert reason == "below_consecutive"


# ── Full correlation tests ───────────────────────────────────────────────────

def test_correlate_fault_full():
    """Full fault correlation produces correct output."""
    zone_hit = {
        "zone": "kitchen",
        "keypoint_side": "left",
        "foot_side": "left",
        "pixel": (300, 400),
        "conf": 0.8,
        "source": "foot",
    }
    result = correlate_fault(
        zone_hit=zone_hit,
        ball_state="LIVE",
        consecutive_frames=10,
        ankle_conf=0.9,
        ball_conf=0.95,
        min_consecutive=3,
    )
    assert result is not None
    assert result["confidence_tier"] == "AUTO_FAULT"
    assert result["composite_confidence"] == pytest.approx(0.855)
    assert result["zone"] == "kitchen"
    assert result["foot_side"] == "left"
    assert result["ball_state"] == "LIVE"
    assert "court_coord" not in result


def test_correlate_fault_filtered():
    """Low confidence -> FILTERED -> returns None."""
    zone_hit = {
        "zone": "kitchen",
        "keypoint_side": "left",
        "foot_side": "left",
        "pixel": (300, 400),
        "conf": 0.3,
        "source": "foot",
    }
    result = correlate_fault(
        zone_hit=zone_hit,
        ball_state="LIVE",
        consecutive_frames=1,
        ankle_conf=0.3,
        ball_conf=0.5,
        min_consecutive=1,
    )
    # 0.3 * 0.1 * 0.5 = 0.015 < 0.15 -> FILTERED
    assert result is None


def test_correlate_fault_bounced_no_fault():
    """BOUNCED state -> no fault regardless of confidence."""
    zone_hit = {
        "zone": "kitchen",
        "keypoint_side": "left",
        "foot_side": "left",
        "pixel": (300, 400),
        "conf": 0.9,
        "source": "foot",
    }
    result = correlate_fault(
        zone_hit=zone_hit,
        ball_state="BOUNCED",
        consecutive_frames=10,
        ankle_conf=0.9,
        ball_conf=0.95,
        min_consecutive=3,
    )
    assert result is None
