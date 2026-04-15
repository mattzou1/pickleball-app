"""Tests for pickleball.annotate pure logic functions."""

import pytest

from pickleball.annotate import (
    COLOR_FOOT_FAULT,
    COLOR_FOOT_SAFE,
    build_status_text,
    get_bounce_flash,
    get_foot_keypoint_color,
    is_fault_frame,
)


# ── Foot keypoint color tests ─────────────────────────────────────────────────

def test_foot_color_in_zone():
    """Foot in kitchen zone returns red color."""
    assert get_foot_keypoint_color(True) == COLOR_FOOT_FAULT


def test_foot_color_out_of_zone():
    """Foot outside kitchen zone returns green color."""
    assert get_foot_keypoint_color(False) == COLOR_FOOT_SAFE


def test_foot_color_distinct():
    """In-zone and out-zone colors are different."""
    assert get_foot_keypoint_color(True) != get_foot_keypoint_color(False)


# ── Status text tests ─────────────────────────────────────────────────────────

def test_status_text_live_bounced():
    """Left LIVE + Right BOUNCED renders correctly."""
    text = build_status_text("LIVE", "BOUNCED", fallback_mode=False)
    assert "Left: LIVE" in text
    assert "Right: BOUNCED" in text


def test_status_text_both_unknown():
    """Both UNKNOWN renders correctly."""
    text = build_status_text("UNKNOWN", "UNKNOWN", fallback_mode=False)
    assert "UNKNOWN" in text


def test_status_text_fallback_mode():
    """Fallback mode shows fallback indicator."""
    text = build_status_text("LIVE", "LIVE", fallback_mode=True)
    assert "FALLBACK" in text.upper()


def test_status_text_no_fallback():
    """Normal mode does not show fallback text."""
    text = build_status_text("LIVE", "LIVE", fallback_mode=False)
    assert "FALLBACK" not in text.upper()


# ── Bounce flash tests ────────────────────────────────────────────────────────

def test_bounce_flash_recent():
    """Bounce occurred 5 frames ago within default window — flash shown."""
    events = [{"frame": 0, "side": "left"}]
    result = get_bounce_flash(5, events, window=15)
    assert result is not None
    assert "LEFT" in result.upper()


def test_bounce_flash_at_frame():
    """Bounce on exact same frame — flash shown."""
    events = [{"frame": 10, "side": "right"}]
    result = get_bounce_flash(10, events, window=15)
    assert result is not None
    assert "RIGHT" in result.upper()


def test_bounce_flash_expired():
    """Bounce occurred 20 frames ago, beyond window=15 — no flash."""
    events = [{"frame": 0, "side": "left"}]
    result = get_bounce_flash(20, events, window=15)
    assert result is None


def test_bounce_flash_empty_events():
    """No bounce events — no flash."""
    result = get_bounce_flash(5, [], window=15)
    assert result is None


def test_bounce_flash_before_bounce():
    """Frame before bounce occurred — no flash."""
    events = [{"frame": 10, "side": "left"}]
    result = get_bounce_flash(5, events, window=15)
    assert result is None


def test_bounce_flash_multiple_events():
    """Most recent bounce in window takes priority."""
    events = [
        {"frame": 0, "side": "left"},     # expired (16 frames ago, outside window=15)
        {"frame": 10, "side": "right"},   # recent (6 frames ago)
    ]
    result = get_bounce_flash(16, events, window=15)
    assert result is not None
    assert "RIGHT" in result.upper()


# ── Fault frame tests ─────────────────────────────────────────────────────────

def test_is_fault_frame_match():
    """Frame in fault set returns True."""
    assert is_fault_frame(10, {5, 10, 15}) is True


def test_is_fault_frame_no_match():
    """Frame not in fault set returns False."""
    assert is_fault_frame(7, {5, 10, 15}) is False


def test_is_fault_frame_empty_set():
    """Empty fault set returns False."""
    assert is_fault_frame(5, set()) is False


# ── Net midpoint computation ──────────────────────────────────────────────────

def test_net_midpoint_symmetric():
    """Midpoint of symmetric net corners is exact center."""
    net_left = [100, 200]
    net_right = [300, 200]
    midpoint = (net_left[0] + net_right[0]) / 2
    assert midpoint == pytest.approx(200.0)


def test_net_midpoint_asymmetric():
    """Midpoint handles non-symmetric x positions."""
    net_left = [150, 180]
    net_right = [350, 220]
    midpoint = (net_left[0] + net_right[0]) / 2
    assert midpoint == pytest.approx(250.0)
