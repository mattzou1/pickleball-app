"""Tests for pickleball.pose module."""

import numpy as np
import pytest

from pickleball.pose import (
    check_kitchen,
    check_player_in_kitchen,
    compute_homography,
    extract_ankle_keypoints,
    extract_foot_keypoints,
    get_ankle_confidence,
    is_wholebody_model,
    transform_to_court,
)
from pickleball.constants import WORLD_CORNERS


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def identity_homography():
    """Homography where pixel coords == court coords (for testing logic)."""
    # Use corners that map 1:1 (pixel coords = world coords)
    pixel_corners = WORLD_CORNERS.copy()
    H = compute_homography(pixel_corners)
    return H


@pytest.fixture
def sample_homography():
    """Realistic homography from a scaled/shifted pixel space."""
    # Pixel corners: scale world by 50 and shift by (100, 50)
    pixel_corners = [[c[0] * 50 + 100, c[1] * 50 + 50] for c in WORLD_CORNERS]
    H = compute_homography(pixel_corners)
    return H


def make_keypoints(num_kp=17, overrides=None):
    """Create a keypoints array with zeros, then apply overrides.

    Args:
        num_kp: total keypoints (17 for COCO, 23+ for WholeBody).
        overrides: dict mapping index -> (x, y, conf).
    """
    kps = np.zeros((num_kp, 3), dtype=np.float32)
    if overrides:
        for idx, (x, y, conf) in overrides.items():
            kps[idx] = [x, y, conf]
    return kps


# ── Homography tests ────────────────────────────────────────────────────────

def test_transform_to_court_known_point(sample_homography):
    """Verify homography maps a known pixel to expected court coord."""
    # Pixel (100, 50) should map to court (0, 0) - left baseline corner
    cx, cy = transform_to_court((100, 50), sample_homography)
    assert abs(cx - 0.0) < 0.1
    assert abs(cy - 0.0) < 0.1


def test_transform_to_court_far_corner(sample_homography):
    """Verify far-right baseline corner maps correctly."""
    # Pixel (100 + 20*50, 50 + 27*50) = (1100, 1400) should map to (20, 27)
    cx, cy = transform_to_court((1100, 1400), sample_homography)
    assert abs(cx - 20.0) < 0.1
    assert abs(cy - 27.0) < 0.1


# ── Zone check tests ────────────────────────────────────────────────────────

def test_check_kitchen_left():
    """Court coord (10, 3) is in left kitchen."""
    assert check_kitchen(10.0, 3.0) == "left"


def test_check_kitchen_right():
    """Court coord (10, 23) is in right kitchen."""
    assert check_kitchen(10.0, 23.0) == "right"


def test_check_kitchen_boundary():
    """Court coord (10, 7) is ON the left kitchen line (fault)."""
    assert check_kitchen(10.0, 7.0) == "left"


def test_check_kitchen_right_boundary():
    """Court coord (10, 20) is ON the right kitchen line (fault)."""
    assert check_kitchen(10.0, 20.0) == "right"


def test_check_outside_kitchen():
    """Court coord (10, 10) is not in either kitchen."""
    assert check_kitchen(10.0, 10.0) is None


def test_check_kitchen_with_buffer():
    """Buffer expands zone: (10, 7.4) is outside, but within 0.5 buffer."""
    assert check_kitchen(10.0, 7.4) is None
    assert check_kitchen(10.0, 7.4, buffer=0.5) == "left"


# ── Model detection tests ───────────────────────────────────────────────────

def test_is_wholebody_model_17kp():
    kps = make_keypoints(17)
    assert is_wholebody_model(kps) is False


def test_is_wholebody_model_23kp():
    kps = make_keypoints(23)
    assert is_wholebody_model(kps) is True


def test_is_wholebody_model_133kp():
    kps = make_keypoints(133)
    assert is_wholebody_model(kps) is True


# ── Foot keypoint extraction tests ──────────────────────────────────────────

def test_foot_kp_below_zone_threshold_skipped():
    """Foot keypoint with confidence 0.1 is skipped (< 0.2)."""
    kps = make_keypoints(23, {17: (100, 200, 0.1)})
    result = extract_foot_keypoints(kps)
    assert len(result) == 0


def test_foot_kp_above_zone_threshold_used():
    """Foot keypoint with confidence 0.25 is used for zone check."""
    kps = make_keypoints(23, {17: (100, 200, 0.25)})
    result = extract_foot_keypoints(kps)
    assert len(result) == 1
    assert result[0]["index"] == 17
    assert result[0]["side"] == "left"


def test_wholebody_indices_correct():
    """Keypoints 17-22 are extracted, not 15-16."""
    overrides = {
        15: (50, 50, 0.9),   # left ankle - should NOT be in foot keypoints
        16: (60, 60, 0.9),   # right ankle - should NOT be in foot keypoints
        17: (100, 200, 0.5),  # left big toe
        18: (105, 205, 0.5),  # left small toe
        19: (95, 195, 0.5),   # left heel
        20: (200, 200, 0.5),  # right big toe
        21: (205, 205, 0.5),  # right small toe
        22: (195, 195, 0.5),  # right heel
    }
    kps = make_keypoints(23, overrides)
    result = extract_foot_keypoints(kps)
    indices = {r["index"] for r in result}
    assert indices == {17, 18, 19, 20, 21, 22}
    assert 15 not in indices
    assert 16 not in indices


# ── Ankle keypoint extraction tests ─────────────────────────────────────────

def test_ankle_below_score_threshold_skipped():
    """Ankle with confidence 0.3 is skipped for scoring (< 0.5)."""
    kps = make_keypoints(17, {15: (100, 200, 0.3)})
    result = extract_ankle_keypoints(kps)
    assert len(result) == 0


def test_ankle_above_threshold_extracted():
    """Ankle with confidence 0.6 is extracted."""
    kps = make_keypoints(17, {15: (100, 200, 0.6)})
    result = extract_ankle_keypoints(kps)
    assert len(result) == 1
    assert result[0]["side"] == "left"


# ── check_player_in_kitchen integration tests ───────────────────────────────

def test_any_foot_kp_in_zone_triggers_fault(identity_homography):
    """Big toe in zone but heel outside -> fault (ANY triggers)."""
    overrides = {
        17: (10, 3, 0.5),   # left big toe IN left kitchen
        19: (10, 10, 0.5),  # left heel OUTSIDE kitchen
    }
    kps = make_keypoints(23, overrides)
    hits = check_player_in_kitchen(kps, identity_homography)
    assert len(hits) >= 1
    assert any(h["zone"] == "left" for h in hits)


def test_all_foot_kps_outside_no_fault(identity_homography):
    """All 6 foot keypoints outside kitchen -> no fault."""
    overrides = {
        17: (10, 10, 0.5),
        18: (10, 10, 0.5),
        19: (10, 10, 0.5),
        20: (10, 10, 0.5),
        21: (10, 10, 0.5),
        22: (10, 10, 0.5),
    }
    kps = make_keypoints(23, overrides)
    hits = check_player_in_kitchen(kps, identity_homography)
    assert len(hits) == 0


def test_coco17_fallback_uses_buffer(identity_homography):
    """COCO 17 model uses ankle + buffer for zone check."""
    # Ankle at (10, 7.3) is outside kitchen (max 7.0) but within buffer (0.5)
    kps = make_keypoints(17, {15: (10, 7.3, 0.6)})
    hits = check_player_in_kitchen(kps, identity_homography)
    assert len(hits) == 1
    assert hits[0]["source"] == "ankle"


# ── Ankle confidence for scoring ─────────────────────────────────────────────

def test_get_ankle_confidence_both():
    """Returns max of both ankles."""
    kps = make_keypoints(23, {15: (0, 0, 0.7), 16: (0, 0, 0.9)})
    assert get_ankle_confidence(kps) == pytest.approx(0.9)


def test_get_ankle_confidence_below_threshold():
    """Returns 0.0 if both ankles below threshold."""
    kps = make_keypoints(17, {15: (0, 0, 0.3), 16: (0, 0, 0.4)})
    assert get_ankle_confidence(kps) == 0.0
