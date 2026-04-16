"""Tests for pickleball.pose module."""

import numpy as np
import pytest

from pickleball.pose import (
    check_player_in_kitchen,
    extract_ankle_keypoints,
    extract_foot_keypoints,
    get_ankle_confidence,
    get_pose_confidence,
    is_wholebody_model,
    point_in_kitchen,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def square_polygon():
    """A 100x100 square polygon from (100,100) to (200,200)."""
    return np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.int32)


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


# ── point_in_kitchen tests ──────────────────────────────────────────────────

def test_point_inside(square_polygon):
    assert point_in_kitchen(150, 150, square_polygon) is True


def test_point_outside(square_polygon):
    assert point_in_kitchen(50, 50, square_polygon) is False


def test_point_on_boundary(square_polygon):
    """Edge counts as inside (pointPolygonTest returns 0 on boundary)."""
    assert point_in_kitchen(100, 150, square_polygon) is True
    assert point_in_kitchen(200, 150, square_polygon) is True


def test_point_on_corner(square_polygon):
    assert point_in_kitchen(100, 100, square_polygon) is True


def test_point_just_outside_within_tolerance(square_polygon):
    """A point 3 px outside the left edge counts as inside under the default tolerance."""
    assert point_in_kitchen(97, 150, square_polygon) is True


def test_point_outside_beyond_tolerance(square_polygon):
    """A point 10 px outside the left edge exceeds the default ~5 px tolerance."""
    assert point_in_kitchen(90, 150, square_polygon) is False


def test_explicit_tolerance_override(square_polygon):
    """Passing tolerance_px=0.0 restores strict-boundary behavior."""
    assert point_in_kitchen(97, 150, square_polygon, tolerance_px=0.0) is False


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
        15: (50, 50, 0.9),
        16: (60, 60, 0.9),
        17: (100, 200, 0.5),
        18: (105, 205, 0.5),
        19: (95, 195, 0.5),
        20: (200, 200, 0.5),
        21: (205, 205, 0.5),
        22: (195, 195, 0.5),
    }
    kps = make_keypoints(23, overrides)
    result = extract_foot_keypoints(kps)
    indices = {r["index"] for r in result}
    assert indices == {17, 18, 19, 20, 21, 22}


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

def test_any_foot_kp_in_zone_triggers_fault(square_polygon):
    """Big toe in zone but heel outside -> fault (ANY triggers)."""
    overrides = {
        17: (150, 150, 0.5),   # left big toe INSIDE polygon
        19: (50, 50, 0.5),     # left heel OUTSIDE polygon
    }
    kps = make_keypoints(23, overrides)
    hits = check_player_in_kitchen(kps, square_polygon)
    assert len(hits) >= 1
    assert all(h["zone"] == "kitchen" for h in hits)
    assert hits[0]["source"] == "foot"


def test_all_foot_kps_outside_no_fault(square_polygon):
    """All 6 foot keypoints outside kitchen -> no fault."""
    overrides = {i: (50, 50, 0.5) for i in range(17, 23)}
    kps = make_keypoints(23, overrides)
    hits = check_player_in_kitchen(kps, square_polygon)
    assert len(hits) == 0


def test_coco17_uses_ankles(square_polygon):
    """COCO 17 model uses ankle keypoints for zone check."""
    kps = make_keypoints(17, {15: (150, 150, 0.6)})
    hits = check_player_in_kitchen(kps, square_polygon)
    assert len(hits) == 1
    assert hits[0]["source"] == "ankle"


def test_foot_side_tagged_by_net_x(square_polygon):
    """foot_side is derived from pixel x vs net_x_pixel."""
    # Two foot keypoints, one on each side of net_x=150.
    overrides = {
        17: (120, 150, 0.5),  # left of net
        20: (180, 150, 0.5),  # right of net
    }
    kps = make_keypoints(23, overrides)
    hits = check_player_in_kitchen(kps, square_polygon, net_x_pixel=150.0)
    sides = {h["foot_side"] for h in hits}
    assert "left" in sides
    assert "right" in sides


def test_foot_side_none_without_net(square_polygon):
    """foot_side is None when net_x_pixel not provided."""
    kps = make_keypoints(23, {17: (150, 150, 0.5)})
    hits = check_player_in_kitchen(kps, square_polygon)
    assert hits[0]["foot_side"] is None


# ── Pose confidence for scoring ──────────────────────────────────────────────

def test_get_ankle_confidence_both():
    """Returns max of both ankles."""
    kps = make_keypoints(23, {15: (0, 0, 0.7), 16: (0, 0, 0.9)})
    assert get_ankle_confidence(kps) == pytest.approx(0.9)


def test_get_ankle_confidence_below_threshold():
    """Returns 0.0 if both ankles below threshold."""
    kps = make_keypoints(17, {15: (0, 0, 0.3), 16: (0, 0, 0.4)})
    assert get_ankle_confidence(kps) == 0.0


def test_pose_confidence_prefers_ankle():
    """Ankle conf ≥ threshold wins over foot-kp fallback."""
    kps = make_keypoints(23, {15: (0, 0, 0.8), 17: (0, 0, 0.9)})
    assert get_pose_confidence(kps) == pytest.approx(0.8)


def test_pose_confidence_foot_fallback_when_ankle_occluded():
    """WholeBody: when ankle < threshold, fall back to best foot-kp conf."""
    kps = make_keypoints(23, {15: (0, 0, 0.3), 16: (0, 0, 0.4), 17: (0, 0, 0.6), 20: (0, 0, 0.55)})
    conf = get_pose_confidence(kps)
    assert conf == pytest.approx(0.6)


def test_pose_confidence_zero_when_nothing_reliable():
    kps = make_keypoints(23, {17: (0, 0, 0.1)})  # below foot threshold
    assert get_pose_confidence(kps) == 0.0


def test_pose_confidence_coco17_no_foot_fallback():
    """COCO 17 model has no foot keypoints — falls back to 0.0 when ankles are low."""
    kps = make_keypoints(17, {15: (0, 0, 0.3)})
    assert get_pose_confidence(kps) == 0.0
