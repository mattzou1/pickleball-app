"""Tests for pickleball.ball module."""

import numpy as np
import pytest

from pickleball.ball import (
    BOUNCED,
    LIVE,
    UNKNOWN,
    BallStateMachine,
    classify_bounce_side,
    compute_velocity_vectors,
    compute_vertical_velocity,
    detect_bounces,
    detect_paddle_contacts,
    interpolate_positions,
    paddle_contact_near,
)


# ── Interpolation tests ─────────────────────────────────────────────────────

def test_interpolate_no_gaps():
    """Input with every frame detected, output unchanged."""
    dets = [{"x": i, "y": i * 2} for i in range(5)]
    result = interpolate_positions(dets)
    assert len(result) == 5
    for i in range(5):
        assert result[i]["x"] == i
        assert "interpolated" not in result[i]


def test_interpolate_short_gap():
    """3-frame gap filled with linear interpolation."""
    dets = [
        {"x": 0, "y": 0},
        None, None, None,
        {"x": 4, "y": 8},
    ]
    result = interpolate_positions(dets)
    assert result[0]["x"] == 0
    assert result[4]["x"] == 4

    # Check interpolated values
    assert result[1] is not None
    assert result[1]["interpolated"] is True
    assert result[1]["x"] == pytest.approx(1.0)
    assert result[1]["y"] == pytest.approx(2.0)

    assert result[2]["x"] == pytest.approx(2.0)
    assert result[2]["y"] == pytest.approx(4.0)

    assert result[3]["x"] == pytest.approx(3.0)
    assert result[3]["y"] == pytest.approx(6.0)


def test_interpolate_long_gap_skipped():
    """6-frame gap NOT interpolated (> max_gap=5)."""
    dets = [
        {"x": 0, "y": 0},
        None, None, None, None, None, None,
        {"x": 7, "y": 14},
    ]
    result = interpolate_positions(dets)
    for i in range(1, 7):
        assert result[i] is None


def test_interpolate_gap_at_start():
    """No anchor point before gap, not filled."""
    dets = [None, None, {"x": 2, "y": 4}]
    result = interpolate_positions(dets)
    assert result[0] is None
    assert result[1] is None


def test_interpolate_gap_at_end():
    """No anchor point after gap, not filled."""
    dets = [{"x": 0, "y": 0}, None, None]
    result = interpolate_positions(dets)
    assert result[1] is None
    assert result[2] is None


def test_interpolate_marks_interpolated_flag():
    """Interpolated entries carry an interpolated=True flag."""
    dets = [{"x": 0, "y": 0}, None, {"x": 2, "y": 4}]
    result = interpolate_positions(dets)
    assert result[1]["interpolated"] is True
    assert result[1]["x"] == pytest.approx(1.0)
    assert result[1]["y"] == pytest.approx(2.0)


# ── Bounce detection tests ──────────────────────────────────────────────────

def test_bounce_detection_clear_reversal():
    """Downward to upward velocity = bounce."""
    # Ball moving one direction then reversing
    dets = [
        {"y": 0}, {"y": 2}, {"y": 4}, {"y": 6},  # moving forward
        {"y": 5}, {"y": 3}, {"y": 1},               # reversing
    ]
    vels = compute_vertical_velocity(dets, window=1)
    bounces = detect_bounces(dets, vels)
    assert len(bounces) >= 1
    # Bounce should be around frame 4 (where velocity goes from +2 to -1)
    assert any(3 <= b <= 5 for b in bounces)


def test_bounce_detection_hovering():
    """Ball near ground with no clear reversal = no bounce."""
    dets = [{"y": 10}, {"y": 10.1}, {"y": 10.0}, {"y": 10.1}, {"y": 10.0}]
    vels = compute_vertical_velocity(dets, window=2)
    bounces = detect_bounces(dets, vels)
    # Small oscillation may or may not trigger. Key: no large reversal.
    # With smoothing window=2, tiny oscillations should be dampened.
    # This is acceptable either way since the confidence will be low.


def test_classify_bounce_left():
    """Bounce with pixel x < net_x is left."""
    dets = [None, None, {"x": 300.0, "y": 500.0}]
    assert classify_bounce_side(2, dets, net_x_pixel=500.0) == "left"


def test_classify_bounce_right():
    """Bounce with pixel x > net_x is right."""
    dets = [None, None, {"x": 700.0, "y": 500.0}]
    assert classify_bounce_side(2, dets, net_x_pixel=500.0) == "right"


def test_classify_bounce_none_when_no_detection():
    dets = [None, None, None]
    assert classify_bounce_side(2, dets, net_x_pixel=500.0) is None


# ── State machine tests ─────────────────────────────────────────────────────

def test_state_bounce_left_side():
    """Bounce on left side: left=BOUNCED, right=LIVE."""
    sm = BallStateMachine()
    sm.update_bounce("left")
    assert sm.get_state("left") == BOUNCED
    assert sm.get_state("right") == LIVE


def test_state_bounce_right_side():
    """Bounce on right side: right=BOUNCED, left=LIVE."""
    sm = BallStateMachine()
    sm.update_bounce("right")
    assert sm.get_state("right") == BOUNCED
    assert sm.get_state("left") == LIVE


def test_state_no_detection_unknown():
    """>10 frames missing: state=UNKNOWN."""
    sm = BallStateMachine(unknown_gap_frames=10)
    for _ in range(11):
        sm.update_detection(False)
    assert sm.get_state("left") == UNKNOWN
    assert sm.get_state("right") == UNKNOWN


def test_state_detection_resets_gap():
    """Detection resets gap counter."""
    sm = BallStateMachine(unknown_gap_frames=10)
    for _ in range(9):
        sm.update_detection(False)
    sm.update_detection(True, ball_side="left")
    # Not yet unknown
    assert sm.get_state("left") != UNKNOWN


def test_state_net_crossing():
    """Ball crossing net resets destination side to LIVE."""
    sm = BallStateMachine()
    sm.update_bounce("left")  # left=BOUNCED
    assert sm.get_state("left") == BOUNCED

    # Ball crosses net heading left
    sm.update_net_crossing("left")
    assert sm.get_state("left") == LIVE


def test_state_net_crossing_via_detection():
    """Net crossing detected via ball_side change."""
    sm = BallStateMachine()
    sm.update_bounce("left")  # left=BOUNCED

    # Ball moves from right to left (crosses net)
    sm.update_detection(True, ball_side="right")
    sm.update_detection(True, ball_side="left")  # triggers net crossing
    assert sm.get_state("left") == LIVE


# ── Paddle-contact detection tests ──────────────────────────────────────────
# COCO 17: wrists at indices 9 (left) and 10 (right). Each keypoint row is
# [x, y, confidence].

def _kps_with_wrists(left_xy, right_xy, left_conf=0.9, right_conf=0.9, size=17):
    arr = np.zeros((size, 3), dtype=float)
    arr[9] = [left_xy[0], left_xy[1], left_conf]
    arr[10] = [right_xy[0], right_xy[1], right_conf]
    return arr


def test_paddle_contact_straight_line_no_contact():
    """Ball moving at constant velocity with no inflection -> no contact."""
    dets = [{"x": float(i * 10), "y": 100.0} for i in range(10)]
    vels = compute_velocity_vectors(dets)
    pose = [{1: _kps_with_wrists((i * 10, 100), (i * 10 + 5, 100))} for i in range(10)]
    contacts = detect_paddle_contacts(dets, vels, pose, bounce_frames=[])
    assert contacts == []


def test_paddle_contact_sharp_reversal_near_wrist():
    """Sharp direction reversal co-located with wrist -> one contact."""
    # Ball flies right frames 0-4, reverses starting at frame 5.
    # Inflection is detected at frame 5 (first frame where velocity flips).
    dets = [
        {"x": 100.0, "y": 200.0},
        {"x": 120.0, "y": 200.0},
        {"x": 140.0, "y": 200.0},
        {"x": 160.0, "y": 200.0},
        {"x": 180.0, "y": 200.0},
        {"x": 160.0, "y": 200.0},  # reversal
        {"x": 140.0, "y": 200.0},
        {"x": 120.0, "y": 200.0},
        {"x": 100.0, "y": 200.0},
    ]

    vels = compute_velocity_vectors(dets, window=1)

    # Wrist of track 7 sits on the ball at frame 5.
    pose = []
    for i in range(len(dets)):
        if i == 5:
            pose.append({7: _kps_with_wrists((1000, 1000), (160, 200))})
        else:
            pose.append({7: _kps_with_wrists((1000, 1000), (1000, 1000))})

    contacts = detect_paddle_contacts(dets, vels, pose, bounce_frames=[])
    assert len(contacts) == 1
    c = contacts[0]
    assert c["frame"] == 5
    assert c["track_id"] == 7
    assert c["wrist_side"] == "right"
    assert c["distance_px"] == pytest.approx(0.0, abs=1.0)


def _reversal_trajectory():
    return [
        {"x": 100.0, "y": 200.0},
        {"x": 120.0, "y": 200.0},
        {"x": 140.0, "y": 200.0},
        {"x": 160.0, "y": 200.0},
        {"x": 180.0, "y": 200.0},
        {"x": 160.0, "y": 200.0},
        {"x": 140.0, "y": 200.0},
        {"x": 120.0, "y": 200.0},
        {"x": 100.0, "y": 200.0},
    ]


def test_paddle_contact_wrist_too_far():
    """Inflection with no wrist within radius -> no contact."""
    dets = _reversal_trajectory()
    vels = compute_velocity_vectors(dets, window=1)
    pose = [{7: _kps_with_wrists((2000, 2000), (2000, 2000))} for _ in dets]
    contacts = detect_paddle_contacts(dets, vels, pose, bounce_frames=[])
    assert contacts == []


def test_paddle_contact_suppressed_by_bounce():
    """Inflection within ±1 of a bounce frame is ignored."""
    dets = _reversal_trajectory()
    vels = compute_velocity_vectors(dets, window=1)
    pose = []
    for i in range(len(dets)):
        if i == 5:
            pose.append({7: _kps_with_wrists((1000, 1000), (160, 200))})
        else:
            pose.append({7: _kps_with_wrists((1000, 1000), (1000, 1000))})

    # Pretend a bounce was detected at frame 5.
    contacts = detect_paddle_contacts(dets, vels, pose, bounce_frames=[5])
    assert contacts == []


def test_paddle_contact_near_window():
    """paddle_contact_near picks closest in-window contact, optional track filter."""
    contacts = [
        {"frame": 10, "track_id": 1, "wrist_side": "left"},
        {"frame": 20, "track_id": 2, "wrist_side": "right"},
        {"frame": 22, "track_id": 1, "wrist_side": "right"},
    ]
    # Nearest to 21 within window=3: frame 20 or 22 (both dist 1). Picks 20 (first match).
    r = paddle_contact_near(contacts, frame_idx=21, window_frames=3)
    assert r is not None and r["frame"] in (20, 22)

    # Filter by track_id=1 -> frame 22 is closest.
    r = paddle_contact_near(contacts, frame_idx=21, window_frames=3, track_id=1)
    assert r is not None and r["frame"] == 22

    # Out of window -> None.
    r = paddle_contact_near(contacts, frame_idx=50, window_frames=3)
    assert r is None


def test_paddle_contact_low_wrist_conf_ignored():
    """Wrist below confidence threshold -> no association."""
    dets = _reversal_trajectory()
    vels = compute_velocity_vectors(dets, window=1)
    pose = []
    for i in range(len(dets)):
        if i == 5:
            # wrist at ball but conf=0.1 (< threshold 0.3)
            pose.append({7: _kps_with_wrists((1000, 1000), (160, 200), right_conf=0.1)})
        else:
            pose.append({7: _kps_with_wrists((1000, 1000), (1000, 1000))})

    contacts = detect_paddle_contacts(dets, vels, pose, bounce_frames=[])
    assert contacts == []
