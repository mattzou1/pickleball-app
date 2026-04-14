"""Tests for pickleball.constants module."""

from pickleball.constants import scale_frame_threshold


def test_fps_scaling_30():
    """Scale factor = 1.0 at 30fps."""
    assert scale_frame_threshold(3, 30.0) == 3
    assert scale_frame_threshold(10, 30.0) == 10


def test_fps_scaling_60():
    """Scale factor = 2.0 at 60fps, thresholds doubled."""
    assert scale_frame_threshold(3, 60.0) == 6
    assert scale_frame_threshold(10, 60.0) == 20


def test_fps_scaling_24():
    """Scale factor = 0.8 at 24fps, thresholds reduced."""
    # 3 * 24/30 = 2.4 -> round to 2
    assert scale_frame_threshold(3, 24.0) == 2
    # 10 * 24/30 = 8.0
    assert scale_frame_threshold(10, 24.0) == 8


def test_fps_scaling_minimum_1():
    """Never returns 0, minimum is 1."""
    assert scale_frame_threshold(1, 1.0) == 1
    assert scale_frame_threshold(1, 0.5) == 1
