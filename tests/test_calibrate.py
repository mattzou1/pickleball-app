"""Tests for calibrate.py pure logic (polygon sanity checks)."""

from calibrate import validate_polygon


def test_valid_convex_quad():
    poly = [[100, 100], [300, 100], [300, 300], [100, 300]]
    assert validate_polygon(poly) is None


def test_rejects_duplicate_points():
    poly = [[100, 100], [105, 102], [300, 300], [100, 300]]
    err = validate_polygon(poly)
    assert err is not None
    assert "too close" in err


def test_rejects_non_convex():
    # Butterfly / self-intersecting quad (points out of order)
    poly = [[100, 100], [300, 300], [300, 100], [100, 300]]
    err = validate_polygon(poly)
    assert err is not None
    assert "convex" in err
