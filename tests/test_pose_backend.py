"""Tests for pose_backend selection helper.

No model weights needed — exercises the pure bbox-area top-N filter.
"""

import numpy as np

from pickleball.pose_backend import _keep_largest


def _bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_keep_all_when_under_limit():
    xyxy = np.array([_bbox(10, 10, 50, 100), _bbox(20, 20, 40, 80)])
    ids = np.array([1, 2])
    out_xyxy, out_ids = _keep_largest(xyxy, ids, 4)
    assert len(out_ids) == 2
    assert set(out_ids.tolist()) == {1, 2}


def test_keep_top_four_by_area():
    # 6 bboxes: 4 large (on-court), 2 small (distant spectators/adjacent court).
    xyxy = np.array([
        _bbox(100, 100, 120, 240),  # id 1 — huge
        _bbox(200, 100, 110, 220),  # id 2 — huge
        _bbox(300, 100, 100, 200),  # id 3 — large
        _bbox(400, 100, 90, 180),   # id 4 — large
        _bbox(500, 100, 30, 60),    # id 5 — small
        _bbox(600, 100, 25, 50),    # id 6 — smallest
    ])
    ids = np.array([1, 2, 3, 4, 5, 6])
    out_xyxy, out_ids = _keep_largest(xyxy, ids, 4)
    assert len(out_ids) == 4
    assert set(out_ids.tolist()) == {1, 2, 3, 4}


def test_empty_returns_empty():
    xyxy = np.zeros((0, 4))
    ids = np.zeros((0,), dtype=int)
    out_xyxy, out_ids = _keep_largest(xyxy, ids, 4)
    assert len(out_xyxy) == 0
    assert len(out_ids) == 0


def test_regression_foreground_player_not_dropped():
    """Regression: a foreground on-court player whose feet are cropped or
    occluded (bbox bottom = mid-torso) would have been dropped by the old
    homography-based bounds filter because the bottom-center projected past
    the far baseline. Bbox area makes no such mistake — foreground player
    has a huge bbox, stays in the top N.
    """
    xyxy = np.array([
        _bbox(400, 300, 200, 400),  # id 99 — foreground player, huge bbox
        _bbox(100, 100, 40, 80),    # spectator
        _bbox(700, 100, 40, 80),    # spectator
        _bbox(50, 400, 40, 80),     # spectator
        _bbox(750, 400, 40, 80),    # spectator
    ])
    ids = np.array([99, 1, 2, 3, 4])
    out_xyxy, out_ids = _keep_largest(xyxy, ids, 4)
    assert 99 in out_ids.tolist(), "Foreground player must survive the top-N filter"


def test_adjacent_court_player_dropped_when_over_limit():
    # 4 close on-court players + 1 distant adjacent-court player → drop the distant one.
    xyxy = np.array([
        _bbox(100, 100, 150, 300),
        _bbox(300, 100, 140, 280),
        _bbox(500, 100, 130, 260),
        _bbox(700, 100, 120, 240),
        _bbox(900, 100, 30, 60),  # adjacent court, small
    ])
    ids = np.array([1, 2, 3, 4, 99])
    out_xyxy, out_ids = _keep_largest(xyxy, ids, 4)
    assert 99 not in out_ids.tolist()
    assert len(out_ids) == 4
