
import pytest

from tracking_logic import (
    target_matches,
    should_stop_search,
    clamp_roi,
)


@pytest.mark.parametrize(
    "target, detected, expected",
    [
        ("ABC123", "ABC123", True),
        ("ABC123", "ID: ABC123", True),
        ("ABC123", "ABC12", False),
        ("ABC123", "XYZ999", False),
        ("", "ABC123", False),
    ],
)
def test_target_matches(target, detected, expected):
    assert target_matches(target, detected) is expected


@pytest.mark.parametrize(
    "lost_count, patience, expected",
    [
        (0, 30, False),
        (29, 30, False),
        (30, 30, True),
        (31, 30, True),
    ],
)
def test_should_stop_search(lost_count, patience, expected):
    assert should_stop_search(lost_count, patience) is expected


@pytest.mark.parametrize(
    "roi, expected",
    [
        ((10, 20, 100, 80), (10, 20, 90, 80)),
        ((-10, 20, 100, 80), (0, 20, 90, 80)),
        ((90, 90, 30, 30), (90, 90, 10, 10)),
        ((200, 200, 10, 10), None),
    ],
)
def test_clamp_roi(roi, expected):
    result = clamp_roi(*roi, frame_width=100, frame_height=100)
    assert result == expected
