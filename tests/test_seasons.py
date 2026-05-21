# ABOUTME: Tests for Stage 11 — ice-season run detection.

import pandas as pd

from gnssir_ice.seasons import _seasons_for


def _series(above_flags):
    """A minimal scored series with the given above-threshold flags."""
    n = len(above_flags)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "mahal_d": [5.0 if a else 1.0 for a in above_flags],
        "above_threshold": above_flags,
    })


def test_detects_one_sustained_season():
    flags = [False] * 3 + [True] * 6 + [False] * 3
    seasons = _seasons_for(_series(flags), sector=-1, min_run_days=5)
    assert len(seasons) == 1
    s = seasons[0]
    assert s["n_days_above"] == 6
    assert s["duration_days"] == 6
    assert s["sector"] == -1


def test_ignores_runs_below_min_length():
    flags = [False] * 5 + [True] * 3 + [False] * 5     # 3-day run < min 5
    assert _seasons_for(_series(flags), -1, 5) == []


def test_detects_multiple_seasons():
    flags = [True] * 6 + [False] * 4 + [True] * 7
    seasons = _seasons_for(_series(flags), -1, 5)
    assert [s["n_days_above"] for s in seasons] == [6, 7]


def test_no_seasons_when_all_open_water():
    assert _seasons_for(_series([False] * 30), -1, 5) == []
