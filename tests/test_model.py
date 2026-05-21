# ABOUTME: Tests for Stages 9/10 — Mahalanobis baseline fit and scoring.

import numpy as np
import pandas as pd
import pytest

from gnssir_ice.constants import MAHAL_FEATURES
from gnssir_ice.model import (MIN_BASELINE_DAYS, apply_baseline,
                              apply_baseline_sectors, fit_baseline)


def _synthetic_daily(n=150, seed=0, shift=0.0):
    """n days of the 10-feature schema, optionally shifted off-baseline."""
    rng = np.random.RandomState(seed)
    data = {f: rng.normal(0, 1, n) + shift for f in MAHAL_FEATURES}
    df = pd.DataFrame(data)
    df["date"] = pd.date_range("2024-07-01", periods=n)
    df["year"] = 2024
    return df


def test_fit_baseline_returns_model():
    model = fit_baseline(_synthetic_daily(), pca_variance=0.95)
    assert 1 <= model["n_components"] <= len(MAHAL_FEATURES)
    assert model["total_variance_explained"] >= 0.95 - 1e-9
    assert model["threshold"] > 0
    assert model["retained_features"] == MAHAL_FEATURES


def test_fit_baseline_too_few_days_raises():
    with pytest.raises(ValueError, match="baseline days"):
        fit_baseline(_synthetic_daily(n=MIN_BASELINE_DAYS - 1))


def test_apply_baseline_self_consistency():
    df = _synthetic_daily()
    model = fit_baseline(df)
    scored = apply_baseline(df, model)
    assert len(scored) == len(df)
    # On the baseline data itself the median distance is modest.
    assert scored["mahal_d"].median() < model["threshold"]
    assert scored["mahal_d"].min() >= 0


def test_apply_baseline_flags_shifted_days():
    baseline_df = _synthetic_daily(seed=0)
    model = fit_baseline(baseline_df)
    # A strongly shifted population should mostly exceed the threshold.
    shifted = apply_baseline(_synthetic_daily(n=60, seed=1, shift=6.0), model)
    assert shifted["above_threshold"].mean() > 0.8


def test_apply_baseline_schema():
    df = _synthetic_daily()
    scored = apply_baseline(df, fit_baseline(df))
    for col in ("date", "year", "mahal_d", "above_threshold", "n_features_used"):
        assert col in scored.columns
    for feat in MAHAL_FEATURES:
        assert f"{feat}_z" in scored.columns


def _synthetic_sectors(n_per_sector=40, sectors=(0, 90, 180), seed=2):
    """Per-sector daily rows (10-feature schema + azimuth_bin)."""
    rng = np.random.RandomState(seed)
    frames = []
    for s in sectors:
        data = {f: rng.normal(0, 1, n_per_sector) for f in MAHAL_FEATURES}
        df = pd.DataFrame(data)
        df["azimuth_bin"] = s
        df["date"] = pd.date_range("2024-07-01", periods=n_per_sector)
        df["year"] = 2024
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_fit_baseline_per_sector_means():
    model = fit_baseline(_synthetic_daily(), sector_frame=_synthetic_sectors(40),
                         min_sector_days=20)
    assert set(model["sector_means"]) == {0, 90, 180}
    for center in model["sector_means"].values():
        assert len(center) == model["n_components"]


def test_fit_baseline_skips_thin_sectors():
    # 10 days/sector is below min_sector_days → no sector baselines.
    model = fit_baseline(_synthetic_daily(), sector_frame=_synthetic_sectors(10),
                         min_sector_days=20)
    assert model["sector_means"] == {}


def test_apply_baseline_sectors_scores_each_sector():
    sectors = _synthetic_sectors(40)
    model = fit_baseline(_synthetic_daily(), sector_frame=sectors,
                         min_sector_days=20)
    scored = apply_baseline_sectors(sectors, model)
    assert not scored.empty
    assert set(scored["azimuth_bin"].unique()) == {0, 90, 180}
    assert "mahal_d" in scored.columns and (scored["mahal_d"] >= 0).all()


def test_apply_baseline_attributes_dropped_days():
    df = _synthetic_daily()
    df.loc[:4, "af_med"] = np.nan                 # 5 days lose one feature
    model = fit_baseline(_synthetic_daily(seed=9))
    scored, dropped = apply_baseline(df, model, return_dropped=True)
    assert len(dropped) == 5
    assert (dropped["nan_features"] == "af_med").all()
    assert len(scored) == len(df) - 5
