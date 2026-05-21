# ABOUTME: Tests for the optional plot command (skips without matplotlib).

import pickle

import numpy as np
import pandas as pd
import pytest

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import MAHAL_FEATURES
from gnssir_ice.plot import (_load_validation, _sampled_month_spans,
                             _season_windows)
from tests.conftest import config_dict


def _write_mahal(resolver, n=160, start="2024-01-01"):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "date": pd.date_range(start, periods=n),
        "year": 2024,
        "mahal_d": np.abs(rng.normal(3, 1, n)),
        "above_threshold": rng.random(n) > 0.7,
        "n_features_used": 10,
    })
    for f in MAHAL_FEATURES:
        df[f"{f}_z"] = rng.normal(0, 1, n)
    resolver.ensure_output_dir()
    df.to_parquet(resolver.daily_mahal_d(), index=False)
    # minimal baseline so the threshold line + baseline band are drawn
    with open(resolver.baseline_pkl(), "wb") as fh:
        pickle.dump({"threshold": 4.0, "baseline_mahal_d": {"p95": 3.0}}, fh)
    return df


def test_season_windows_splits_by_ice_season():
    dates = pd.Series(pd.date_range("2021-11-15", "2023-03-01"))
    windows = _season_windows(dates)
    assert [w[0] for w in windows] == [2022, 2023]   # two Nov-Jun seasons


def test_load_validation_requires_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,foo\n2024-01-01,1\n")
    with pytest.raises(ValueError, match="ice_fraction"):
        _load_validation(bad)


def test_sampled_month_spans_marks_open_water_months():
    start, end = pd.Timestamp(2021, 7, 1), pd.Timestamp(2022, 6, 30)
    spans = _sampled_month_spans(start, end, months=[7, 8], years=[2021, 2022])
    assert len(spans) == 2                       # Jul + Aug 2021
    assert all(s.year == 2021 for s, _ in spans)


def test_sampled_month_spans_skips_non_open_water_years():
    start, end = pd.Timestamp(2021, 7, 1), pd.Timestamp(2022, 6, 30)
    assert _sampled_month_spans(start, end, [7, 8], [2020, 2023]) == []


def test_plot_station_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from gnssir_ice.plot import plot_station

    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    _write_mahal(resolver)
    out = plot_station(cfg)
    assert out.exists() and out.stat().st_size > 0


def test_plot_station_with_validation_overlay(tmp_path):
    pytest.importorskip("matplotlib")
    from gnssir_ice.plot import plot_station

    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    _write_mahal(resolver)
    val = tmp_path / "validation.csv"
    pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=160),
        "ice_fraction": np.linspace(1.0, 0.0, 160),
    }).to_csv(val, index=False)
    out = plot_station(cfg, validation=str(val))
    assert out.exists() and out.stat().st_size > 0


def test_plot_station_features_writes_separate_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from gnssir_ice.plot import plot_station

    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    _write_mahal(resolver)
    out = plot_station(cfg, features=True)
    assert out.exists() and out.stat().st_size > 0
    feat = out.with_name(f"{out.stem}_features{out.suffix}")
    assert feat.exists() and feat.stat().st_size > 0
