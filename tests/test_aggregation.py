# ABOUTME: Tests for Stage 4 daily aggregation — the 10-feature schema.

import numpy as np
import pandas as pd

from gnssir_ice.aggregate import aggregate_station_year
from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import MAHAL_FEATURES
from tests.conftest import config_dict

YEAR = 2024


def _synthetic_arc_table(n_days=30, arcs_per_day=60, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for d in range(n_days):
        doy = 180 + d
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1)
        for a in range(arcs_per_day):
            rows.append({
                "year": YEAR, "doy": doy, "date": date.strftime("%Y-%m-%d"),
                "sat": int(1 + a % 25), "freq": [1, 20, 5][a % 3],
                "azimuth_bin": (a % 4) * 90,
                "RH": 4.0 + rng.normal(0, 0.1),
                "Amp": 12.0 + rng.normal(0, 1),
                "PkNoise": 3.5 + rng.normal(0, 0.4),
                "CLR": 8.0 + rng.normal(0, 1),
                "PR": 2.0 + rng.normal(0, 0.3),
                "AF": 1.5 + rng.normal(0, 0.2),
                "gamma": 0.004 + abs(rng.normal(0, 0.001)),
                "gamma_r2": 0.8 + rng.rand() * 0.2,
                "MS": 45.0 + rng.normal(0, 2),
                "VS": 30.0 + rng.normal(0, 3),
                "full_arc": True,
            })
    return pd.DataFrame(rows)


def test_aggregate_produces_10_features(tmp_path):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    resolver.ensure_output_dir()
    _synthetic_arc_table().to_parquet(resolver.arc_head(YEAR), index=False)

    out = aggregate_station_year(cfg, YEAR)
    daily = pd.read_parquet(out)

    pooled = daily[daily["azimuth_bin"] == -1]
    assert len(pooled) == 30
    for feat in MAHAL_FEATURES:
        assert feat in pooled.columns, feat
        assert pooled[feat].notna().all(), feat


def test_rh_std_norm_falls_back_without_norm_column(tmp_path):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    resolver.ensure_output_dir()
    _synthetic_arc_table().to_parquet(resolver.arc_head(YEAR), index=False)

    daily = pd.read_parquet(aggregate_station_year(cfg, YEAR))
    pooled = daily[daily["azimuth_bin"] == -1]
    # No RH_norm column → rh_std_norm duplicates rh_std_raw.
    np.testing.assert_allclose(pooled["rh_std_norm"], pooled["rh_std_raw"])


def test_pooled_and_sector_rows_present(tmp_path):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    resolver.ensure_output_dir()
    _synthetic_arc_table().to_parquet(resolver.arc_head(YEAR), index=False)

    daily = pd.read_parquet(aggregate_station_year(cfg, YEAR))
    assert (daily["azimuth_bin"] == -1).sum() == 30   # pooled
    assert (daily["azimuth_bin"] >= 0).sum() > 0      # sectors
