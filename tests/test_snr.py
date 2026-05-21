# ABOUTME: Tests for Stage 1 SNR consolidation — including short-file padding.

import numpy as np
import pandas as pd
import pytest

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import SNR66_COLUMNS
from gnssir_ice.snr import consolidate_snr
from tests.conftest import config_dict

YEAR = 2024


def _write_snr(resolver, doy, n_cols, n_rows=40, seed=0):
    """Write a synthetic snr66 file with ``n_cols`` columns for one day."""
    rng = np.random.RandomState(seed)
    rows = rng.uniform(1, 45, size=(n_rows, n_cols))
    rows[:, 0] = rng.randint(1, 32, n_rows)          # sat
    path = resolver.snr_file(YEAR, doy)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%.4f")


def test_consolidate_pads_short_files(tmp_path):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    _write_snr(resolver, 10, n_cols=11, seed=1)      # full snr66
    _write_snr(resolver, 11, n_cols=9, seed=2)       # short (no S7/S8)

    out = consolidate_snr(cfg, YEAR)
    df = pd.read_parquet(out)

    assert set(df["doy"]) == {10, 11}                # both days kept
    assert list(df.columns) == SNR66_COLUMNS + ["doy"]
    # The 9-column day has its two trailing signal columns zero-filled.
    short = df[df["doy"] == 11]
    assert (short["S7"] == 0).all()
    assert (short["S8"] == 0).all()


def test_consolidate_skips_malformed_files(tmp_path):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    _write_snr(resolver, 20, n_cols=11, seed=3)
    _write_snr(resolver, 21, n_cols=4, seed=4)       # too few columns

    df = pd.read_parquet(consolidate_snr(cfg, YEAR))
    assert set(df["doy"]) == {20}                    # malformed day dropped


def test_consolidate_no_files_raises(tmp_path):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    with pytest.raises(FileNotFoundError, match="no SNR files"):
        consolidate_snr(cfg, YEAR)
