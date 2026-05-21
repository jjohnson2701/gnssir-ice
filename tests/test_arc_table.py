# ABOUTME: Tests for the arc-table builder (gnssir + subdaily merge).

import numpy as np
import pandas as pd
import pytest

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.arc_table import build_arc_table
from tests.conftest import config_dict

YEAR, DOY = 2024, 200


def _make_arcs(n=80):
    """Synthetic per-arc retrievals shared by the gnssir + subdaily files."""
    rng = np.random.RandomState(0)
    rows = []
    for i in range(n):
        rows.append({
            "year": YEAR, "doy": DOY,
            "RH": round(4.0 + rng.rand(), 3),
            "sat": int(1 + i % 30),
            "UTCtime": round(0.1 + i * 0.2, 3),
            "Azim": round(rng.rand() * 360, 2),
            "Amp": round(10 + rng.rand() * 5, 2),
            "eminO": 5.0, "emaxO": 25.0, "NumbOf": 200,
            "freq": [1, 20, 5][i % 3], "rise": 1 if i % 2 else -1,
            "EdotF": 0.001, "PkNoise": round(3 + rng.rand(), 2),
            "DelT": 15.0, "MJD": round(60000.0 + i * 0.011, 6),
            "refr_model": 1,
        })
    return rows


_GNSSIR_COLS = ["year", "doy", "RH", "sat", "UTCtime", "Azim", "Amp", "eminO",
                "emaxO", "NumbOf", "freq", "rise", "EdotF", "PkNoise", "DelT",
                "MJD", "refr_model"]


def _write_gnssir_txt(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("% synthetic gnssir output\n")
        f.write("% year doy RH sat ...\n")
        for r in rows:
            f.write(" ".join(str(r[c]) for c in _GNSSIR_COLS) + "\n")


def _write_withrhdotif(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("% synthetic subdaily withrhdotIF output\n")
        for r in rows:
            base = [r[c] for c in _GNSSIR_COLS]
            # cols 17-21 (month day hour minute sec), 22-24 corrected RHs
            tail = [7, 18, 12, 30, 0,
                    round(r["RH"] - 0.03, 4),   # rh_rhdot_corrected
                    -0.03,                       # rhdot_correction
                    round(r["RH"] - 0.05, 4)]    # rh_if_corrected
            f.write(" ".join(str(v) for v in base + tail) + "\n")


def _setup(tmp_path, n_subdaily):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    resolver = PathResolver(cfg)
    rows = _make_arcs()
    _write_gnssir_txt(resolver.gnssir_results_dir(YEAR) / f"{DOY:03d}.txt", rows)
    if n_subdaily is not None:
        _write_withrhdotif(resolver.subdaily_file(YEAR), rows[:n_subdaily])
    return cfg, resolver, rows


def test_build_arc_table_merges_corrected_rh(tmp_path):
    cfg, resolver, rows = _setup(tmp_path, n_subdaily=70)
    out = build_arc_table(cfg, YEAR)
    arc = pd.read_parquet(out)

    assert len(arc) == len(rows)
    assert "RH_raw" in arc.columns
    # 70 arcs corrected, 10 edited out.
    assert int(arc["subdaily_qc_pass"].sum()) == 70
    corrected = arc[arc["subdaily_qc_pass"]]
    np.testing.assert_allclose(
        corrected["RH"].to_numpy(),
        (corrected["RH_raw"] - 0.05).to_numpy(), atol=1e-3)
    # Edited-out arcs keep the raw RH.
    dropped = arc[~arc["subdaily_qc_pass"]]
    np.testing.assert_allclose(dropped["RH"].to_numpy(),
                               dropped["RH_raw"].to_numpy())


def test_build_arc_table_adds_derived_columns(tmp_path):
    cfg, resolver, rows = _setup(tmp_path, n_subdaily=80)
    arc = pd.read_parquet(build_arc_table(cfg, YEAR))
    for col in ("date", "azimuth_bin", "wse"):
        assert col in arc.columns


def test_missing_subdaily_file_raises(tmp_path):
    cfg, resolver, rows = _setup(tmp_path, n_subdaily=None)
    with pytest.raises(FileNotFoundError, match="subdaily"):
        build_arc_table(cfg, YEAR)


def test_low_subdaily_match_rate_raises(tmp_path):
    # Only 20 of 80 arcs corrected (25%) — below the 50% match-rate floor.
    cfg, resolver, rows = _setup(tmp_path, n_subdaily=20)
    with pytest.raises(RuntimeError, match="subdaily"):
        build_arc_table(cfg, YEAR)
