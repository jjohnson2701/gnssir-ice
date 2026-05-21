# ABOUTME: End-to-end smoke test — full pipeline on a synthetic station-year.

"""Drives :func:`run_pipeline` over a fully synthetic station-year written to
disk (snr66 + gnssir + subdaily files). Catches inter-stage wiring bugs that the
per-stage unit tests cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnssir_ice import StationConfig, run_pipeline
from gnssir_ice.config import PathResolver
from tests.conftest import config_dict

YEAR = 2024
DOYS = list(range(160, 235))      # 75 days, all within Jun-Sep
ARCS_PER_DAY = 12
N_SAMPLES = 120
WAVELENGTH = 0.19029

_GNSSIR_COLS = ["year", "doy", "RH", "sat", "UTCtime", "Azim", "Amp", "eminO",
                "emaxO", "NumbOf", "freq", "rise", "EdotF", "PkNoise", "DelT",
                "MJD", "refr_model"]


def _build_arc_snr(rh, gamma, phase, amp_if, seed):
    """Return (elev, sod, snr_db) for one synthetic rising L1 arc."""
    rng = np.random.RandomState(seed)
    elev = np.linspace(5.0, 26.0, N_SAMPLES)
    sod0 = (seed % ARCS_PER_DAY) * 4000.0
    sod = sod0 + np.arange(N_SAMPLES) * 15.0
    sin_e = np.sin(np.radians(elev))
    k = 2 * np.pi / WAVELENGTH
    omega = 4 * np.pi * rh / WAVELENGTH
    envelope = np.exp(-4 * k ** 2 * gamma * sin_e ** 2)
    trend = 47.0 - 0.25 * elev
    interference = amp_if * envelope * np.cos(omega * sin_e + phase)
    snr_db = trend + interference + rng.normal(0, 0.15, N_SAMPLES)
    return elev, sod, snr_db


def _write_station(tmp_path):
    cfg = StationConfig.from_dict(config_dict(
        tmp_path,
        baseline={"open_water_months": [6, 7, 8, 9],
                  "normalization_months": [6, 7, 8, 9],
                  "open_water_years": [YEAR]},
        processing={"years": [YEAR]},
    ))
    resolver = PathResolver(cfg)
    gnssir_dir = resolver.gnssir_results_dir(YEAR)
    gnssir_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    subdaily_rows = []
    for doy in DOYS:
        snr_lines = []
        gnssir_rows = []
        rh_day = 4.0 + 0.3 * np.sin(doy / 12.0)
        for i in range(ARCS_PER_DAY):
            sat = i + 1
            seed = doy * 100 + i
            rh = rh_day + rng.normal(0, 0.05)
            gamma = 0.004 + abs(rng.normal(0, 0.0015))
            phase = rng.uniform(-np.pi, np.pi)
            amp_if = 5.0 + rng.normal(0, 0.5)
            elev, sod, snr_db = _build_arc_snr(rh, gamma, phase, amp_if, seed)

            azim = (90.0 + i * 17.0) % 360.0
            for j in range(N_SAMPLES):
                # snr66: sat elev azim sod edot S6 S1 S2 S5 S7 S8
                snr_lines.append(
                    f"{sat} {elev[j]:.4f} {azim:.3f} {sod[j]:.3f} 0 "
                    f"0 {snr_db[j]:.4f} 0 0 0 0")

            window = (elev >= cfg.gnssir.e1) & (elev <= cfg.gnssir.e2)
            utctime = float(np.mean(sod[window]) / 3600.0)
            mjd = 60000.0 + doy + i * 0.01
            row = {
                "year": YEAR, "doy": doy, "RH": round(rh, 4), "sat": sat,
                "UTCtime": round(utctime, 6), "Azim": round(azim, 3),
                "Amp": round(12.0 + rng.normal(0, 1), 3),
                "eminO": 5.0, "emaxO": 26.0, "NumbOf": N_SAMPLES,
                "freq": 1, "rise": 1, "EdotF": 0.001,
                "PkNoise": round(3.5 + rng.normal(0, 0.4), 3),
                "DelT": 15.0, "MJD": round(mjd, 6), "refr_model": 1,
            }
            gnssir_rows.append(row)
            subdaily_rows.append(row)

        snr_path = resolver.snr_file(YEAR, doy)
        snr_path.parent.mkdir(parents=True, exist_ok=True)
        snr_path.write_text("\n".join(snr_lines) + "\n")

        with open(gnssir_dir / f"{doy:03d}.txt", "w") as f:
            f.write("% synthetic gnssir output\n")
            for r in gnssir_rows:
                f.write(" ".join(str(r[c]) for c in _GNSSIR_COLS) + "\n")

    sub_path = resolver.subdaily_file(YEAR)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sub_path, "w") as f:
        f.write("% synthetic subdaily withrhdotIF\n")
        for r in subdaily_rows:
            base = [r[c] for c in _GNSSIR_COLS]
            tail = [7, 18, 12, 30, 0,
                    round(r["RH"] - 0.03, 4), -0.03, round(r["RH"] - 0.05, 4)]
            f.write(" ".join(str(v) for v in base + tail) + "\n")
    return cfg


def test_full_pipeline_smoke(tmp_path):
    cfg = _write_station(tmp_path)
    result = run_pipeline(cfg, years=[YEAR])

    assert result["daily_mahal_d"].exists()
    assert result["ice_seasons"].exists()
    mahal = pd.read_parquet(result["daily_mahal_d"])
    assert len(mahal) > 0
    for col in ("date", "year", "mahal_d", "above_threshold"):
        assert col in mahal.columns
    assert mahal["mahal_d"].notna().all()
    assert (mahal["mahal_d"] >= 0).all()

    # The final arc artifact carries RHdot-corrected RH + the SNR features.
    arc = pd.read_parquet(PathResolver(cfg).arc_head(YEAR))
    assert "RH_raw" in arc.columns
    assert arc["subdaily_qc_pass"].all()
    assert arc["CLR"].notna().any()
    assert arc["AF"].notna().any()
    # Each chain artifact is written immutably by its own stage.
    for path in (PathResolver(cfg).arc_table(YEAR),
                 PathResolver(cfg).arc_features(YEAR),
                 PathResolver(cfg).arc_af(YEAR),
                 PathResolver(cfg).arc_norm(YEAR)):
        assert path.exists(), path.name

    # Phase-3 outputs: run manifest + per-sector scores.
    resolver = PathResolver(cfg)
    assert resolver.run_manifest().exists()
    assert resolver.sector_mahal_d().exists()
    sec = pd.read_parquet(resolver.sector_mahal_d())
    assert "azimuth_bin" in sec.columns and len(sec) > 0

    # Phase-4 outputs: CSV companion + ice-season table.
    assert resolver.daily_mahal_d_csv().exists()
    assert resolver.ice_seasons().exists()
