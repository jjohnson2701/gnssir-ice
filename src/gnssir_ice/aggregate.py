# ABOUTME: Stage 8 — aggregate per-arc features to daily × azimuth-sector rows.
# ABOUTME: Produces the 10-feature daily_features table consumed by the model.

"""Stage 8 — daily feature aggregation.

Collapses the per-arc table to one row per (date, azimuth sector), plus a
pooled row (``azimuth_bin = -1``) per day. The pooled rows carry the 10-feature
model schema (:data:`gnssir_ice.constants.MAHAL_FEATURES`).

When the per-(PRN, signal) ``*_norm`` columns are present (Stage 7), the daily
aggregates are computed on them; otherwise the raw columns are used and
``rh_std_norm`` falls back to ``rh_std_raw``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import GAMMA_R2_MIN
from gnssir_ice.provenance import write_parquet

logger = logging.getLogger(__name__)

# Per-arc feature → daily-median feature name (SNR features).
_MEDIAN_FEATURES = {"CLR": "clr_med", "PR": "pr_med", "AF": "af_med",
                    "gamma": "gamma_med", "SP": "sp_med"}


def _aggregate_sector(arcs: pd.DataFrame) -> dict:
    """Aggregate one (date, azimuth_bin) group into a feature row."""
    row: dict = {}
    n = len(arcs)
    row["n_arcs"] = n
    row["n_sats"] = int(arcs["sat"].nunique())

    if "full_arc" in arcs.columns:
        fa = arcs["full_arc"]
        row["n_full_arcs"] = int(fa.sum())
        row["frac_full_arc"] = float(fa.mean())

    if "gamma_r2" in arcs.columns:
        gr2 = arcs["gamma_r2"].dropna()
        row["gamma_r2_med"] = float(gr2.median()) if len(gr2) else np.nan

    # LSP-RH vs Strandberg-fit-RH agreement — a γ-fit convergence QC diagnostic.
    if "gamma_h_fit" in arcs.columns and "RH_snr" in arcs.columns:
        d = (arcs["gamma_h_fit"] - arcs["RH_snr"]).abs().dropna()
        row["rh_lsp_strandberg_med"] = float(d.median()) if len(d) else np.nan

    def _pick(raw_col: str) -> str:
        norm = f"{raw_col}_norm"
        return norm if norm in arcs.columns else raw_col

    # --- reflector-height spread -----------------------------------------
    rh = arcs["RH"]
    row["rh_mean"] = float(rh.mean())
    row["rh_median"] = float(rh.median())
    row["rh_range"] = float(rh.max() - rh.min())
    rh_std_raw = float(rh.std()) if len(rh.dropna()) >= 2 else np.nan
    row["rh_std_raw"] = rh_std_raw
    if "RH_norm" in arcs.columns:
        rhn = arcs["RH_norm"].dropna()
        row["rh_std_norm"] = float(rhn.std()) if len(rhn) >= 2 else np.nan
    else:
        # No Phase-2B normalization: rh_std_norm duplicates rh_std_raw and PCA
        # collapses the redundant axis.
        row["rh_std_norm"] = rh_std_raw

    # --- amplitude / peak-to-noise / SNR power ---------------------------
    amp = arcs[_pick("Amp")].dropna()
    row["amp_mean"] = float(amp.mean()) if len(amp) else np.nan
    if "PkNoise" in arcs.columns:
        pk = arcs[_pick("PkNoise")].dropna()
        row["p2n_mean"] = float(pk.mean()) if len(pk) else np.nan
    if "MS" in arcs.columns:
        ms = arcs[_pick("MS")].dropna()
        row["ms_mean"] = float(ms.mean()) if len(ms) else np.nan
    if "VS" in arcs.columns:
        vs = arcs[_pick("VS")].dropna()
        row["vs_mean"] = float(vs.mean()) if len(vs) else np.nan

    # --- SNR feature daily medians ---------------------------------------
    for raw_col, out_col in _MEDIAN_FEATURES.items():
        if raw_col not in arcs.columns:
            continue
        if raw_col == "gamma" and "gamma_r2" in arcs.columns:
            mask = arcs["gamma_r2"] >= GAMMA_R2_MIN
        else:
            mask = pd.Series(True, index=arcs.index)
        vals = arcs.loc[mask, _pick(raw_col)].dropna()
        row[out_col] = float(vals.median()) if len(vals) else np.nan

    return row


def aggregate_station_year(config: StationConfig, year: int,
                           force: bool = False) -> Path:
    """Stage 8 — produce ``{station}_{year}_daily_features.parquet``.

    Returns the output path.
    """
    resolver = PathResolver(config)
    out_path = resolver.daily_features(year)
    if out_path.exists() and not force:
        logger.info("Stage 8 aggregate: %s exists — skipping", out_path.name)
        return out_path

    arc_path = resolver.arc_head(year)
    if not arc_path.exists():
        raise FileNotFoundError(
            f"per-arc table not found: {arc_path} — run the earlier stages")
    arc = pd.read_parquet(arc_path)
    if "CLR" not in arc.columns:
        raise RuntimeError(
            f"{arc_path.name} has no SNR features — run Stage 3 (extract) first")

    if "date" not in arc.columns:
        arc["date"] = pd.to_datetime(
            arc["year"].astype(int) * 1000 + arc["doy"].astype(int),
            format="%Y%j").dt.strftime("%Y-%m-%d")
    if "azimuth_bin" not in arc.columns:
        arc["azimuth_bin"] = 0

    rows = []
    for date, day in arc.groupby("date"):
        for az in sorted(day["azimuth_bin"].unique()):
            sector = day[day["azimuth_bin"] == az]
            if sector.empty:
                continue
            entry = _aggregate_sector(sector)
            entry["date"] = date
            entry["azimuth_bin"] = int(az)
            rows.append(entry)
        pooled = _aggregate_sector(day)
        pooled["date"] = date
        pooled["azimuth_bin"] = -1
        rows.append(pooled)

    if not rows:
        raise RuntimeError(f"no daily feature rows for {config.station} {year}")

    daily = pd.DataFrame(rows)
    daily["year"] = year
    daily["doy"] = pd.to_datetime(daily["date"]).dt.dayofyear
    key_cols = ["year", "doy", "date", "azimuth_bin", "n_arcs"]
    key_cols = [c for c in key_cols if c in daily.columns]
    other = sorted(c for c in daily.columns if c not in key_cols)
    daily = daily[key_cols + other]

    resolver.ensure_output_dir()
    write_parquet(daily, out_path, config)
    n_pooled = int((daily["azimuth_bin"] == -1).sum())
    logger.info("Stage 8 aggregate: %s (%d rows, %d pooled days)",
                out_path.name, len(daily), n_pooled)
    return out_path
