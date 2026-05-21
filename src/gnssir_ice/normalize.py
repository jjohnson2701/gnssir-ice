# ABOUTME: Stages 6/7 — per-(PRN, signal) z-score baseline and its application.
# ABOUTME: Removes per-satellite/antenna offsets so daily features are comparable.

"""Per-(PRN, signal) z-score normalization.

Each ``(satellite, frequency)`` channel carries its own antenna-gain and
baseline-SNR offset. Stage 6 (:func:`build_prn_baseline`) pools open-water arcs
across years and computes a per-channel mean/std for each feature, with a
fallback hierarchy when a channel is thinly sampled:

  * level 0 — per-(PRN, signal): ≥30 open-water arcs
  * level 1 — per-signal (frequency only)
  * level 2 — station-wide
  * level 3 — passthrough (left un-normalized)

Stage 7 (:func:`apply_prn_normalization`) writes ``{feature}_norm`` columns
into ``arc_norm.parquet``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import GAMMA_R2_MIN, PRN_BASELINE_FEATURES

logger = logging.getLogger(__name__)

MIN_ARCS_PRN_SIGNAL = 30   # level 0 → level 1 threshold
MIN_ARCS_PER_SIGNAL = 30   # level 1 → level 2 threshold
MIN_ARCS_STATION = 30      # level 2 → level 3 threshold


# ---------------------------------------------------------------------------
# Stage 6 — build the per-PRN baseline
# ---------------------------------------------------------------------------
def _load_open_water_arcs(config: StationConfig) -> pd.DataFrame | None:
    """Pool open-water arcs across ``baseline.open_water_years``."""
    resolver = PathResolver(config)
    months = set(config.baseline.normalization_months)
    frames = []
    for y in config.baseline.open_water_years:
        p = resolver.arc_pre_norm(y)
        if not p.exists():
            logger.warning("%s %d: %s missing — skipping",
                            config.station, y, p.name)
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        logger.error("%s: no per-arc data for open_water_years %s",
                      config.station, config.baseline.open_water_years)
        return None
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    return df[df["month"].isin(months)].copy()


def _stats(values: pd.Series) -> tuple[float, float, int]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), (float(v.std(ddof=1)) if n >= 2 else 0.0), n


def _values_for_feature(summer: pd.DataFrame, feat: str) -> pd.DataFrame:
    """Rows where ``feat`` is usable (γ gated by gamma_r2)."""
    if feat not in summer.columns:
        return pd.DataFrame()
    if feat == "gamma" and "gamma_r2" in summer.columns:
        mask = summer["gamma_r2"] >= GAMMA_R2_MIN
        return summer.loc[mask, ["sat", "freq", feat]].dropna()
    return summer.loc[:, ["sat", "freq", feat]].dropna()


def build_prn_baseline(config: StationConfig, force: bool = False) -> Path | None:
    """Stage 6 — compute the per-(PRN, signal) z-score baseline.

    Writes ``{station}_prn_baseline.json``. Returns its path, or None if there
    were no open-water arcs.
    """
    resolver = PathResolver(config)
    out_path = resolver.prn_baseline()
    if out_path.exists() and not force:
        logger.info("Stage 6 prn-baseline: %s exists — skipping", out_path.name)
        return out_path

    summer = _load_open_water_arcs(config)
    if summer is None or summer.empty:
        logger.warning("Stage 6 prn-baseline: no open-water arcs")
        return None

    out = {
        "station": config.station,
        "open_water_years": list(config.baseline.open_water_years),
        "normalization_months": list(config.baseline.normalization_months),
        "n_open_water_arcs": int(len(summer)),
        "features": {},
    }

    for feat in PRN_BASELINE_FEATURES:
        feat_df = _values_for_feature(summer, feat)
        if feat_df.empty:
            logger.warning("%s: no usable values for feature %s",
                           config.station, feat)
            out["features"][feat] = {"available": False}
            continue

        per_ps = feat_df.groupby(["sat", "freq"])[feat].agg(["mean", "std", "count"])
        per_ps = per_ps.rename(columns={"count": "n"})
        per_s = feat_df.groupby("freq")[feat].agg(["mean", "std", "count"])
        per_s = per_s.rename(columns={"count": "n"})
        smean, sstd, sn = _stats(feat_df[feat])

        lookup: dict[tuple[int, int], tuple[float, float, int, int]] = {}
        pairs = summer[["sat", "freq"]].drop_duplicates().itertuples(
            index=False, name=None)
        for sat, freq in pairs:
            sat, freq = int(sat), int(freq)
            if (sat, freq) in per_ps.index:
                row = per_ps.loc[(sat, freq)]
                if (row["n"] >= MIN_ARCS_PRN_SIGNAL and pd.notna(row["std"])
                        and row["std"] > 0):
                    lookup[(sat, freq)] = (float(row["mean"]), float(row["std"]),
                                           int(row["n"]), 0)
                    continue
            if freq in per_s.index:
                srow = per_s.loc[freq]
                if (srow["n"] >= MIN_ARCS_PER_SIGNAL and pd.notna(srow["std"])
                        and srow["std"] > 0):
                    lookup[(sat, freq)] = (float(srow["mean"]), float(srow["std"]),
                                           int(srow["n"]), 1)
                    continue
            if sn >= MIN_ARCS_STATION and sstd > 0:
                lookup[(sat, freq)] = (smean, sstd, sn, 2)
                continue
            lookup[(sat, freq)] = (np.nan, np.nan, 0, 3)

        out["features"][feat] = {
            "available": True,
            "lookup": {f"{s}_{f}": v for (s, f), v in lookup.items()},
            "station_mean": smean, "station_std": sstd, "station_n": sn,
            "per_signal": {
                int(f): {
                    "mean": float(per_s.loc[f, "mean"]),
                    "std": (float(per_s.loc[f, "std"])
                            if pd.notna(per_s.loc[f, "std"]) else 0.0),
                    "n": int(per_s.loc[f, "n"]),
                }
                for f in per_s.index
            },
            "fallback_counts": {
                f"level_{lvl}": sum(1 for v in lookup.values() if v[3] == lvl)
                for lvl in (0, 1, 2, 3)
            },
        }

    resolver.ensure_output_dir()
    out_path.write_text(json.dumps(out, indent=2, default=float))
    logger.info("Stage 6 prn-baseline: %s (%d open-water arcs)",
                out_path.name, len(summer))
    return out_path


# ---------------------------------------------------------------------------
# Stage 7 — apply the normalization
# ---------------------------------------------------------------------------
def _normalize_feature(df: pd.DataFrame, feat: str,
                       feat_info: dict) -> tuple[pd.Series, pd.Series]:
    """Return ``(z_values, passthrough_flags)`` for one feature."""
    lookup = feat_info["lookup"]
    per_signal = feat_info.get("per_signal", {})
    smean = feat_info.get("station_mean", float("nan"))
    sstd = feat_info.get("station_std", float("nan"))
    sn = feat_info.get("station_n", 0)

    raw = pd.to_numeric(df[feat], errors="coerce")
    z = pd.Series(np.nan, index=df.index, dtype=float)
    passthrough = pd.Series(False, index=df.index, dtype=bool)

    for (sat, freq), group_idx in df.groupby(["sat", "freq"]).groups.items():
        key = f"{int(sat)}_{int(freq)}"
        info = lookup.get(key)
        if info is not None:
            mean_v, std_v, _n_v, level = info
            if level < 3 and std_v and std_v > 0:
                z.loc[group_idx] = (raw.loc[group_idx] - mean_v) / std_v
                continue
        ps = per_signal.get(str(int(freq))) or per_signal.get(int(freq))
        if ps and ps.get("std", 0) > 0 and ps.get("n", 0) >= 30:
            z.loc[group_idx] = (raw.loc[group_idx] - ps["mean"]) / ps["std"]
            continue
        if sn >= 30 and sstd and sstd > 0:
            z.loc[group_idx] = (raw.loc[group_idx] - smean) / sstd
            continue
        z.loc[group_idx] = raw.loc[group_idx]
        passthrough.loc[group_idx] = True

    return z, passthrough


def apply_prn_normalization(config: StationConfig, year: int,
                            force: bool = False) -> Path:
    """Stage 7 — write ``{feature}_norm`` columns into ``arc_norm.parquet``."""
    resolver = PathResolver(config)
    out_path = resolver.arc_norm(year)
    if out_path.exists() and not force:
        logger.info("Stage 7 normalize: %s exists — skipping", out_path.name)
        return out_path

    bl_path = resolver.prn_baseline()
    if not bl_path.exists():
        raise FileNotFoundError(
            f"per-PRN baseline not found: {bl_path} — run Stage 6 (prn-baseline)")
    baseline = json.loads(bl_path.read_text())

    in_path = resolver.arc_pre_norm(year)
    if not in_path.exists():
        raise FileNotFoundError(
            f"per-arc table not found: {in_path} — run Stage 3 (extract) "
            "and Stage 5 (af-correct) first")
    df = pd.read_parquet(in_path)

    n_done = 0
    for feat in PRN_BASELINE_FEATURES:
        info = baseline["features"].get(feat, {})
        if not info.get("available", False):
            logger.warning("%s: feature %s unavailable in baseline — skipping",
                           config.station, feat)
            continue
        if feat not in df.columns:
            logger.warning("%s %d: column %s missing in per-arc table — skipping",
                           config.station, year, feat)
            continue
        z, passthrough = _normalize_feature(df, feat, info)
        df[f"{feat}_norm"] = z
        df[f"{feat}_norm_passthrough"] = passthrough
        n_done += 1

    resolver.ensure_output_dir()
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Stage 7 normalize: %s %d — %d features normalized",
                config.station, year, n_done)
    return out_path
