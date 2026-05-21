# ABOUTME: Stage 11 — extract freeze-up / break-up dates from the mahal_d series.
# ABOUTME: A season is a sustained run of consecutive above-threshold days.

"""Stage 11 — ice-season extraction.

Turns the daily Mahalanobis-distance series into discrete ice seasons. A season
is a maximal run of consecutive above-threshold scored days of length at least
``baseline.ice_season_min_run_days``; its freeze-up date is the run's first day
and its break-up date the run's last. Seasons are extracted for the pooled
series and, when ``sector_mahal_d`` exists, for each azimuth sector.

Missing (unscored) days do not break a run — runs are over consecutive scored
rows, so a one-day coverage gap does not split a season.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gnssir_ice.config import PathResolver, StationConfig

logger = logging.getLogger(__name__)

_COLUMNS = ["sector", "freeze_up", "break_up", "duration_days",
            "n_days_above", "peak_mahal_d", "peak_date"]


def _runs(above: pd.Series, min_run_days: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs of above-threshold runs >= min length."""
    flags = above.to_numpy(dtype=bool)
    runs, i, n = [], 0, len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        if (j - i + 1) >= min_run_days:
            runs.append((i, j))
        i = j + 1
    return runs


def _seasons_for(df: pd.DataFrame, sector: int,
                 min_run_days: int) -> list[dict]:
    """Extract season records from one date-sorted scored series."""
    df = df.sort_values("date").reset_index(drop=True)
    rows = []
    for i, j in _runs(df["above_threshold"], min_run_days):
        seg = df.iloc[i:j + 1]
        peak = seg.loc[seg["mahal_d"].idxmax()]
        fu, bu = pd.Timestamp(seg["date"].iloc[0]), pd.Timestamp(seg["date"].iloc[-1])
        rows.append({
            "sector": sector,
            "freeze_up": fu.date(),
            "break_up": bu.date(),
            "duration_days": (bu - fu).days + 1,
            "n_days_above": j - i + 1,
            "peak_mahal_d": round(float(peak["mahal_d"]), 4),
            "peak_date": pd.Timestamp(peak["date"]).date(),
        })
    return rows


def extract_ice_seasons(config: StationConfig, force: bool = False) -> Path:
    """Stage 11 — extract ice seasons from the scored Mahalanobis-distance series.

    Writes ``{station}_ice_seasons.csv`` (one row per season; ``sector = -1`` is
    the pooled, all-azimuth series). Returns the output path.
    """
    resolver = PathResolver(config)
    out_path = resolver.ice_seasons()
    if out_path.exists() and not force:
        logger.info("Stage 11 ice-seasons: %s exists — skipping", out_path.name)
        return out_path

    mahal_path = resolver.daily_mahal_d()
    if not mahal_path.exists():
        raise FileNotFoundError(
            f"daily_mahal_d not found: {mahal_path} — run Stage 10 (score)")
    min_run = config.baseline.ice_season_min_run_days

    pooled = pd.read_parquet(mahal_path)
    pooled["date"] = pd.to_datetime(pooled["date"])
    rows = _seasons_for(pooled, -1, min_run)

    sec_path = resolver.sector_mahal_d()
    if sec_path.exists():
        sec = pd.read_parquet(sec_path)
        sec["date"] = pd.to_datetime(sec["date"])
        for s, grp in sec.groupby("azimuth_bin"):
            rows.extend(_seasons_for(grp, int(s), min_run))

    seasons = pd.DataFrame(rows, columns=_COLUMNS).sort_values(
        ["sector", "freeze_up"]).reset_index(drop=True)
    resolver.ensure_output_dir()
    seasons.to_csv(out_path, index=False)
    logger.info(
        "Stage 11 ice-seasons: %s (%d season(s): %d pooled, %d sector)",
        out_path.name, len(seasons), int((seasons["sector"] == -1).sum()),
        int((seasons["sector"] >= 0).sum()))
    return out_path
