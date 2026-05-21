# ABOUTME: Pipeline orchestration — wires the 11 stages into one run_pipeline().
# ABOUTME: The only module that imports the stage drivers.

"""Pipeline orchestration.

:func:`run_pipeline` walks the 11-stage pipeline for a station:

    1 consolidate-snr → 2 build-arc-table → 3 extract → 4 af-baseline
      → 5 af-correct → 6 prn-baseline → 7 normalize → 8 aggregate
      → 9 fit-baseline → 10 score → 11 ice-seasons

Per-arc data flows through an immutable artifact chain (arc_table →
arc_features → arc_af → arc_norm). The AF baseline (stage 4) and the per-PRN
baseline (stage 6) both pool ``baseline.open_water_years``, so ``run_pipeline``
processes the union of the requested years and the open-water years through
the per-year stages before fitting.
"""

from __future__ import annotations

import logging

from gnssir_ice.aggregate import aggregate_station_year
from gnssir_ice.af_gain import apply_af_correction, build_af_baseline
from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.extract import extract_station_year
from gnssir_ice.arc_table import build_arc_table
from gnssir_ice.model import fit_station_baseline, score_station
from gnssir_ice.normalize import apply_prn_normalization, build_prn_baseline
from gnssir_ice.provenance import write_run_manifest
from gnssir_ice.seasons import extract_ice_seasons
from gnssir_ice.snr import consolidate_snr

logger = logging.getLogger(__name__)


def process_year_extract(config: StationConfig, year: int,
                          jobs: int = 1, force: bool = False) -> None:
    """Per-year arc extraction — stages 1-3: consolidate-snr → build-arc-table
    → extract."""
    logger.info("=== %s %d : arc extraction (stages 1-3) ===",
                config.station, year)
    consolidate_snr(config, year, force=force)
    build_arc_table(config, year, force=force)
    extract_station_year(config, year, jobs=jobs, force=force)


def run_pipeline(config: StationConfig, years: list[int] | None = None,
                 jobs: int = 1, force: bool = False) -> dict:
    """Run the full 11-stage pipeline for a station.

    Args:
        config: the station configuration
        years: years to score (default: ``config.years``)
        jobs: parallel workers for the extraction stage
        force: recompute stage outputs even if they already exist

    Returns a dict with the ``baseline`` and ``daily_mahal_d`` output paths.
    """
    score_years = list(years or config.years)
    if not score_years:
        raise ValueError("no years to process — pass years or set processing.years")
    all_years = sorted(set(score_years) | set(config.baseline.open_water_years))
    if set(all_years) != set(score_years):
        logger.info("processing %s (adds baseline open-water years)", all_years)

    # Stages 1-3 — per-year arc extraction.
    for year in all_years:
        process_year_extract(config, year, jobs=jobs, force=force)

    # Stages 4-5 — pooled antenna-gain AF correction.
    if config.options.af_baseline:
        logger.info("=== %s : antenna-gain AF correction (stages 4-5) ===",
                    config.station)
        if build_af_baseline(config, force=force) is not None:
            for year in all_years:
                apply_af_correction(config, year, force=force)
        else:
            logger.warning("  no AF baseline could be built — AF left "
                            "uncorrected for this run")
            config.options.af_baseline = False
    else:
        logger.info("af_baseline disabled — AF left uncorrected")

    # Stages 6-7 — per-PRN normalization (stage 6 pools years; 7 is per-year).
    if config.options.per_prn_normalization:
        logger.info("=== %s : per-PRN normalization (stages 6-7) ===",
                    config.station)
        build_prn_baseline(config, force=force)
        for year in all_years:
            apply_prn_normalization(config, year, force=force)
    else:
        logger.warning("per_prn_normalization disabled — rh_std_norm will "
                        "equal rh_std_raw (PCA collapses the redundant axis)")

    # Stage 8 — daily aggregation.
    for year in all_years:
        aggregate_station_year(config, year, force=force)

    # Stages 9-11 — baseline fit, scoring, ice-season extraction.
    logger.info("=== %s : baseline fit + scoring (stages 9-11) ===",
                config.station)
    baseline_path = fit_station_baseline(config, force=force)
    mahal_path = score_station(config, years=score_years)
    seasons_path = extract_ice_seasons(config, force=force)

    _log_run_summary(config, score_years, mahal_path)
    resolver = PathResolver(config)
    write_run_manifest(config, resolver.run_manifest(),
                       scored_years=sorted(score_years))
    logger.info("  run manifest: %s", resolver.run_manifest().name)
    return {"baseline": baseline_path, "daily_mahal_d": mahal_path,
            "ice_seasons": seasons_path}


def _log_run_summary(config: StationConfig, score_years: list[int],
                     mahal_path) -> None:
    """Log a plain-language summary of what the run actually produced."""
    import pandas as pd

    resolver = PathResolver(config)
    df = pd.read_parquet(mahal_path)
    df["date"] = pd.to_datetime(df["date"])
    scored = len(df)

    # Honest denominator: days that produced daily features (i.e. the station
    # was recording). Days with no input data at all are not a coverage gap —
    # counting them against the full calendar cried wolf on partial-year runs.
    observed = 0
    for y in score_years:
        p = resolver.daily_features(y)
        if p.exists():
            fy = pd.read_parquet(p, columns=["azimuth_bin"])
            observed += int((fy["azimuth_bin"] == -1).sum())

    logger.info("=== %s : run summary ===", config.station)
    logger.info("  scored %d of %d observed days (%.0f%%), years %s",
                scored, observed,
                100 * scored / observed if observed else 0.0,
                sorted(int(y) for y in score_years))
    logger.info("  date span: %s to %s",
                df["date"].min().date(), df["date"].max().date())
    logger.info("  above threshold: %d days (%.0f%%)",
                int(df["above_threshold"].sum()),
                100 * df["above_threshold"].mean() if scored else 0.0)
    gap = observed - scored
    if observed and gap > observed * 0.05:
        logger.warning("  %d observed day(s) had data but were not scored — "
                        "NaN daily features; see the Stage 10 log for the "
                        "per-feature breakdown.", gap)
