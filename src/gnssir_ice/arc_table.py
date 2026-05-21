# ABOUTME: Stage 2 — build arc_table from gnssrefl gnssir output + subdaily RHdot.
# ABOUTME: Merges raw per-arc retrievals with the RHdot/IF-corrected reflector height.

"""Arc-table builder.

gnssrefl's ``gnssir`` writes one ``.txt`` file of per-arc retrievals per day;
``subdaily`` writes a ``.withrhdotIF`` file with the RHdot + inter-frequency
corrected reflector height. :func:`build_arc_table` reads both, joins them, and
writes ``{station}_{year}_arc_table.parquet`` — the per-arc table every later
stage consumes.

The RHdot/IF correction is **required**: ``build_arc_table`` raises if the
``.withrhdotIF`` file is missing, or if the fraction of arcs that receive the
correction falls below ``options.min_subdaily_match_rate`` — a guard against a
silent gnssrefl output-format mismatch. Arcs edited out by ``subdaily`` keep
their raw RH (``RH_raw``); the corrected ``RH`` for those rows falls back to
``RH_raw`` and ``subdaily_qc_pass`` is False.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import (GNSSIR_COLUMNS, SUBDAILY_MATCH_WARN,
                                  TESTED_GNSSREFL_VERSION)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
def _az_bin(azimuth, width: int) -> int:
    """Map an azimuth (0-360) to its sector bin (lower bound, degrees)."""
    return int(azimuth // width) * width


def add_derived_columns(df: pd.DataFrame,
                        antenna_height_m: float | None = None,
                        sector_width: int = 30) -> pd.DataFrame:
    """Add ``azimuth_bin``, ``date`` and (optionally) ``wse``.

    ``sector_width`` (degrees) sets the azimuth-sector bin width. Operates in
    place and returns the same DataFrame.
    """
    if "Azim" in df.columns:
        df["azimuth_bin"] = df["Azim"].apply(
            lambda a: _az_bin(a, sector_width)).astype(np.int16)
    if "year" in df.columns and "doy" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(int) * 1000 + df["doy"].astype(int),
            format="%Y%j",
        ).dt.strftime("%Y-%m-%d")
    if antenna_height_m is not None and "RH" in df.columns:
        df["wse"] = antenna_height_m - df["RH"]
    return df


# ---------------------------------------------------------------------------
# gnssir per-day .txt reader
# ---------------------------------------------------------------------------
def read_gnssir_txt(path) -> pd.DataFrame | None:
    """Read one gnssrefl ``gnssir`` per-day ``.txt`` file (17-column format).

    Returns a DataFrame with :data:`GNSSIR_COLUMNS`, or None if the file is
    empty / unparseable.
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return None
    header = 0
    for line in lines:
        if line.lstrip().startswith("%"):
            header += 1
        else:
            break
    if header >= len(lines):
        return None
    try:
        df = pd.read_csv(path, skiprows=header, sep=r"\s+", header=None)
    except (pd.errors.EmptyDataError, ValueError):
        return None
    if df.empty:
        return None
    n = len(df.columns)
    ncan = len(GNSSIR_COLUMNS)
    if n >= ncan:
        df = df.iloc[:, :ncan]
        df.columns = GNSSIR_COLUMNS
    else:
        df.columns = GNSSIR_COLUMNS[:n]
    return df


def read_gnssir_results(results_dir) -> pd.DataFrame:
    """Read and concatenate every gnssir per-day ``.txt`` in a results directory."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"gnssir results directory not found: {results_dir}")
    frames = []
    for txt in sorted(results_dir.glob("*.txt")):
        df = read_gnssir_txt(txt)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"no parseable gnssir .txt files in {results_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# subdaily .withrhdotIF reader
# ---------------------------------------------------------------------------
def read_withrhdotIF(path) -> pd.DataFrame:
    """Read a gnssrefl ``subdaily`` ``.withrhdotIF`` file.

    Returns a DataFrame with the join keys plus ``rh_rhdot_corrected``,
    ``rhdot_correction`` and ``rh_if_corrected`` (columns 22/23/24).
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 25:
                continue
            rows.append({
                "year": int(parts[0]),
                "doy": int(parts[1]),
                "sat": int(parts[3]),
                "freq": int(parts[10]),
                "MJD": float(parts[15]),
                "rh_rhdot_corrected": float(parts[22]),
                "rhdot_correction": float(parts[23]),
                "rh_if_corrected": float(parts[24]),
            })
    return pd.DataFrame(rows)


def _parse_gnssrefl_version(path) -> str | None:
    """Extract the gnssrefl version from a subdaily file's ``%`` header.

    The first header line reads, e.g., ``% Results for ross calculated with
    gnssrefl v3.19.3 on ...``. Returns the version string, or None if absent.
    """
    try:
        with open(path) as f:
            for line in f:
                if not line.lstrip().startswith("%"):
                    break
                m = re.search(r"gnssrefl\s+v?(\d+\.\d+(?:\.\d+)?)", line)
                if m:
                    return m.group(1)
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# Stage 2 — build_arc_table
# ---------------------------------------------------------------------------
def _merge_subdaily(arc: pd.DataFrame, corrected: pd.DataFrame,
                    antenna_height_m: float | None) -> tuple[pd.DataFrame, int]:
    """Merge RHdot/IF-corrected RH into the arc table on (year, doy, sat, freq, MJD)."""
    arc = arc.copy()
    arc["RH_raw"] = arc["RH"]

    if corrected.empty:
        arc["RH_with_rhdot"] = np.nan
        arc["RHdot_corr_m"] = np.nan
        arc["IF_corr_m"] = np.nan
        arc["subdaily_qc_pass"] = False
        return arc, 0

    corr = corrected.copy()
    corr["IF_corr_m"] = corr["rh_if_corrected"] - corr["rh_rhdot_corrected"]
    corr = corr.rename(columns={
        "rh_if_corrected": "RH_corrected",
        "rh_rhdot_corrected": "RH_with_rhdot",
        "rhdot_correction": "RHdot_corr_m",
    })
    # MJD (sub-second precision) rounded to ~1 s absorbs gnssrefl rounding drift.
    arc["_mjd_key"] = arc["MJD"].round(5)
    corr["_mjd_key"] = corr["MJD"].round(5)
    keys = ["year", "doy", "sat", "freq", "_mjd_key"]
    merged = arc.merge(
        corr[keys + ["RH_corrected", "RH_with_rhdot", "RHdot_corr_m", "IF_corr_m"]],
        on=keys, how="left",
    ).drop(columns=["_mjd_key"])

    n_matched = int(merged["RH_corrected"].notna().sum())
    merged["subdaily_qc_pass"] = merged["RH_corrected"].notna()
    merged["RH"] = merged["RH_corrected"].fillna(merged["RH_raw"])
    merged = merged.drop(columns=["RH_corrected"])
    if antenna_height_m is not None:
        merged["wse"] = antenna_height_m - merged["RH"]
    return merged, n_matched


def build_arc_table(config: StationConfig, year: int,
                    force: bool = False) -> Path:
    """Stage 2 — build the per-arc table for a station-year.

    Reads gnssrefl ``gnssir`` per-day output and the ``subdaily`` ``.withrhdotIF``
    file, joins them, and writes ``{station}_{year}_arc_table.parquet`` with raw
    RH preserved as ``RH_raw`` and the RHdot/IF-corrected value in ``RH``.

    Raises :class:`FileNotFoundError` if the ``.withrhdotIF`` file is missing
    (the RHdot correction is required).
    """
    resolver = PathResolver(config)
    out_path = resolver.arc_table(year)
    if out_path.exists() and not force:
        logger.info("Stage 2 build-arc-table: %s exists — skipping", out_path.name)
        return out_path

    arc = read_gnssir_results(resolver.gnssir_results_dir(year))
    arc = arc[arc["year"].astype(int) == year].reset_index(drop=True)
    if arc.empty:
        raise ValueError(f"no gnssir retrievals for {config.station} {year}")

    add_derived_columns(
        arc, antenna_height_m=config.coordinates.ellipsoidal_height_m,
        sector_width=config.baseline.sector_width_deg)

    subdaily_path = resolver.subdaily_file(year)
    if not subdaily_path.exists():
        raise FileNotFoundError(
            f"subdaily .withrhdotIF file required but not found: {subdaily_path}\n"
            f"Run `subdaily {config.station.lower()} {year}` with gnssrefl first.")

    version = _parse_gnssrefl_version(subdaily_path)
    if version and version != TESTED_GNSSREFL_VERSION:
        logger.warning(
            "  subdaily file was produced by gnssrefl v%s; gnssir-ice's format "
            "parsing was verified against v%s — proceed with caution",
            version, TESTED_GNSSREFL_VERSION)

    corrected = read_withrhdotIF(subdaily_path)
    if corrected.empty:
        logger.warning(
            "subdaily file %s yielded no usable rows (need >= 25 columns)",
            subdaily_path.name)
    arc, n_matched = _merge_subdaily(
        arc, corrected, config.coordinates.ellipsoidal_height_m)

    match_rate = n_matched / len(arc)
    floor = config.options.min_subdaily_match_rate
    if match_rate < floor:
        raise RuntimeError(
            f"{config.station} {year}: only {n_matched}/{len(arc)} arcs "
            f"({match_rate:.0%}) received the subdaily RHdot/IF correction — "
            f"below the options.min_subdaily_match_rate floor of {floor:.0%}. "
            f"This usually means the gnssrefl subdaily output format does not "
            f"match what gnssir-ice expects (tested against gnssrefl "
            f"v{TESTED_GNSSREFL_VERSION}); inspect {subdaily_path.name}.")
    if match_rate < SUBDAILY_MATCH_WARN:
        logger.warning(
            "  subdaily match rate %.0f%% is below the %.0f%% warning level — "
            "unusually low but not fatal; check %s",
            100 * match_rate, 100 * SUBDAILY_MATCH_WARN, subdaily_path.name)

    resolver.ensure_output_dir()
    arc.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(
        "Stage 2 build-arc-table: %s (%d arcs, %d/%d RHdot+IF-corrected, %.1f%%)",
        out_path.name, len(arc), n_matched, len(arc), 100 * match_rate)
    return out_path
