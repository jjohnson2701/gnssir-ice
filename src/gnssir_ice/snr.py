# ABOUTME: SNR-file reader + Stage 1 consolidation of daily snr66 files.
# ABOUTME: Reads gnssrefl snr66 files and packs a station-year into one parquet.

"""SNR file I/O.

:func:`read_snr_file` parses one gnssrefl ``snr66`` file. :func:`consolidate_snr`
(pipeline Stage 1) gathers every daily SNR file of a station-year into a single
columnar parquet, so the feature extractor reads one file instead of ~365 ASCII
files.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import SNR66_COLUMNS

logger = logging.getLogger(__name__)

# Minimum SNR-file columns to be usable: sat, elev, azim, sod, edot, S6, S1.
# A file with at least this many can serve GPS L1; absent later signal columns
# are zero-padded.
_MIN_SNR_COLS = 7


def read_snr_file(path) -> np.ndarray:
    """Read a gnssrefl ``snr66`` file into an ``(n_obs, 11)`` array.

    Columns: sat, elevation(deg), azimuth(deg), seconds-of-day, edot,
    then S6, S1, S2, S5, S7, S8 (dB-Hz). Transparently reads ``.gz`` files.
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return np.loadtxt(f)
    return np.loadtxt(path)


def _resolve_snr_path(resolver: PathResolver, year: int, doy: int) -> Path | None:
    """Return the SNR file path for a day, accepting a ``.gz`` variant."""
    p = resolver.snr_file(year, doy)
    if p.exists():
        return p
    gz = Path(str(p) + ".gz")
    if gz.exists():
        return gz
    return None


def consolidate_snr(config: StationConfig, year: int,
                    force: bool = False) -> Path:
    """Stage 1 — consolidate a station-year's daily SNR files into one parquet.

    Reads every available ``snr66`` file for ``year`` and writes
    ``{station}_{year}_snr.parquet`` (the 11 snr66 columns plus ``doy``).

    Returns the output path. Raises :class:`FileNotFoundError` if no SNR files
    are found for the year.
    """
    resolver = PathResolver(config)
    out_path = resolver.snr_consolidated(year)
    if out_path.exists() and not force:
        logger.info("Stage 1 consolidate-snr: %s exists — skipping", out_path.name)
        return out_path

    resolver.ensure_output_dir()
    ncol = len(SNR66_COLUMNS)
    frames = []
    n_days = 0
    n_padded = 0          # short files padded with zero signal columns
    n_malformed = 0       # too few columns to use — skipped
    n_unreadable = 0
    for doy in range(1, 367):
        snr_path = _resolve_snr_path(resolver, year, doy)
        if snr_path is None:
            continue
        try:
            data = read_snr_file(snr_path)
        except Exception as exc:  # noqa: BLE001 - log and skip a bad file
            logger.debug("  failed to read %s: %s", snr_path.name, exc)
            n_unreadable += 1
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        cols = data.shape[1]
        if cols >= ncol:
            arr = data[:, :ncol]
        elif cols >= _MIN_SNR_COLS:
            # A short snr66 file (e.g. missing trailing Galileo S7/S8 columns)
            # is still valid for GPS — pad the absent signals with zeros.
            arr = np.zeros((data.shape[0], ncol))
            arr[:, :cols] = data
            n_padded += 1
        else:
            n_malformed += 1
            continue
        df = pd.DataFrame(arr, columns=SNR66_COLUMNS)
        df["doy"] = doy
        frames.append(df)
        n_days += 1

    if n_padded:
        logger.warning("  padded %d short SNR file(s) to %d columns "
                        "(absent trailing signals zero-filled)", n_padded, ncol)
    if n_malformed:
        logger.warning("  skipped %d SNR file(s) with < %d columns (unusable)",
                        n_malformed, _MIN_SNR_COLS)
    if n_unreadable:
        logger.warning("  skipped %d unreadable SNR file(s)", n_unreadable)

    if not frames:
        raise FileNotFoundError(
            f"no SNR files found for {config.station} {year} under "
            f"{resolver.snr_file(year, 1).parent}")

    combined = pd.concat(frames, ignore_index=True)
    combined["sat"] = combined["sat"].astype(np.int16)
    combined["doy"] = combined["doy"].astype(np.int16)
    combined.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Stage 1 consolidate-snr: %s (%d days, %d obs)",
                out_path.name, n_days, len(combined))
    return out_path


def load_consolidated_snr(config: StationConfig, year: int) -> pd.DataFrame:
    """Load the consolidated SNR parquet for a station-year."""
    path = PathResolver(config).snr_consolidated(year)
    if not path.exists():
        raise FileNotFoundError(
            f"consolidated SNR not found: {path} — run Stage 1 (consolidate-snr)")
    return pd.read_parquet(path)
