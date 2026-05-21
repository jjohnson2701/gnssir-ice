# ABOUTME: Stages 4/5 — antenna-gain AF baseline + the cheap AF correction.
# ABOUTME: Song 2022 Eq. 23, applied from persisted power curves (no SNR re-read).

"""Area-factor antenna-gain correction.

The raw CWT power curve is shaped by both the reflecting surface and the
antenna gain pattern. Song 2022 Eq. 23 removes the antenna term: for each
``(satellite, frequency)`` the open-water power curve is averaged into a
baseline, and every arc's AF is computed relative to it.

* Stage 4 (:func:`build_af_baseline`) averages the persisted open-water power
  curves per ``(sat, freq)``, pooled across ``baseline.open_water_years``,
  onto a common ``sin(ε)`` grid — one baseline per station. The antenna gain
  pattern is fixed hardware, so pooling years gives more arcs per channel.
* Stage 5 (:func:`apply_af_correction`) subtracts that baseline and
  re-integrates each arc's AF — a cheap pass over the persisted curves, no SNR
  re-read and no second CWT.
"""

from __future__ import annotations

import logging
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import ARC_JOIN_KEYS, SIN_GRID_SIZE
from gnssir_ice.extract import load_power_curves
from gnssir_ice.features import integrate_af

logger = logging.getLogger(__name__)


def _month_of(year: int, doy: int) -> int:
    return (_date(year, 1, 1) + pd.Timedelta(days=int(doy) - 1)).month


# ---------------------------------------------------------------------------
# Stage 4 — build the pooled AF baseline
# ---------------------------------------------------------------------------
def build_af_baseline(config: StationConfig,
                      force: bool = False) -> Path | None:
    """Stage 4 — build the pooled per-(sat, freq) antenna-gain AF baseline.

    Averages the persisted open-water power curves (months in
    ``baseline.normalization_months``) pooled across
    ``baseline.open_water_years`` onto a common ``sin(ε)`` grid, and writes
    ``{station}_af_baseline.npz`` — one baseline per station.

    Returns the output path, or None if there were no open-water power curves.
    """
    resolver = PathResolver(config)
    out_path = resolver.af_baseline()
    if out_path.exists() and not force:
        logger.info("Stage 4 af-baseline: %s exists — skipping", out_path.name)
        return out_path

    months = set(config.baseline.normalization_months)
    frames = []
    for y in config.baseline.open_water_years:
        pc_path = resolver.power_curves(y)
        if not pc_path.exists():
            logger.warning("Stage 4 af-baseline: %s missing — skipping year %d",
                           pc_path.name, y)
            continue
        pc = load_power_curves(pc_path)
        if pc.empty:
            continue
        pc["month"] = pc["doy"].map(lambda d, yr=y: _month_of(yr, d))
        frames.append(pc[pc["month"].isin(months)])

    open_water = (pd.concat(frames, ignore_index=True)
                  if frames else pd.DataFrame())
    if open_water.empty:
        logger.warning("Stage 4 af-baseline: no open-water power curves for "
                        "months %s, years %s — skipping",
                        sorted(months), list(config.baseline.open_water_years))
        return None

    cap = config.baseline.af_baseline_max_arcs
    if cap is not None and len(open_water) > cap:
        open_water = open_water.sample(cap, random_state=42)

    g = config.gnssir
    sin_grid = np.linspace(np.sin(np.radians(g.e1)),
                           np.sin(np.radians(g.e2)), SIN_GRID_SIZE)

    accumulator: dict[tuple[int, int], list[np.ndarray]] = {}
    for _, row in open_water.iterrows():
        x = np.asarray(row["sin_elev"], dtype=float)
        pcurve = np.asarray(row["power_curve"], dtype=float)
        if len(x) < 2:
            continue
        try:
            interp = interp1d(x, pcurve, bounds_error=False, fill_value=0.0)
            curve = interp(sin_grid)
        except ValueError:
            continue
        key = (int(row["sat"]), int(row["freq"]))
        accumulator.setdefault(key, []).append(curve)

    if not accumulator:
        logger.warning("Stage 4 af-baseline: no usable power curves")
        return None

    keys = np.array(sorted(accumulator.keys()))
    baselines = np.array([np.mean(accumulator[tuple(k)], axis=0) for k in keys])

    resolver.ensure_output_dir()
    np.savez(out_path, sin_grid=sin_grid, keys=keys, baselines=baselines)
    logger.info("Stage 4 af-baseline: %s (%d sat/freq baselines from %d arcs, "
                "years %s)", out_path.name, len(keys), len(open_water),
                list(config.baseline.open_water_years))
    return out_path


def load_af_baseline(path: Path) -> tuple[np.ndarray, dict]:
    """Load an AF baseline npz → ``(sin_grid, {(sat, freq): curve})``."""
    z = np.load(path)
    sin_grid = z["sin_grid"]
    baselines = {(int(k[0]), int(k[1])): curve
                 for k, curve in zip(z["keys"], z["baselines"])}
    return sin_grid, baselines


# ---------------------------------------------------------------------------
# Stage 5 — apply the AF correction
# ---------------------------------------------------------------------------
def apply_af_correction(config: StationConfig, year: int,
                        force: bool = False) -> Path:
    """Stage 5 — recompute the corrected AF from persisted power curves.

    Reads ``arc_features.parquet``, subtracts the matching ``(sat, freq)`` AF
    baseline from each arc's power curve, re-integrates, and writes
    ``arc_af.parquet`` — ``arc_features`` with the ``AF`` column replaced. An
    arc with no matching baseline keeps its uncorrected AF.
    """
    resolver = PathResolver(config)
    out_path = resolver.arc_af(year)
    if out_path.exists() and not force:
        logger.info("Stage 5 af-correct: %s exists — skipping", out_path.name)
        return out_path

    pc_path = resolver.power_curves(year)
    bl_path = resolver.af_baseline()
    feat_path = resolver.arc_features(year)
    if not pc_path.exists():
        raise FileNotFoundError(
            f"power curves not found: {pc_path} — run Stage 3 (extract)")
    if not bl_path.exists():
        raise FileNotFoundError(
            f"AF baseline not found: {bl_path} — run Stage 4 (af-baseline)")
    if not feat_path.exists():
        raise FileNotFoundError(
            f"arc_features not found: {feat_path} — run Stage 3 (extract)")

    pc = load_power_curves(pc_path)
    sin_grid, baselines = load_af_baseline(bl_path)

    af_rows = []
    n_corrected = 0
    for _, row in pc.iterrows():
        key = (int(row["sat"]), int(row["freq"]))
        bl = baselines.get(key)
        x = row["sin_elev"]
        curve = row["power_curve"]
        if bl is not None:
            af = integrate_af(curve, x, baseline_power_curve=bl,
                              baseline_sin_grid=sin_grid)
            n_corrected += 1
        else:
            af = integrate_af(curve, x)
        af_rows.append({
            "doy": int(row["doy"]), "sat": int(row["sat"]),
            "UTCtime": float(row["UTCtime"]), "rise": int(row["rise"]),
            "freq": int(row["freq"]), "AF": af,
        })
    af_df = pd.DataFrame(af_rows)

    arc = pd.read_parquet(feat_path)
    if af_df.empty:
        logger.warning("Stage 5 af-correct: no power curves — AF left uncorrected")
    else:
        arc = arc.drop(columns=["AF"], errors="ignore").merge(
            af_df, on=ARC_JOIN_KEYS, how="left")

    resolver.ensure_output_dir()
    arc.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Stage 5 af-correct: %s — %d/%d arcs baseline-corrected",
                out_path.name, n_corrected, len(af_df))
    return out_path
