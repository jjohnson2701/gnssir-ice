# ABOUTME: Stage 3 — extract per-arc SNR features and persist CWT power curves.
# ABOUTME: Reads consolidated SNR + arc_table; writes arc_features + power_curves.

"""Stage 3 — per-arc SNR feature extraction.

For each retrieval in the arc table, this segments the matching SNR arc,
detrends it, and computes the per-arc features (:func:`extract_arc_features`).
The area factor is computed **uncorrected**; each full arc's raw CWT power
curve is persisted to ``{station}_{year}_power_curves.npz`` so the antenna-gain
correction (:mod:`gnssir_ice.af_gain`) can be applied later without re-reading
SNR or re-running the CWT.

Outputs: ``arc_features.parquet`` (the arc table plus the feature columns), a
human-readable ``arc_features.csv``, and the power-curve sidecar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gnssir_ice.arcs import (detrend_arc, find_matching_segment,
                             segment_satellite_arcs)
from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import ARC_JOIN_KEYS, FREQ_TO_COL, FREQ_WAVELENGTH
from gnssir_ice.features import extract_arc_features
from gnssir_ice.snr import load_consolidated_snr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-day extraction
# ---------------------------------------------------------------------------
def _extract_day(snr_data: np.ndarray, day_arcs: pd.DataFrame,
                 params: dict) -> list[dict]:
    """Extract features for every (arc, frequency) of one day.

    ``snr_data`` is the 11-column SNR array for the day; ``day_arcs`` is the
    arc-table slice for the same day.
    """
    e1, e2 = params["e1"], params["e2"]
    min_rh, max_rh = params["minH"], params["maxH"]
    poly_order = params["polyV"]
    pele = params["pele"]
    precision = params["desiredP"]

    arc_groups = day_arcs.groupby(["sat", "UTCtime", "rise"])
    results: list[dict] = []
    segment_cache: dict = {}

    for (sat, utctime, rise), freq_rows in arc_groups:
        if sat not in segment_cache:
            sat_data = snr_data[snr_data[:, 0] == sat]
            if len(sat_data) < 5:
                segment_cache[sat] = ([], None)
            else:
                arcs = segment_satellite_arcs(sat_data[:, 3], sat_data[:, 1])
                segment_cache[sat] = (arcs, sat_data)

        arcs, sat_data = segment_cache[sat]
        if sat_data is None or len(arcs) == 0:
            continue

        seg_idx = find_matching_segment(
            arcs, sat_data[:, 3], sat_data[:, 1],
            target_utctime=utctime, target_rise=rise, e1=e1, e2=e2)
        if seg_idx < 0:
            continue

        arc = arcs[seg_idx]
        arc_data = sat_data[arc["start_idx"]:arc["end_idx"]]
        arc_ele = arc_data[:, 1]

        for _, row in freq_rows.iterrows():
            freq = int(row["freq"])
            col_idx = FREQ_TO_COL.get(freq)
            if col_idx is None or col_idx >= arc_data.shape[1]:
                continue
            snr_db_col = arc_data[:, col_idx]
            if np.all(snr_db_col == 0):
                continue
            wavelength = FREQ_WAVELENGTH.get(freq)
            if wavelength is None:
                continue

            snr_lin = np.power(10, snr_db_col / 20)
            detrended = detrend_arc(arc_ele, snr_lin, poly_order, pele)

            feats = extract_arc_features(
                arc_ele, snr_db_col, snr_lin, detrended,
                wavelength, e1, e2, min_rh, max_rh, precision)
            if feats is None:
                continue
            feats["doy"] = int(day_arcs["doy"].iloc[0])
            feats["sat"] = int(sat)
            feats["UTCtime"] = float(utctime)
            feats["rise"] = int(rise)
            feats["freq"] = freq
            results.append(feats)
    return results


def _day_worker(args):
    """multiprocessing worker — must be module-level for pickling."""
    snr_data, day_arc_records, params = args
    day_arcs = pd.DataFrame.from_records(day_arc_records)
    return _extract_day(snr_data, day_arcs, params)


# ---------------------------------------------------------------------------
# Power-curve persistence (ragged store: concatenated values + offsets)
# ---------------------------------------------------------------------------
def _save_power_curves(path: Path, rows: list[dict]) -> int:
    """Persist each full arc's raw CWT power curve to a ragged npz store."""
    keys, curves, sins = [], [], []
    for r in rows:
        pc = r.get("_af_power_curve")
        se = r.get("_af_sin_elev")
        if pc is None or se is None:
            continue
        keys.append((r["doy"], r["sat"], r["UTCtime"], r["rise"], r["freq"]))
        curves.append(np.asarray(pc, dtype=float))
        sins.append(np.asarray(se, dtype=float))
    if not curves:
        np.savez(path, doy=np.array([]), sat=np.array([]), utc=np.array([]),
                 rise=np.array([]), freq=np.array([]),
                 offsets=np.array([0]), curve_values=np.array([]),
                 sin_values=np.array([]))
        return 0
    lengths = np.array([len(c) for c in curves])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    karr = np.array(keys, dtype=float)
    np.savez(
        path,
        doy=karr[:, 0].astype(np.int32), sat=karr[:, 1].astype(np.int32),
        utc=karr[:, 2], rise=karr[:, 3].astype(np.int32),
        freq=karr[:, 4].astype(np.int32),
        offsets=offsets,
        curve_values=np.concatenate(curves),
        sin_values=np.concatenate(sins),
    )
    return len(curves)


def load_power_curves(path: Path) -> pd.DataFrame:
    """Load the persisted power curves into a DataFrame.

    Each row carries the arc join keys plus ``power_curve`` and ``sin_elev``
    (1-D arrays).
    """
    z = np.load(path)
    offsets = z["offsets"]
    n = len(offsets) - 1
    cv, sv = z["curve_values"], z["sin_values"]
    rows = []
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i + 1])
        rows.append({
            "doy": int(z["doy"][i]), "sat": int(z["sat"][i]),
            "UTCtime": float(z["utc"][i]), "rise": int(z["rise"][i]),
            "freq": int(z["freq"][i]),
            "power_curve": cv[a:b], "sin_elev": sv[a:b],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3 driver
# ---------------------------------------------------------------------------
def extract_station_year(config: StationConfig, year: int,
                         jobs: int = 1, force: bool = False) -> Path:
    """Stage 3 — extract per-arc features for a station-year.

    Reads the consolidated SNR parquet and ``arc_table.parquet``, computes the
    per-arc SNR features, and writes ``arc_features.parquet`` (the arc table
    plus the feature columns), a human-readable ``arc_features.csv``, and the
    persisted power curves. Returns the ``arc_features.parquet`` path.
    """
    resolver = PathResolver(config)
    arc_table_path = resolver.arc_table(year)
    arc_features_path = resolver.arc_features(year)
    arc_features_csv = resolver.arc_features_csv(year)
    if arc_features_path.exists() and not force:
        logger.info("Stage 3 extract: %s exists — skipping", arc_features_path.name)
        return arc_features_path

    snr_df = load_consolidated_snr(config, year)
    if not arc_table_path.exists():
        raise FileNotFoundError(
            f"arc_table not found: {arc_table_path} — run Stage 2 (build-arc-table)")
    per_arc = pd.read_parquet(arc_table_path)

    g = config.gnssir
    params = {"e1": g.e1, "e2": g.e2, "minH": g.minH, "maxH": g.maxH,
              "polyV": g.polyV, "pele": tuple(g.pele), "desiredP": g.desiredP}

    doys = sorted(per_arc["doy"].unique())
    logger.info("Stage 3 extract: %s %d — %d days, %d arcs",
                config.station, year, len(doys), len(per_arc))

    snr_by_doy = {int(d): grp for d, grp in snr_df.groupby("doy")}
    snr_cols = [c for c in snr_df.columns if c != "doy"]

    missing = [int(d) for d in doys if int(d) not in snr_by_doy]
    if missing:
        logger.warning(
            "  %d of %d arc-table days have no SNR data and will get no "
            "features (e.g. DOY %s) — check Stage 1 coverage",
            len(missing), len(doys),
            ", ".join(str(d) for d in missing[:5])
            + ("..." if len(missing) > 5 else ""))

    day_args = []
    for doy in doys:
        snr_grp = snr_by_doy.get(int(doy))
        if snr_grp is None:
            continue
        snr_arr = snr_grp[snr_cols].to_numpy(dtype=float)
        day_arcs = per_arc[per_arc["doy"] == doy]
        day_args.append((snr_arr, day_arcs.to_dict("records"), params))

    if jobs > 1:
        from multiprocessing import Pool
        with Pool(jobs) as pool:
            day_results = pool.map(_day_worker, day_args)
        all_results = [r for day in day_results for r in day]
    else:
        all_results = []
        for args in day_args:
            all_results.extend(_day_worker(args))

    if not all_results:
        raise RuntimeError(f"no features extracted for {config.station} {year}")

    n_curves = _save_power_curves(resolver.power_curves(year), all_results)
    logger.info("  persisted %d power curves", n_curves)

    # Feature DataFrame (drop the underscore-prefixed power-curve payload).
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in all_results])
    logger.info("  extracted %d feature rows (%.1f%% full arcs)",
                len(df), 100 * df["full_arc"].mean())

    # Stage 3 artifact: arc_table + the per-arc features (feature RH renamed to
    # avoid colliding with the gnssrefl/subdaily RH).
    df_feat = df.rename(columns={"RH": "RH_snr"})
    feat_cols = [c for c in df_feat.columns if c not in ARC_JOIN_KEYS]
    per_arc_base = per_arc.drop(
        columns=[c for c in feat_cols if c in per_arc.columns], errors="ignore")
    arc_features = per_arc_base.merge(
        df_feat[ARC_JOIN_KEYS + feat_cols], on=ARC_JOIN_KEYS, how="left")
    resolver.ensure_output_dir()
    arc_features.to_parquet(arc_features_path, index=False, engine="pyarrow")

    # Human-readable CSV companion (per-arc features as first extracted; the
    # antenna-gain-corrected AF lands later, in arc_af.parquet).
    df_csv = df.copy()
    df_csv["year"] = year
    df_csv["phase_deg"] = np.degrees(df_csv["phase"])
    if "Amp" in arc_features.columns:
        amp_lookup = arc_features.set_index(ARC_JOIN_KEYS)["Amp"]
        df_csv = df_csv.set_index(ARC_JOIN_KEYS)
        df_csv["amp_raw"] = amp_lookup
        df_csv = df_csv.reset_index()
    csv_cols = [c for c in [
        "year", "doy", "sat", "UTCtime", "rise", "freq",
        "CLR", "PR", "AF", "gamma", "gamma_r2", "gamma_h_fit", "phase_deg",
        "SP", "MS", "VS",
        "full_arc", "clr_peak_power", "clr_total_power", "amp_raw",
    ] if c in df_csv.columns]
    df_csv[csv_cols].to_csv(arc_features_csv, index=False, float_format="%.6f")
    logger.info("  wrote %s (+ %s)", arc_features_path.name, arc_features_csv.name)
    return arc_features_path
