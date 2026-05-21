# ABOUTME: Stages 9/10 — fit the open-water Mahalanobis baseline and score days.
# ABOUTME: StandardScaler → PCA-whitening → Mahalanobis distance, pooled + per-sector.

"""Stages 9/10 — open-water baseline and Mahalanobis scoring.

:func:`fit_baseline` fits a StandardScaler → PCA-whitening model on pooled
open-water days, and records a per-azimuth-sector open-water mean for every
sector with enough data. :func:`apply_baseline` scores the pooled daily
features against it; :func:`apply_baseline_sectors` scores each azimuth sector
against its own recentred baseline. The drivers :func:`fit_station_baseline`
and :func:`score_station` wire those to the configured station-years.

The Mahalanobis distance is computed in PCA-whitened space: PCA scores are
decorrelated by construction, so their covariance is the diagonal matrix of
component variances and the distance reduces to the Euclidean norm of the
standardised principal components.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import MAHAL_FEATURES

logger = logging.getLogger(__name__)

MIN_BASELINE_DAYS = 60


# ---------------------------------------------------------------------------
# Fit / apply
# ---------------------------------------------------------------------------
def _mahalanobis(z_pc: np.ndarray, center: np.ndarray,
                 cov_pc_inv: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of PCA scores from ``center`` (per row)."""
    diff = z_pc - center
    d2 = np.einsum("ij,jk,ik->i", diff, cov_pc_inv, diff)
    return np.sqrt(np.clip(d2, 0, None))


def fit_baseline(pooled_frame: pd.DataFrame,
                 sector_frame: pd.DataFrame | None = None,
                 pca_variance: float = 0.95,
                 threshold_percentile: float = 99.0,
                 min_sector_days: int = 20, **meta) -> dict:
    """Fit a PCA-whitening + Mahalanobis baseline on open-water daily features.

    ``pooled_frame`` (the pooled, all-azimuth daily rows) must contain the
    :data:`MAHAL_FEATURES` columns. ``sector_frame``, if given, must also carry
    an ``azimuth_bin`` column; each sector with at least ``min_sector_days``
    complete open-water days gets its own recentred baseline. Extra keyword
    args are stored in the returned model dict as metadata.

    Raises :class:`ValueError` if fewer than :data:`MIN_BASELINE_DAYS` complete
    pooled feature rows are available.
    """
    feat_df = pooled_frame[MAHAL_FEATURES].dropna()
    n_baseline = len(feat_df)
    if n_baseline < MIN_BASELINE_DAYS:
        raise ValueError(
            f"only {n_baseline} complete baseline days; need "
            f">= {MIN_BASELINE_DAYS}")

    scaler = StandardScaler().fit(feat_df.values)
    pca = PCA(n_components=len(MAHAL_FEATURES)).fit(scaler.transform(feat_df.values))
    pcs = pca.transform(scaler.transform(feat_df.values))

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, pca_variance) + 1)
    n_comp = min(n_comp, len(MAHAL_FEATURES))
    pcs = pcs[:, :n_comp]

    mean_pc = pcs.mean(axis=0)
    # PCA scores are decorrelated by construction, so their covariance is the
    # diagonal matrix of component variances — invert it directly (PCA
    # whitening). No np.cov / np.linalg.inv needed; the off-diagonal terms of
    # np.cov(pcs.T) are only sampling noise.
    cov_pc_inv = np.diag(1.0 / pca.explained_variance_[:n_comp])

    mahal_d = _mahalanobis(pcs, mean_pc, cov_pc_inv)
    thr = float(np.percentile(mahal_d, threshold_percentile))

    # Per-sector open-water PC means (shared structure, per-sector recentring).
    sector_means: dict[int, np.ndarray] = {}
    sector_days: dict[int, int] = {}
    if sector_frame is not None and not sector_frame.empty \
            and "azimuth_bin" in sector_frame.columns:
        for sec, grp in sector_frame.groupby("azimuth_bin"):
            sg = grp[MAHAL_FEATURES].dropna()
            sector_days[int(sec)] = len(sg)
            if len(sg) < min_sector_days:
                continue
            spc = pca.transform(scaler.transform(sg.values))[:, :n_comp]
            sector_means[int(sec)] = spc.mean(axis=0)

    model = {
        "fit_date": datetime.now().strftime("%Y-%m-%d"),
        "retained_features": list(MAHAL_FEATURES),
        "n_baseline_days_used": n_baseline,
        "scaler": scaler,
        "pca": pca,
        "mean_pc": mean_pc,
        "cov_pc_inv": cov_pc_inv,
        "n_components": n_comp,
        "explained_variance_per_pc": list(pca.explained_variance_ratio_[:n_comp]),
        "total_variance_explained": float(cumvar[n_comp - 1]),
        "threshold_percentile": threshold_percentile,
        "threshold": thr,
        "threshold_theoretical_chi2": float(
            scipy.stats.chi2.ppf(threshold_percentile / 100.0, df=n_comp)),
        "baseline_mahal_d": {
            "mean": float(mahal_d.mean()),
            "median": float(np.median(mahal_d)),
            "p95": float(np.percentile(mahal_d, 95)),
            "max": float(mahal_d.max()),
        },
        "sector_means": sector_means,
        "sector_baseline_days": sector_days,
        "min_sector_days": min_sector_days,
    }
    model.update(meta)
    return model


def _score_frame(sub: pd.DataFrame, model: dict,
                 center: np.ndarray) -> pd.DataFrame:
    """Score the complete-feature rows of ``sub`` against a baseline ``center``."""
    retained = model["retained_features"]
    z_std = model["scaler"].transform(sub[retained].values)
    z_pc = model["pca"].transform(z_std)[:, :model["n_components"]]
    mahal_d = _mahalanobis(z_pc, center, model["cov_pc_inv"])
    out = pd.DataFrame({"mahal_d": mahal_d,
                        "above_threshold": mahal_d > model["threshold"],
                        "n_features_used": int(len(retained))})
    if "date" in sub.columns:
        out.insert(0, "date", pd.to_datetime(sub["date"].values))
    if "year" in sub.columns:
        out.insert(1, "year", sub["year"].values.astype(int))
    for j, f in enumerate(retained):
        out[f"{f}_z"] = z_std[:, j]
    return out


def _dropped_report(dropped: pd.DataFrame, retained: list[str]) -> pd.DataFrame:
    """Per-day record of which feature(s) were NaN, for days excluded from scoring."""
    if dropped.empty:
        return pd.DataFrame()
    rows = []
    for _, r in dropped.iterrows():
        nan_feats = [f for f in retained if pd.isna(r[f])]
        rows.append({
            "date": r["date"] if "date" in dropped.columns else None,
            "year": r["year"] if "year" in dropped.columns else None,
            "nan_features": ",".join(nan_feats),
        })
    return pd.DataFrame(rows)


def apply_baseline(daily_df: pd.DataFrame, model: dict,
                   return_dropped: bool = False):
    """Score pooled daily features against the open-water baseline.

    Days missing any of the model features are excluded. With
    ``return_dropped`` the call returns ``(scored, dropped)``, where
    ``dropped`` attributes each excluded day to the NaN feature(s).
    """
    retained = model["retained_features"]
    keep = [c for c in ("date", "year") if c in daily_df.columns]
    full = daily_df[keep + retained].copy()
    complete = full[retained].notna().all(axis=1)
    sub = full[complete]
    scored = (_score_frame(sub, model, model["mean_pc"])
              if not sub.empty else pd.DataFrame())
    if return_dropped:
        return scored, _dropped_report(full[~complete], retained)
    return scored


def apply_baseline_sectors(sector_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Score each azimuth sector against its own recentred open-water baseline.

    Sectors with no baseline in the model (too few open-water days, or never
    populated) are skipped.
    """
    sector_means = model.get("sector_means", {})
    if sector_df.empty or not sector_means \
            or "azimuth_bin" not in sector_df.columns:
        return pd.DataFrame()
    retained = model["retained_features"]
    frames = []
    for sec, grp in sector_df.groupby("azimuth_bin"):
        center = sector_means.get(int(sec))
        if center is None:
            continue
        keep = [c for c in ("date", "year") if c in grp.columns]
        sub = grp[keep + retained].dropna(subset=retained)
        if sub.empty:
            continue
        scored = _score_frame(sub, model, center)
        scored.insert(min(2, len(scored.columns)), "azimuth_bin", int(sec))
        frames.append(scored)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def _load_daily(resolver: PathResolver, year: int) -> pd.DataFrame:
    """Load the full daily-feature table for a year (pooled + sector rows)."""
    p = resolver.daily_features(year)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = year
    return df


def fit_station_baseline(config: StationConfig, force: bool = False) -> Path:
    """Stage 9 (fit) — fit and pickle the station's open-water baseline."""
    resolver = PathResolver(config)
    out_path = resolver.baseline_pkl()
    if out_path.exists() and not force:
        logger.info("Stage 9 fit-baseline: %s exists — skipping", out_path.name)
        return out_path

    months = set(config.baseline.open_water_months)
    pooled_frames, sector_frames = [], []
    for y in config.baseline.open_water_years:
        df = _load_daily(resolver, y)
        if df.empty:
            continue
        df = df[df["date"].dt.month.isin(months)]
        pooled_frames.append(df[df["azimuth_bin"] == -1])
        sector_frames.append(df[df["azimuth_bin"] >= 0])
    pooled = (pd.concat(pooled_frames, ignore_index=True)
              if pooled_frames else pd.DataFrame())
    if pooled.empty:
        raise RuntimeError(
            f"{config.station}: no daily_features for open_water_years "
            f"{config.baseline.open_water_years} — run Stage 8 (aggregate)")
    sector = (pd.concat(sector_frames, ignore_index=True)
              if sector_frames else pd.DataFrame())

    model = fit_baseline(
        pooled, sector_frame=sector,
        pca_variance=config.baseline.pca_variance,
        threshold_percentile=config.baseline.threshold_percentile,
        min_sector_days=config.baseline.min_sector_baseline_days,
        station=config.station,
        open_water_months=sorted(months),
        open_water_years=list(config.baseline.open_water_years))

    resolver.ensure_output_dir()
    with open(out_path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info(
        "Stage 9 fit-baseline: %s (n=%d, %d PCs, %.1f%% var, threshold=%.3f, "
        "%d sector baseline(s))",
        out_path.name, model["n_baseline_days_used"], model["n_components"],
        model["total_variance_explained"] * 100, model["threshold"],
        len(model["sector_means"]))
    return out_path


def _log_dropped(dropped_frames: list[pd.DataFrame], retained: list[str]) -> None:
    """Log how many days were dropped for NaN features, attributed by feature."""
    frames = [d for d in dropped_frames if not d.empty]
    if not frames:
        return
    dropped = pd.concat(frames, ignore_index=True)
    counts: dict[str, int] = {}
    for feats in dropped["nan_features"]:
        for f in str(feats).split(","):
            if f:
                counts[f] = counts.get(f, 0) + 1
    summary = ", ".join(f"{f}={n}" for f, n in sorted(counts.items())) or "none"
    logger.warning("  %d day(s) excluded from scoring for NaN features "
                   "(by feature: %s)", len(dropped), summary)


def score_station(config: StationConfig, years: list[int] | None = None) -> Path:
    """Stage 10 (score) — score station-years through the fitted baseline.

    Writes the pooled ``daily_mahal_d.parquet`` and, when the model carries
    per-sector baselines, ``sector_mahal_d.parquet``.
    """
    from gnssir_ice.provenance import write_parquet

    resolver = PathResolver(config)
    bl_path = resolver.baseline_pkl()
    if not bl_path.exists():
        raise FileNotFoundError(
            f"baseline not found: {bl_path} — run Stage 9 (fit-baseline)")
    with open(bl_path, "rb") as fh:
        model = pickle.load(fh)

    score_years = years or config.years or list(config.baseline.open_water_years)
    pooled_frames, sector_frames, dropped_frames = [], [], []
    for y in score_years:
        df = _load_daily(resolver, y)
        if df.empty:
            logger.warning("  %s %d: no daily_features — skipping",
                           config.station, y)
            continue
        scored, dropped = apply_baseline(
            df[df["azimuth_bin"] == -1], model, return_dropped=True)
        if not scored.empty:
            pooled_frames.append(scored)
        if not dropped.empty:
            dropped_frames.append(dropped)
        sec = apply_baseline_sectors(df[df["azimuth_bin"] >= 0], model)
        if not sec.empty:
            sector_frames.append(sec)
        logger.info("  %s %d: scored %d days (%d above threshold)",
                    config.station, y, len(scored),
                    int(scored["above_threshold"].sum())
                    if not scored.empty else 0)
    if not pooled_frames:
        raise RuntimeError(f"{config.station}: nothing scored")

    out = pd.concat(pooled_frames, ignore_index=True).sort_values("date") \
        .reset_index(drop=True)
    out_path = resolver.daily_mahal_d()
    resolver.ensure_output_dir()
    write_parquet(out, out_path, config, result="daily_mahal_d",
                  scored_years=sorted(score_years))
    out.to_csv(resolver.daily_mahal_d_csv(), index=False, float_format="%.6f")
    logger.info("Stage 10 score: %s (+ %s, %d days, years %s)",
                out_path.name, resolver.daily_mahal_d_csv().name,
                len(out), sorted(int(y) for y in out["year"].unique()))

    if sector_frames:
        sec_out = pd.concat(sector_frames, ignore_index=True) \
            .sort_values(["date", "azimuth_bin"]).reset_index(drop=True)
        write_parquet(sec_out, resolver.sector_mahal_d(), config,
                      result="sector_mahal_d")
        logger.info("  sector scores: %s (%d sector-days, %d sectors)",
                    resolver.sector_mahal_d().name, len(sec_out),
                    sec_out["azimuth_bin"].nunique())

    _log_dropped(dropped_frames, model["retained_features"])
    return out_path
