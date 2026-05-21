# ABOUTME: Optional result plotting — per-ice-year mahal_d panels vs validation.
# ABOUTME: Needs the [viz] extra (matplotlib); imported lazily by the CLI.

"""Result plotting (optional — needs the ``[viz]`` extra: matplotlib).

:func:`plot_station` renders the scored daily Mahalanobis-distance series as one
panel per ice year (Jul–Jun), each with its own date axis, vertical stripes
over the open-water months the baseline was sampled from, the alarm threshold,
and — when a validation CSV is supplied — an independent ground-truth ice
series overlaid on a second axis. With ``features`` it *also* writes a separate
``*_features.png`` — a per-feature z-score heatmap per ice year, showing
*which* feature drove a high ``mahal_d``.

The validation CSV is a generic two-column file: ``date`` (YYYY-MM-DD) and
``ice_fraction`` (0 = open water, 1 = full ice). Convert GLERL ice
concentration or hand-labelled camera data into that schema — see
``examples/camera_labels_to_validation.py``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from gnssir_ice.config import PathResolver, StationConfig
from gnssir_ice.constants import MAHAL_FEATURES

logger = logging.getLogger(__name__)


def _season_windows(dates: pd.Series) -> list:
    """Ice-year windows (Jul 1 … Jun 30) that overlap the scored dates."""
    windows = []
    for end_year in range(int(dates.min().year), int(dates.max().year) + 2):
        start = pd.Timestamp(end_year - 1, 7, 1)
        end = pd.Timestamp(end_year, 6, 30)
        if ((dates >= start) & (dates <= end)).any():
            windows.append((end_year, start, end))
    return windows


def _season_label(end_year) -> str:
    return (f"{end_year - 1}-{str(end_year)[2:]} ice year"
            if end_year else "scored series")


def _sampled_month_spans(start: pd.Timestamp, end: pd.Timestamp,
                         months, years) -> list:
    """``(month_start, month_end)`` intervals within ``[start, end]`` that fed
    the open-water baseline — a configured open-water month of an open-water
    year."""
    months, years = set(months), set(years)
    spans, m = [], pd.Timestamp(start.year, start.month, 1)
    while m <= end:
        nxt = m + pd.offsets.MonthBegin(1)
        if m.month in months and m.year in years:
            spans.append((max(m, start), min(nxt, end)))
        m = nxt
    return spans


def _load_validation(path: Path) -> pd.Series:
    """Load a generic validation CSV → a date-indexed ``ice_fraction`` series."""
    v = pd.read_csv(path)
    cols = {c.lower(): c for c in v.columns}
    if "date" not in cols or "ice_fraction" not in cols:
        raise ValueError(
            "validation CSV must have 'date' and 'ice_fraction' columns; "
            f"got {list(v.columns)}")
    v["_d"] = pd.to_datetime(v[cols["date"]])
    return v.set_index("_d")[cols["ice_fraction"]].sort_index()


def _write_feature_figure(df, windows, feat_z, station, path) -> None:
    """Write the per-feature z-score heatmap figure (one panel per ice year)."""
    import matplotlib.pyplot as plt

    n = len(windows)
    fig, axg = plt.subplots(n, 1, figsize=(13, 2.1 * n), squeeze=False,
                            layout="constrained")
    axes = list(axg[:, 0])
    mesh = None
    for ax, (end_year, start, end) in zip(axes, windows):
        m = df[(df["date"] >= start) & (df["date"] <= end)]
        ax.set_xlim(start, end)
        ax.set_xlabel("date")
        ax.set_title(f"{station}  {_season_label(end_year)}", fontsize=10)
        if not m.empty:
            zmat = m[feat_z].to_numpy(dtype=float).T
            mesh = ax.pcolormesh(m["date"].to_numpy(), range(len(feat_z)),
                                 zmat, shading="nearest", cmap="RdBu_r",
                                 vmin=-4, vmax=4)
            ax.set_yticks(range(len(feat_z)))
            ax.set_yticklabels([c[:-2] for c in feat_z], fontsize=7)
            ax.invert_yaxis()
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, location="right", shrink=0.6, pad=0.015,
                     label="feature z-score")
    fig.suptitle(f"{station} — per-feature z-scores", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot: %s", path)


def plot_station(config: StationConfig, out: str | Path | None = None,
                 validation: str | Path | None = None,
                 features: bool = False) -> Path:
    """Render the scored daily series for a station to a PNG.

    One panel per ice year (Jul–Jun): ``mahal_d`` on a log axis, with vertical
    stripes over the open-water months sampled for the baseline, the alarm
    threshold, and above-threshold days marked. With ``validation`` (a generic
    ``date, ice_fraction`` CSV) a ground-truth ice series is overlaid on a
    second axis. With ``features`` a companion ``*_features.png`` is also
    written (a per-feature z-score heatmap). Returns the main figure path.
    Raises :class:`RuntimeError` if matplotlib (the ``[viz]`` extra) is absent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "plotting needs matplotlib — install with: "
            "pip install 'gnssir-ice[viz]'") from exc

    resolver = PathResolver(config)
    mahal_path = resolver.daily_mahal_d()
    if not mahal_path.exists():
        raise FileNotFoundError(
            f"daily_mahal_d not found: {mahal_path} — run the pipeline first")
    df = pd.read_parquet(mahal_path).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    if df.empty:
        raise RuntimeError("daily_mahal_d is empty — nothing to plot")

    threshold = None
    bl_path = resolver.baseline_pkl()
    if bl_path.exists():
        with open(bl_path, "rb") as fh:
            threshold = pickle.load(fh).get("threshold")

    valid = _load_validation(Path(validation)) if validation else None
    ow_months = config.baseline.open_water_months
    ow_years = config.baseline.open_water_years

    windows = _season_windows(df["date"])
    if not windows:                       # data never enters a Jul–Jun window
        windows = [(None, df["date"].min(), df["date"].max())]
    n = len(windows)

    # --- main figure: mahal_d, one panel per ice year ---
    fig, axg = plt.subplots(n, 1, figsize=(13, 2.7 * n), squeeze=False)
    axes = list(axg[:, 0])
    for ax, (end_year, start, end) in zip(axes, windows):
        m = df[(df["date"] >= start) & (df["date"] <= end)]
        below = m[~m["above_threshold"]]
        above = m[m["above_threshold"]]
        for s, e in _sampled_month_spans(start, end, ow_months, ow_years):
            ax.axvspan(s, e, color="#fde9a8", zorder=0)
        ax.plot(m["date"], m["mahal_d"], lw=0.6, color="#999999", alpha=0.7,
                zorder=1)
        ax.scatter(below["date"], below["mahal_d"], s=6, color="#999999",
                   alpha=0.8, zorder=2)
        ax.scatter(above["date"], above["mahal_d"], s=10, color="#d62728",
                   zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(start, end)
        if threshold is not None:
            ax.axhline(threshold, ls="--", lw=0.8, color="#d62728",
                       alpha=0.8, zorder=2)
        ax.set_ylabel("mahal_d  (log)")
        ax.set_xlabel("date")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{config.station}  {_season_label(end_year)}",
                     fontsize=10)
        if valid is not None:
            vv = valid[(valid.index >= start) & (valid.index <= end)]
            if not vv.empty:
                ax2 = ax.twinx()
                ax2.plot(vv.index, vv.values * 100.0, color="#1f77b4", lw=1.4,
                         zorder=1)
                ax2.set_ylim(0, 105)
                ax2.set_ylabel("validation ice %", color="#1f77b4")
                ax2.tick_params(axis="y", colors="#1f77b4")

    handles = [
        Line2D([], [], marker="o", ls="", color="#999999", label="mahal_d"),
        Line2D([], [], marker="o", ls="", color="#d62728",
               label="above threshold"),
        Patch(facecolor="#fde9a8", label="baseline sample months"),
    ]
    if threshold is not None:
        handles.append(Line2D([], [], ls="--", color="#d62728",
                              label="alarm threshold"))
    if valid is not None:
        handles.append(Line2D([], [], color="#1f77b4",
                              label="validation ice %"))
    axes[0].legend(handles=handles, loc="upper right", fontsize=8,
                   framealpha=0.85)
    fig.suptitle(
        f"{config.station} — daily GNSS-IR ice-tracking distance"
        + ("  vs validation" if valid is not None else ""), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    out_path = Path(out) if out else (
        resolver.output_root / f"{config.station}_mahal_d.png")
    resolver.ensure_output_dir()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot: %s", out_path)

    # --- optional companion figure: per-feature z-score heatmap ---
    if features:
        feat_z = [f"{f}_z" for f in MAHAL_FEATURES if f"{f}_z" in df.columns]
        if feat_z:
            feat_path = out_path.with_name(
                f"{out_path.stem}_features{out_path.suffix}")
            _write_feature_figure(df, windows, feat_z, config.station,
                                  feat_path)
        else:
            logger.warning("--features: daily_mahal_d has no *_z columns — "
                            "skipping the feature figure")
    return out_path
