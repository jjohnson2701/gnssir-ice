# ABOUTME: Command-line interface — the only module with argparse / __main__.
# ABOUTME: `gnssir-ice <subcommand>` walks a station through the pipeline.

"""``gnssir-ice`` command-line interface.

One console entry point with subcommands for the whole pipeline (``run``) and
for each individual stage. Run ``gnssir-ice --help`` or
``gnssir-ice <subcommand> --help`` for usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gnssir_ice import __version__
from gnssir_ice.config import StationConfig

logger = logging.getLogger("gnssir_ice")

_CONFIG_TEMPLATE = """\
# gnssir-ice station configuration — edit the values below.
station: {station}

coordinates:                     # optional provenance — unused by the model
  latitude_deg: 0.0
  longitude_deg: 0.0
  ellipsoidal_height_m: 0.0      # antenna height; only adds the diagnostic wse

gnssir:                          # mirror the gnssrefl station JSON
  e1: 5.0                        # min elevation (deg)
  e2: 25.0                       # max elevation (deg)
  minH: 2.0                      # reflector-height search min (m)
  maxH: 8.0                      # reflector-height search max (m)
  polyV: 4                       # detrend polynomial order
  pele: [5, 30]                  # poly-fit elevation window (deg)
  desiredP: 0.005                # LSP RH grid step (m)

baseline:
  open_water_months: [7, 8]              # months pooled for the Mahalanobis fit
  normalization_months: [6, 7, 8, 9, 10] # per-PRN / AF-baseline reference window
  open_water_years: [2022, 2023, 2024, 2025]
  pca_variance: 0.95
  threshold_percentile: 99
  af_baseline_max_arcs: null     # cap per AF baseline (null = use all arcs)
  sector_width_deg: 30           # azimuth-sector width for per-sector scoring
  min_sector_baseline_days: 20   # min open-water days for a sector baseline
  ice_season_min_run_days: 5     # min consecutive above-threshold days/season

processing:
  years: [2022, 2023, 2024, 2025]

paths:
  refl_code: "${{REFL_CODE}}"    # gnssrefl root (snr + subdaily resolve under it)
  snr_filename: "{{station_lower}}{{doy}}0.{{yy}}.snr66"
  snr_dir: "{{refl_code}}/{{year}}/snr/{{station_lower}}"
  gnssir_output: "{{refl_code}}/{{year}}/results/{{station_lower}}"
  subdaily_file: "{{refl_code}}/Files/{{station_lower}}/{{station_lower}}_{{year}}_subdaily_edit.txt.withrhdotIF"
  output_root: "./results/{{station}}"

options:
  per_prn_normalization: true
  af_baseline: true
  min_subdaily_match_rate: 0.5   # build-arc-table fails below this match rate
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_years(tokens: list[str] | None) -> list[int] | None:
    """Parse ``--years`` tokens: explicit ints or a ``START-END`` range."""
    if not tokens:
        return None
    years: list[int] = []
    for tok in tokens:
        if "-" in tok and not tok.startswith("-"):
            lo, hi = tok.split("-", 1)
            years.extend(range(int(lo), int(hi) + 1))
        else:
            years.append(int(tok))
    return sorted(set(years))


def _load(args) -> StationConfig:
    return StationConfig.load(args.config)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------
def _cmd_init_config(args) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        logger.error("%s exists (use --force to overwrite)", out)
        return 1
    out.write_text(_CONFIG_TEMPLATE.format(station=args.station))
    logger.info("wrote config scaffold: %s", out)
    return 0


def _cmd_run(args) -> int:
    from gnssir_ice.pipeline import run_pipeline
    cfg = _load(args)
    if args.no_af_baseline:
        cfg.options.af_baseline = False
    if args.no_prn_norm:
        cfg.options.per_prn_normalization = False
    result = run_pipeline(cfg, years=_parse_years(args.years),
                          jobs=args.jobs, force=args.force)
    logger.info("done — baseline: %s", result["baseline"])
    logger.info("done — daily_mahal_d: %s", result["daily_mahal_d"])
    logger.info("done — ice_seasons: %s", result["ice_seasons"])
    return 0


def _cmd_consolidate_snr(args) -> int:
    from gnssir_ice.snr import consolidate_snr
    consolidate_snr(_load(args), args.year, force=args.force)
    return 0


def _cmd_build_arc_table(args) -> int:
    from gnssir_ice.arc_table import build_arc_table
    build_arc_table(_load(args), args.year, force=args.force)
    return 0


def _cmd_extract(args) -> int:
    from gnssir_ice.extract import extract_station_year
    extract_station_year(_load(args), args.year, jobs=args.jobs, force=args.force)
    return 0


def _cmd_af_baseline(args) -> int:
    from gnssir_ice.af_gain import build_af_baseline
    build_af_baseline(_load(args), force=args.force)
    return 0


def _cmd_af_correct(args) -> int:
    from gnssir_ice.af_gain import apply_af_correction
    apply_af_correction(_load(args), args.year, force=args.force)
    return 0


def _cmd_prn_baseline(args) -> int:
    from gnssir_ice.normalize import build_prn_baseline
    build_prn_baseline(_load(args), force=args.force)
    return 0


def _cmd_normalize(args) -> int:
    from gnssir_ice.normalize import apply_prn_normalization
    apply_prn_normalization(_load(args), args.year, force=args.force)
    return 0


def _cmd_aggregate(args) -> int:
    from gnssir_ice.aggregate import aggregate_station_year
    aggregate_station_year(_load(args), args.year, force=args.force)
    return 0


def _cmd_fit_baseline(args) -> int:
    from gnssir_ice.model import fit_station_baseline
    fit_station_baseline(_load(args), force=args.force)
    return 0


def _cmd_score(args) -> int:
    from gnssir_ice.model import score_station
    score_station(_load(args), years=_parse_years(args.years))
    return 0


def _cmd_ice_seasons(args) -> int:
    from gnssir_ice.seasons import extract_ice_seasons
    extract_ice_seasons(_load(args), force=args.force)
    return 0


def _cmd_plot(args) -> int:
    from gnssir_ice.plot import plot_station
    plot_station(_load(args), out=args.out, validation=args.validation,
                 features=args.features)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gnssir-ice",
        description="GNSS-IR ice features tracked against an open-water baseline.")
    p.add_argument("--version", action="version", version=f"gnssir-ice {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp, *, config=True):
        if config:
            sp.add_argument("--config", required=True, help="station config (YAML/JSON)")
        sp.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sp = sub.add_parser("init-config", help="scaffold a blank station config")
    sp.add_argument("--station", required=True, help="station id (e.g. ROSS)")
    sp.add_argument("--out", required=True, help="output config path")
    sp.add_argument("--force", action="store_true")
    _common(sp, config=False)
    sp.set_defaults(func=_cmd_init_config)

    sp = sub.add_parser("run", help="run the full pipeline")
    _common(sp)
    sp.add_argument("--years", nargs="+", help="years or a START-END range")
    sp.add_argument("--jobs", type=int, default=1, help="extraction workers")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--no-af-baseline", action="store_true",
                    help="skip the antenna-gain AF correction (config override)")
    sp.add_argument("--no-prn-norm", action="store_true",
                    help="skip per-PRN normalization (config override)")
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("consolidate-snr", help="Stage 1 — consolidate SNR files")
    _common(sp)
    sp.add_argument("--year", type=int, required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_consolidate_snr)

    sp = sub.add_parser("build-arc-table", help="Stage 2 — build the arc table")
    _common(sp)
    sp.add_argument("--year", type=int, required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_build_arc_table)

    sp = sub.add_parser("extract", help="Stage 3 — extract per-arc features")
    _common(sp)
    sp.add_argument("--year", type=int, required=True)
    sp.add_argument("--jobs", type=int, default=1)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_extract)

    sp = sub.add_parser("af-baseline",
                        help="Stage 4 — build the pooled AF baseline")
    _common(sp)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_af_baseline)

    sp = sub.add_parser("af-correct", help="Stage 5 — apply the AF correction")
    _common(sp)
    sp.add_argument("--year", type=int, required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_af_correct)

    sp = sub.add_parser("prn-baseline", help="Stage 6 — build the per-PRN baseline")
    _common(sp)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_prn_baseline)

    sp = sub.add_parser("normalize", help="Stage 7 — apply per-PRN normalization")
    _common(sp)
    sp.add_argument("--year", type=int, required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_normalize)

    sp = sub.add_parser("aggregate", help="Stage 8 — aggregate daily features")
    _common(sp)
    sp.add_argument("--year", type=int, required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_aggregate)

    sp = sub.add_parser("fit-baseline", help="Stage 9 — fit the open-water baseline")
    _common(sp)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_fit_baseline)

    sp = sub.add_parser("score", help="Stage 10 — score days against the baseline")
    _common(sp)
    sp.add_argument("--years", nargs="+", help="years or a START-END range")
    sp.set_defaults(func=_cmd_score)

    sp = sub.add_parser("ice-seasons",
                        help="Stage 11 — extract freeze-up/break-up dates")
    _common(sp)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=_cmd_ice_seasons)

    sp = sub.add_parser("plot",
                        help="plot the scored mahal_d series (needs [viz] extra)")
    _common(sp)
    sp.add_argument("--out", help="output PNG path (default: the results dir)")
    sp.add_argument("--validation",
                    help="validation CSV (date, ice_fraction) to overlay")
    sp.add_argument("--features", action="store_true",
                    help="add a per-feature z-score heatmap strip per panel")
    sp.set_defaults(func=_cmd_plot)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
