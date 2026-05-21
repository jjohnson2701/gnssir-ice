#!/usr/bin/env python3
# ABOUTME: Builds the opt-in regression fixture from a real station config.

"""Build the opt-in regression fixture from a real station config.

The regression test (``pytest -m regression``) re-runs the full pipeline and
checks it still reproduces a saved reference. That reference is not committed —
generate it on a machine that has the gnssrefl inputs:

    python tests/fixtures/make_regression_fixture.py --config ross.yaml

This copies the config to ``tests/fixtures/regression/<station>_regression.yaml``
and snapshots the produced ``daily_mahal_d`` as the reference parquet. The
config keeps its ``${REFL_CODE}``-relative input paths, so the regression test
re-runs against the same real data. ``tests/fixtures/regression/`` is
gitignored — the fixture stays on the machine that has the data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "regression"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the gnssir-ice regression fixture.")
    ap.add_argument("--config", required=True,
                    help="a real station config (YAML/JSON) with valid inputs")
    args = ap.parse_args(argv)

    # gnssir_ice must be importable (pip install -e . or PYTHONPATH=src).
    from gnssir_ice import StationConfig, run_pipeline

    cfg_path = Path(args.config)
    config = StationConfig.load(cfg_path)
    station = config.station.lower()

    result = run_pipeline(config, force=True)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dst = FIXTURE_DIR / f"{station}_regression.yaml"
    ref_dst = FIXTURE_DIR / f"{station}_daily_mahal_d_reference.parquet"
    shutil.copyfile(cfg_path, cfg_dst)
    shutil.copyfile(result["daily_mahal_d"], ref_dst)

    print(f"wrote {cfg_dst}")
    print(f"wrote {ref_dst}")
    print("\nregression fixture ready — run:  pytest -m regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
