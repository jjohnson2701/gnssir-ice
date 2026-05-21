# ABOUTME: Opt-in end-to-end reproduction check against real station output.

"""Regression check — opt-in (`pytest -m regression`).

Reproduces a real station's ``daily_mahal_d`` and compares it, innermost-first
(arc_features → daily_features → mahal_d), to a saved reference. Skipped unless
the reference fixtures are present, so it never runs in CI.

To enable it, drop a reference snapshot under ``tests/fixtures/regression/``:
``ross_regression.yaml`` plus the reference ``arc_features`` / ``daily_features``
/ ``daily_mahal_d`` files, then run ``pytest -m regression``.
"""

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "regression"
CONFIG = FIXTURE_DIR / "ross_regression.yaml"

pytestmark = pytest.mark.regression


@pytest.mark.skipif(not CONFIG.exists(),
                    reason="regression fixtures not present")
def test_reproduces_reference_mahal_d():
    import pandas as pd

    from gnssir_ice import StationConfig, run_pipeline

    cfg = StationConfig.load(CONFIG)
    result = run_pipeline(cfg, force=True)

    produced = pd.read_parquet(result["daily_mahal_d"]).set_index("date")
    reference = pd.read_parquet(
        FIXTURE_DIR / "ross_daily_mahal_d_reference.parquet").set_index("date")
    common = produced.index.intersection(reference.index)
    assert len(common) > 0

    rel = (produced.loc[common, "mahal_d"]
           - reference.loc[common, "mahal_d"]).abs() / \
        reference.loc[common, "mahal_d"].abs()
    assert (rel < 1e-6).mean() >= 0.99
