# ABOUTME: Package root — version, StationConfig, and a lazy run_pipeline export.
# ABOUTME: `import gnssir_ice` then gnssir_ice.run_pipeline / StationConfig.

"""gnssir-ice — GNSS-IR ice features tracked against an open-water baseline.

Run gnssrefl (``rinex2snr`` → ``gnssir`` → ``subdaily``) first, then point
:func:`gnssir_ice.run_pipeline` at a :class:`~gnssir_ice.config.StationConfig`
to produce the 10 daily SNR features and a daily Mahalanobis distance from the
open-water baseline.
"""

from __future__ import annotations

import os as _os
from typing import TYPE_CHECKING

from gnssir_ice.config import StationConfig

# gnssir-ice parallelises the extraction stage with multiprocessing (one worker
# per --jobs). Letting numpy/scipy BLAS also thread inside every worker
# oversubscribes the CPU — jobs x n_cores threads — and thrashes. Pin the
# BLAS/OpenMP pools to one thread per process here, before any stage module
# imports numpy, so --jobs is the single, honest width knob. An explicit env
# var still wins (setdefault).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "0.1.0"

__all__ = ["StationConfig", "run_pipeline", "__version__"]

if TYPE_CHECKING:  # for type-checkers only — never imported at runtime
    from gnssir_ice.pipeline import run_pipeline


def __getattr__(name: str):
    """Lazily expose ``run_pipeline``.

    Keeps a bare ``import gnssir_ice`` (and ``gnssir-ice --version``) from
    pulling in the heavy pipeline stack (scipy / scikit-learn).
    """
    if name == "run_pipeline":
        from gnssir_ice.pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
