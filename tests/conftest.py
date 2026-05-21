# ABOUTME: Shared test fixtures — synthetic arcs and station configs.
# ABOUTME: Lets the core suite run with no external data.

"""Pytest fixtures and synthetic-data helpers for the gnssir-ice test suite."""

from __future__ import annotations

import numpy as np
import pytest

from gnssir_ice.config import StationConfig


# ---------------------------------------------------------------------------
# Synthetic SNR arc
# ---------------------------------------------------------------------------
def make_arc(rh=4.2, gamma=0.004, phase=0.6, amp=80.0, wavelength=0.19029,
             e1=5.0, e2=25.0, n=400, noise=0.0, seed=0):
    """Return ``(elevation_deg, detrended)`` for a clean synthetic SNR arc.

    The arc follows the Strandberg model
    ``dSNR = A·exp(-4k²γ sin²ε)·cos(4πRH/λ·sinε + φ)``.
    """
    ele = np.linspace(e1, e2, n)
    sin_e = np.sin(np.radians(ele))
    k = 2 * np.pi / wavelength
    omega = 4 * np.pi * rh / wavelength
    envelope = np.exp(-4 * k ** 2 * gamma * sin_e ** 2)
    dsnr = amp * envelope * np.cos(omega * sin_e + phase)
    if noise:
        dsnr = dsnr + np.random.RandomState(seed).normal(0, noise, n)
    return ele, dsnr


# ---------------------------------------------------------------------------
# Station config
# ---------------------------------------------------------------------------
def config_dict(tmp_path, **overrides):
    """A minimal valid station-config mapping rooted at ``tmp_path``."""
    data = {
        "station": "TEST",
        "coordinates": {
            "latitude_deg": 48.0, "longitude_deg": -87.0,
            "ellipsoidal_height_m": 150.0,
        },
        "gnssir": {
            "e1": 5.0, "e2": 25.0, "minH": 2.0, "maxH": 8.0,
            "polyV": 4, "pele": [5, 30], "desiredP": 0.005,
        },
        "baseline": {
            "open_water_months": [7, 8],
            "normalization_months": [6, 7, 8, 9, 10],
            "open_water_years": [2024],
            "pca_variance": 0.95,
            "threshold_percentile": 99,
        },
        "processing": {"years": [2024]},
        "paths": {
            "refl_code": str(tmp_path / "refl_code"),
            "snr_filename": "{station_lower}{doy}0.{yy}.snr66",
            "snr_dir": "{refl_code}/{year}/snr/{station_lower}",
            "gnssir_output": "{refl_code}/{year}/results/{station_lower}",
            "subdaily_file": ("{refl_code}/Files/{station_lower}/"
                              "{station_lower}_{year}_subdaily_edit.txt.withrhdotIF"),
            "output_root": str(tmp_path / "results" / "{station}"),
        },
        "options": {"per_prn_normalization": True, "af_baseline": True},
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(data.get(key), dict):
            data[key].update(val)
        else:
            data[key] = val
    return data


@pytest.fixture
def station_config(tmp_path):
    """A valid :class:`StationConfig` with all paths under ``tmp_path``."""
    return StationConfig.from_dict(config_dict(tmp_path))
