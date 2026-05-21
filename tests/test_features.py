# ABOUTME: Tests for per-arc feature math against a synthetic clean SNR arc.

import numpy as np

from gnssir_ice.features import (compute_area_factor, compute_damping_strandberg,
                                 compute_lsp_features, compute_phase,
                                 extract_arc_features, integrate_af)
from tests.conftest import make_arc

WAVELENGTH = 0.19029


def test_lsp_recovers_known_rh():
    rh_true = 4.20
    ele, dsnr = make_arc(rh=rh_true, gamma=0.0, phase=0.0)
    sin_e = np.sin(np.radians(ele))
    lsp = compute_lsp_features(sin_e, dsnr, WAVELENGTH, 2.0, 8.0, 0.005)
    assert abs(lsp["RH"] - rh_true) < 0.05
    assert lsp["SP"] > 0


def test_phase_recovers_known_phase():
    phase_true = 0.7
    ele, dsnr = make_arc(rh=4.2, gamma=0.0, phase=phase_true)
    sin_e = np.sin(np.radians(ele))
    phase = compute_phase(sin_e, dsnr, WAVELENGTH, rh=4.2)
    assert abs(phase - phase_true) < 0.05


def test_damping_recovers_known_gamma():
    gamma_true = 0.006
    ele, dsnr = make_arc(rh=4.2, gamma=gamma_true, phase=0.3)
    gamma, gamma_r2, h_fit = compute_damping_strandberg(
        ele, dsnr, WAVELENGTH, rh_init=4.2, min_rh=2.0, max_rh=8.0)
    assert gamma_r2 > 0.95
    assert abs(gamma - gamma_true) < 0.002
    assert abs(h_fit - 4.2) < 0.05


def test_area_factor_returns_power_curve():
    ele, dsnr = make_arc(rh=4.2, gamma=0.004, phase=0.3)
    sin_e = np.sin(np.radians(ele))
    af, curve, x = compute_area_factor(
        sin_e, dsnr, WAVELENGTH, 2.0, 8.0, return_power=True)
    assert np.isfinite(af) and af > 0
    assert curve.shape == x.shape
    assert np.all(np.diff(x) >= 0)  # sin(elev) sorted ascending


def test_integrate_af_matches_compute_area_factor():
    ele, dsnr = make_arc(rh=4.2, gamma=0.004, phase=0.3)
    sin_e = np.sin(np.radians(ele))
    af, curve, x = compute_area_factor(
        sin_e, dsnr, WAVELENGTH, 2.0, 8.0, return_power=True)
    # Re-integrating the persisted curve must reproduce the AF exactly.
    assert integrate_af(curve, x) == af


def test_extract_arc_features_full_arc():
    ele, dsnr = make_arc(rh=4.2, gamma=0.004, phase=0.3)
    snr_lin = dsnr + 100.0
    snr_db = 20 * np.log10(np.abs(snr_lin))
    feats = extract_arc_features(ele, snr_db, snr_lin, dsnr,
                                 WAVELENGTH, 5.0, 25.0, 2.0, 8.0, 0.005)
    assert feats is not None
    assert feats["full_arc"] is True or feats["full_arc"] == np.True_
    for key in ("CLR", "PR", "AF", "gamma", "phase", "MS", "VS"):
        assert key in feats
    assert feats["_af_power_curve"] is not None
    assert np.isfinite(feats["AF"])


def test_extract_arc_features_short_arc_returns_none():
    ele = np.linspace(5, 25, 10)
    dsnr = np.zeros(10)
    assert extract_arc_features(ele, dsnr, dsnr, dsnr,
                                WAVELENGTH, 5.0, 25.0, 2.0, 8.0, 0.005) is None
