# ABOUTME: Per-arc SNR feature math — CLR/PR/SP/RH, area factor, damping, phase.
# ABOUTME: Pure functions on detrended SNR arcs; no I/O, no config.

"""Per-arc SNR feature extraction.

From one detrended SNR arc this computes the literature-based ice indicators:

  * ``CLR``  — clarity ratio (Purnell 2024): P1 / mean(other LSP peaks)
  * ``PR``   — peak ratio (Purnell 2024): P1 / P2
  * ``AF``   — area factor (Song 2022): integral of the CWT power curve at the
    dominant reflector height
  * ``gamma``— damping parameter (Strandberg 2017): SNR envelope decay rate
  * ``MS``   — mean raw SNR (dB)
  * ``VS``   — variance of detrended SNR
  * ``phase``— reflection phase at the LSP-derived RH

:func:`compute_area_factor` can also return the raw CWT power curve so the
pipeline can persist it once and apply the antenna-gain correction cheaply
later (see :mod:`gnssir_ice.af_gain`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle, find_peaks
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

from gnssir_ice.wavelet import cwt, morlet2

# np.trapz is deprecated in NumPy 2.x in favour of np.trapezoid.
_trapz = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# LSP features
# ---------------------------------------------------------------------------
def compute_lsp_features(sin_elev, detrended, wavelength, min_rh, max_rh,
                         precision):
    """Compute LSP-derived features: CLR, PR, SP, RH.

    CLR/PR/SP are computed on LSP **power** (Purnell 2024 §III convention).

    Returns a dict with keys ``CLR``, ``PR``, ``SP``, ``RH``,
    ``clr_peak_power``, ``clr_total_power``. CLR/PR are NaN when fewer than 2
    LSP peaks are detected (the ratios are then undefined).
    """
    cf = wavelength / 2
    x = sin_elev / cf

    rh_grid = np.arange(min_rh, max_rh + precision, precision)
    angular_freq = 2 * np.pi * rh_grid

    pgram = lombscargle(x, detrended, angular_freq, normalize=False)
    LSP_power = pgram / len(x)

    peak_indices, _ = find_peaks(LSP_power)
    if len(peak_indices) == 0:
        peak_idx = np.argmax(LSP_power)
        return {"CLR": float("nan"), "PR": float("nan"),
                "SP": float(LSP_power[peak_idx]),
                "RH": float(rh_grid[peak_idx]),
                "clr_peak_power": float(LSP_power[peak_idx]),
                "clr_total_power": 0.0}

    peak_powers = LSP_power[peak_indices]
    sorted_idx = np.argsort(peak_powers)[::-1]
    p1_idx = peak_indices[sorted_idx[0]]
    p1 = peak_powers[sorted_idx[0]]

    if len(peak_powers) > 1:
        other_peaks = np.delete(peak_powers, sorted_idx[0])
        clr_total_power = float(np.mean(other_peaks))
        clr = float(p1 / clr_total_power)
    else:
        clr_total_power = 0.0
        clr = float("nan")
    clr_peak_power = float(p1)

    if len(peak_powers) >= 2:
        p2 = peak_powers[sorted_idx[1]]
        pr = float(p1 / p2) if p2 > 0 else float(p1)
    else:
        pr = float("nan")

    return {
        "CLR": clr, "PR": pr, "SP": float(p1),
        "RH": float(rh_grid[p1_idx]),
        "clr_peak_power": clr_peak_power,
        "clr_total_power": clr_total_power,
    }


# ---------------------------------------------------------------------------
# Area factor (Song 2022)
# ---------------------------------------------------------------------------
def integrate_af(power_curve, sin_elev, baseline_power_curve=None,
                 baseline_sin_grid=None):
    """Integrate a CWT power curve into an area factor (Song 2022 Eq. 21).

    With ``baseline_power_curve`` / ``baseline_sin_grid`` supplied, the per-PRN
    open-water antenna-gain power curve is subtracted first (Eq. 23), isolating
    the surface-driven power change. ``power_curve`` is not modified in place.

    Args:
        power_curve: CWT power at the dominant RH, one value per arc sample
        sin_elev: sin(elevation) coordinate of each sample (sorted ascending)
        baseline_power_curve: optional per-PRN baseline power curve
        baseline_sin_grid: sin(ε) grid the baseline is defined on

    Returns:
        Area factor (float).
    """
    pc = np.asarray(power_curve, dtype=float).copy()
    x = np.asarray(sin_elev, dtype=float)

    if baseline_power_curve is not None and baseline_sin_grid is not None:
        bl_interp = interp1d(
            baseline_sin_grid, baseline_power_curve,
            bounds_error=False, fill_value=np.nan,
        )(x)
        valid_bl = ~np.isnan(bl_interp)
        oob_frac = 1.0 - valid_bl.mean() if len(valid_bl) > 0 else 1.0
        if oob_frac <= 0.5 and valid_bl.sum() >= 5:
            pc[valid_bl] = np.maximum(pc[valid_bl] - bl_interp[valid_bl], 0.0)
        # else: baseline domain too narrow — integrate the uncorrected curve

    # Song 2022 Eq. 21: AF = (1/n) · ∫ P_peak(θ) dθ in elevation radians.
    theta_rad = np.arcsin(np.clip(x, -1.0, 1.0))
    n_samples = len(pc)
    return float(_trapz(pc, theta_rad) / n_samples)


def compute_area_factor(sin_elev, detrended, wavelength, min_rh, max_rh,
                        baseline_power_curve=None, baseline_sin_grid=None,
                        return_power=False):
    """Compute the wavelet-derived area factor.

    Runs a Morlet CWT on dSNR(sin ε), extracts the power curve at the dominant
    RH, and integrates it (:func:`integrate_af`).

    Args:
        sin_elev: sin(elevation) array
        detrended: detrended SNR array
        wavelength: carrier wavelength (m)
        min_rh, max_rh: reflector-height search range (m)
        baseline_power_curve, baseline_sin_grid: optional Eq. 23 baseline
        return_power: if True, also return the raw power curve and its sin(ε)
            coordinates so the pipeline can persist them

    Returns:
        ``af`` (float), or ``(af, power_curve, sin_elev_sorted)`` when
        ``return_power``. On failure: ``nan`` or ``(nan, None, None)``.
    """
    sort_idx = np.argsort(sin_elev)
    x = sin_elev[sort_idx]
    y = detrended[sort_idx]

    nan_result = (np.nan, None, None) if return_power else np.nan

    if len(y) < 20:
        return nan_result

    cf = wavelength / 2
    dx = np.mean(np.diff(x / cf))
    if dx <= 0:
        return nan_result

    # Morlet center frequency; scale ↔ RH mapping: scale = w / (2π·RH·dx).
    w = 5.0
    rh_values = np.linspace(min_rh, max_rh, 100)
    scales = w / (2 * np.pi * rh_values * dx)
    valid = scales > 1
    if valid.sum() < 5:
        return nan_result
    scales = scales[valid]

    cwtmatr = cwt(y, morlet2, scales, w=w)
    power = np.abs(cwtmatr) ** 2

    integrated = np.sum(power, axis=1)
    dominant_idx = np.argmax(integrated)
    power_curve = power[dominant_idx, :]

    af = integrate_af(power_curve, x,
                      baseline_power_curve=baseline_power_curve,
                      baseline_sin_grid=baseline_sin_grid)

    if return_power:
        return af, power_curve.copy(), x.copy()
    return af


# ---------------------------------------------------------------------------
# Damping parameter (Strandberg 2017)
# ---------------------------------------------------------------------------
def compute_damping_strandberg(elevation_deg, detrended, wavelength,
                               rh_init, min_rh=None, max_rh=None,
                               gamma_init=1e-3, gamma_max=1.0):
    """Damping γ via the Strandberg 2017 joint nonlinear least-squares fit.

    Fits ``dSNR = [C₁·sin(ω·sin ε) + C₂·cos(ω·sin ε)]·exp(−4k²γ sin²ε)``
    jointly for ``(h, C₁, C₂, γ)`` with ``ω = 4πh/λ``, ``k = 2π/λ``.

    Returns ``(gamma, gamma_r2, h_fit)``; ``(nan, nan, nan)`` on failure.
    """
    nan_result = (np.nan, np.nan, np.nan)
    n = len(detrended)
    if n < 20 or rh_init is None or rh_init <= 0 or not np.isfinite(rh_init):
        return nan_result

    sin_e = np.sin(np.radians(elevation_deg))
    sin2_e = sin_e ** 2
    k = 2 * np.pi / wavelength

    def model(params):
        h, c1, c2, gamma = params
        omega = 4 * np.pi * h / wavelength
        envelope = np.exp(-4 * k ** 2 * gamma * sin2_e)
        return envelope * (c1 * np.sin(omega * sin_e) + c2 * np.cos(omega * sin_e))

    def residuals(params):
        return model(params) - detrended

    omega0 = 4 * np.pi * rh_init / wavelength
    A_lin = np.column_stack([np.sin(omega0 * sin_e), np.cos(omega0 * sin_e)])
    try:
        c0, *_ = np.linalg.lstsq(A_lin, detrended, rcond=None)
        c1_init, c2_init = float(c0[0]), float(c0[1])
    except Exception:
        c1_init, c2_init = float(np.std(detrended)), float(np.std(detrended))

    p0 = [float(rh_init), c1_init, c2_init, float(gamma_init)]

    h_lo = max(0.05, min_rh) if min_rh is not None else 0.05
    h_hi = max_rh if max_rh is not None else float(rh_init) * 3.0
    bounds = (
        [h_lo, -np.inf, -np.inf, 0.0],
        [h_hi, np.inf, np.inf, float(gamma_max)],
    )

    try:
        result = least_squares(residuals, p0, bounds=bounds,
                               method="trf", max_nfev=200)
    except Exception:
        return nan_result

    if not result.success:
        return nan_result

    h_fit, c1_fit, c2_fit, gamma_fit = result.x
    pred = model(result.x)
    ss_res = float(np.sum((detrended - pred) ** 2))
    ss_tot = float(np.sum((detrended - np.mean(detrended)) ** 2))
    gamma_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    gamma_r2 = float(np.clip(gamma_r2, 0.0, 1.0))

    return (float(max(0.0, gamma_fit)), gamma_r2, float(h_fit))


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------
def compute_phase(sin_elev, detrended, wavelength, rh):
    """Reflection phase from detrended SNR at a known RH (matched filter).

    Projects dSNR onto cos/sin references at ω = 4πRH/λ. Returns the phase in
    radians wrapped to [−π, π], or NaN on failure.
    """
    if len(detrended) < 10 or rh <= 0:
        return np.nan

    omega = 4 * np.pi * rh / wavelength
    arg = omega * sin_elev
    cos_ref = np.cos(arg)
    sin_ref = np.sin(arg)

    a = np.sum(detrended * cos_ref)
    b = np.sum(detrended * sin_ref)
    if a == 0 and b == 0:
        return np.nan

    return float(np.arctan2(-b, a))


# ---------------------------------------------------------------------------
# Full per-arc feature extraction
# ---------------------------------------------------------------------------
def extract_arc_features(elevation, snr_db, snr_linear, detrended,
                         wavelength, e1, e2, min_rh, max_rh, precision):
    """Compute all per-arc features for one arc on one frequency.

    AF is computed **uncorrected** here; the antenna-gain correction is applied
    later from the persisted power curve (:mod:`gnssir_ice.af_gain`). For full
    arcs the raw power curve and its sin(ε) coordinates are returned under the
    underscore-prefixed keys ``_af_power_curve`` / ``_af_sin_elev`` so the
    extractor can persist them.

    Returns a feature dict, or None when the windowed arc is too short.
    """
    mask = (elevation >= e1) & (elevation <= e2)
    if mask.sum() < 15:
        return None

    ele_w = elevation[mask]
    sin_e = np.sin(np.radians(ele_w))
    dsnr = detrended[mask]
    snr_db_w = snr_db[mask]

    # Full-arc check (Song 2022): must span close to [e1, e2].
    ele_range = ele_w.max() - ele_w.min()
    expected_range = e2 - e1
    full_arc = ele_range >= 0.8 * expected_range

    lsp = compute_lsp_features(sin_e, dsnr, wavelength, min_rh, max_rh,
                               precision)

    # AF and γ are mechanically biased by truncated elevation ranges, so they
    # are only computed for full arcs (Song 2022 ≥80% coverage).
    af_power_curve = None
    af_sin_elev = None
    if full_arc:
        af, af_power_curve, af_sin_elev = compute_area_factor(
            sin_e, dsnr, wavelength, min_rh, max_rh, return_power=True)
        gamma, gamma_r2, gamma_h_fit = compute_damping_strandberg(
            ele_w, dsnr, wavelength,
            rh_init=lsp["RH"], min_rh=min_rh, max_rh=max_rh)
    else:
        af = np.nan
        gamma = gamma_r2 = gamma_h_fit = np.nan

    phase = compute_phase(sin_e, dsnr, wavelength, lsp["RH"])

    ms = float(np.mean(snr_db_w))
    vs = float(np.var(dsnr, ddof=1)) if len(dsnr) >= 2 else float("nan")

    return {
        "CLR": lsp["CLR"], "PR": lsp["PR"], "SP": lsp["SP"], "RH": lsp["RH"],
        "clr_peak_power": lsp["clr_peak_power"],
        "clr_total_power": lsp["clr_total_power"],
        "AF": af, "gamma": gamma, "gamma_r2": gamma_r2,
        "gamma_h_fit": gamma_h_fit, "phase": phase, "MS": ms, "VS": vs,
        "full_arc": full_arc,
        "_af_power_curve": af_power_curve,
        "_af_sin_elev": af_sin_elev,
    }
