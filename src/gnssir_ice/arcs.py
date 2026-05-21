# ABOUTME: Satellite-arc segmentation, arc matching, and polynomial detrending.
# ABOUTME: Splits an SNR record into rise/set arcs and matches them to retrievals.

"""Arc geometry helpers.

An SNR file holds every observation of a day; these functions split one
satellite's stream into individual rise/set arcs (:func:`segment_satellite_arcs`),
match a segmented arc to a gnssrefl retrieval (:func:`find_matching_segment`),
and remove the direct-signal polynomial trend (:func:`detrend_arc`).
"""

from __future__ import annotations

import numpy as np


def segment_satellite_arcs(seconds_of_day, elevation, gap_seconds=300):
    """Segment one satellite's observations into individual arcs.

    Splits on time gaps > ``gap_seconds`` OR elevation direction reversals.

    Returns a list of dicts, each with ``start_idx``, ``end_idx`` (one past
    last, for slicing), and ``rise`` (1 if elevation rising, -1 if setting).
    """
    n = len(seconds_of_day)
    if n < 2:
        return []

    last_dir = 0  # 1=rising, -1=setting, 0=undetermined

    splits = [0]
    for i in range(1, n):
        dt = seconds_of_day[i] - seconds_of_day[i - 1]
        if dt > gap_seconds or dt < 0:
            splits.append(i)
            last_dir = 0
            continue

        d = elevation[i] - elevation[i - 1]
        if abs(d) < 0.01:
            continue  # skip near-zero changes (apex plateau)

        cur_dir = 1 if d > 0 else -1
        if last_dir != 0 and cur_dir != last_dir:
            splits.append(i)
        last_dir = cur_dir

    arcs = []
    for j in range(len(splits)):
        start = splits[j]
        end = splits[j + 1] if j + 1 < len(splits) else n
        if end - start < 5:
            continue

        seg_ele = elevation[start:end]
        net = seg_ele[-1] - seg_ele[0]
        rise = 1 if net > 0 else -1

        arcs.append({"start_idx": start, "end_idx": end, "rise": rise})
    return arcs


def find_matching_segment(arcs, seconds_of_day, elevation,
                          target_utctime, target_rise, e1, e2):
    """Find which segmented arc matches a gnssrefl retrieval.

    Args:
        arcs: list of dicts from :func:`segment_satellite_arcs`
        seconds_of_day: full SOD array for this satellite
        elevation: full elevation array for this satellite
        target_utctime: UTCtime (hours) from the retrieval
        target_rise: rise direction (1 or -1) from the retrieval
        e1, e2: elevation window for the windowed-mean-time comparison

    Returns:
        Index into ``arcs``, or -1 if no match within 1 hour.
    """
    best_idx = -1
    best_dt = float("inf")

    for i, arc in enumerate(arcs):
        if arc["rise"] != target_rise:
            continue

        seg_sod = seconds_of_day[arc["start_idx"]:arc["end_idx"]]
        seg_ele = elevation[arc["start_idx"]:arc["end_idx"]]

        mask = (seg_ele >= e1) & (seg_ele <= e2)
        if mask.sum() < 3:
            continue

        windowed_sod = seg_sod[mask]
        mean_utc_hours = np.mean(windowed_sod) / 3600.0

        dt = abs(mean_utc_hours - target_utctime)
        if dt > 12:  # midnight wrap
            dt = 24 - dt

        if dt < best_dt:
            best_dt = dt
            best_idx = i

    if best_dt > 1.0:
        return -1
    return best_idx


def detrend_arc(elevation, snr_linear, poly_order=4, pele=(5, 30)):
    """Remove the polynomial direct-signal trend from an SNR arc.

    Args:
        elevation: elevation angles (degrees)
        snr_linear: SNR in linear units (10**(dB/20))
        poly_order: polynomial order for the trend fit
        pele: (min, max) elevation range used for the trend fit

    Returns:
        Detrended SNR array (same length as input).
    """
    pele_min, pele_max = pele
    fit_mask = (elevation >= pele_min) & (elevation <= pele_max)
    if fit_mask.sum() < poly_order + 1:
        return snr_linear - np.mean(snr_linear)

    coeffs = np.polyfit(elevation[fit_mask], snr_linear[fit_mask], poly_order)
    trend = np.polyval(coeffs, elevation)
    return snr_linear - trend
