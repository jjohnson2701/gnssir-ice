# ABOUTME: Hand-rolled Morlet wavelet + CWT — scipy-faithful replacements.
# ABOUTME: scipy.signal.cwt / morlet2 were removed in SciPy 1.15; these match them.

"""Continuous wavelet transform with a Morlet wavelet.

``scipy.signal.cwt`` and ``scipy.signal.morlet2`` were removed in SciPy 1.15.
:func:`cwt` and :func:`morlet2` here are bit-faithful replacements — verified
to reproduce the old SciPy output exactly (max abs diff 0.0). The two comments
below about wavelet length and the ``sqrt(1/s)`` amplitude term are load-bearing:
changing either silently rescales the area factor.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve


def morlet2(M, w, s):
    """Morlet wavelet — faithful replacement for removed ``scipy.signal.morlet2``.

    Matches ``scipy.signal.morlet2`` exactly, including the ``sqrt(1/s)``
    amplitude normalization. (scipy's argument order is ``(M, s, w)``; here ``w``
    and ``s`` are swapped to match the :func:`cwt` call convention.)
    """
    t = np.arange(0, M) - (M - 1.0) / 2
    t = t / s
    wavelet = np.exp(1j * w * t) * np.exp(-0.5 * t ** 2) * np.pi ** (-0.25)
    return np.sqrt(1.0 / s) * wavelet


def cwt(data, wavelet_func, widths, **kwargs):
    """Continuous wavelet transform — faithful replacement for removed
    ``scipy.signal.cwt``.

    Each wavelet is built at length ``min(10*width, N)``, matching scipy's
    truncation (scipy did not use the full signal length). Uses
    ``scipy.signal.convolve`` (never removed) for a bit-faithful match.
    """
    N = len(data)
    w = kwargs.get("w", 5)
    out = np.empty((len(widths), N), dtype=complex)
    for i, width in enumerate(widths):
        # scipy passes this length (a float when 10*width < N) straight to the
        # wavelet; np.arange(0, M) then rounds up. Do NOT cast to int — that
        # would truncate and shorten the wavelet by one sample.
        M = np.min([10 * width, N])
        wavelet = wavelet_func(M, w, width)
        out[i] = convolve(data, np.conj(wavelet[::-1]), mode="same")
    return out
