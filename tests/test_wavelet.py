# ABOUTME: Tests for the hand-rolled Morlet CWT — shape, scaling, localization.

import numpy as np

from gnssir_ice.wavelet import cwt, morlet2


def test_morlet2_length_and_dtype():
    w = morlet2(50, 5.0, 4.0)
    assert w.shape == (50,)
    assert np.iscomplexobj(w)


def test_morlet2_sqrt_inv_s_normalization():
    # The sqrt(1/s) amplitude term: a larger scale gives a smaller peak.
    big = np.abs(morlet2(120, 5.0, 2.0)).max()
    small = np.abs(morlet2(120, 5.0, 8.0)).max()
    assert big > small


def test_cwt_shape():
    data = np.random.RandomState(0).randn(200)
    out = cwt(data, morlet2, np.array([3.0, 10.0, 40.0]), w=5)
    assert out.shape == (3, 200)
    assert np.iscomplexobj(out)


def test_cwt_accepts_fractional_widths():
    # M = min(10*width, N) is passed straight through (not int-cast); a
    # fractional width must not raise.
    data = np.random.RandomState(1).randn(150)
    out = cwt(data, morlet2, np.array([3.7, 7.3]), w=5)
    assert out.shape == (2, 150)
    assert np.all(np.isfinite(out))


def test_cwt_localizes_a_tone():
    # A pure tone should produce the most power at the scale matching it.
    n = 600
    t = np.arange(n)
    period = 30.0
    signal = np.cos(2 * np.pi * t / period)
    w = 5.0
    # For a Morlet of angular frequency w, scale ≈ period * w / (2π).
    widths = np.linspace(5, 60, 40)
    power = np.abs(cwt(signal, morlet2, widths, w=w)) ** 2
    best = widths[np.argmax(power.sum(axis=1))]
    expected = period * w / (2 * np.pi)
    assert abs(best - expected) < 8.0
