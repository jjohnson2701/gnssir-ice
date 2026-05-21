# ABOUTME: Shared constants — GNSS frequency maps, column schemas, feature lists.
# ABOUTME: No I/O, no config; pure lookup tables used across the pipeline.

"""Constants for the gnssir-ice pipeline.

Frequency-code conventions follow gnssrefl. SNR-file column layout follows the
gnssrefl ``snr66`` format. The 10-feature model schema (:data:`MAHAL_FEATURES`)
is the contract between :mod:`gnssir_ice.aggregate` and :mod:`gnssir_ice.model`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GNSS frequency code → SNR-file column index (0-based)
# gnssrefl snr66 columns: sat, elev, azim, sod, edot, S6, S1, S2, S5, S7, S8
# ---------------------------------------------------------------------------
FREQ_TO_COL = {
    1: 6, 101: 6, 201: 6, 301: 6,        # L1  → column 6
    2: 7, 20: 7, 102: 7, 302: 7,         # L2C → column 7
    5: 8, 205: 8,                        # L5  → column 8
    206: 5, 306: 5,                      # E5b/B3 → column 5
    207: 9, 307: 9,                      # E5  → column 9
    208: 10,                             # E6  → column 10
}

# GNSS carrier wavelengths in meters.
FREQ_WAVELENGTH = {
    1: 0.19029, 101: 0.19029, 201: 0.19029, 301: 0.19029,   # L1
    2: 0.24421, 20: 0.24421, 102: 0.24421, 302: 0.24421,    # L2
    5: 0.25482, 205: 0.25482,                               # L5
    206: 0.24834, 306: 0.24834,                             # E5b
    207: 0.25478, 307: 0.25478,                             # E5
    208: 0.23405,                                           # E6
}

# ---------------------------------------------------------------------------
# Column schemas
# ---------------------------------------------------------------------------
# gnssrefl ``snr66`` SNR-file columns (11, 0-based order).
SNR66_COLUMNS = [
    "sat", "elev", "azim", "sod", "edot",
    "S6", "S1", "S2", "S5", "S7", "S8",
]

# gnssrefl ``gnssir`` per-arc output columns (17, fixed order).
GNSSIR_COLUMNS = [
    "year",       # (1)  year
    "doy",        # (2)  day of year
    "RH",         # (3)  reflector height (m)
    "sat",        # (4)  satellite number
    "UTCtime",    # (5)  UTC time (hours)
    "Azim",       # (6)  azimuth (deg)
    "Amp",        # (7)  amplitude (v/v)
    "eminO",      # (8)  min elevation observed (deg)
    "emaxO",      # (9)  max elevation observed (deg)
    "NumbOf",     # (10) number of values used
    "freq",       # (11) frequency code
    "rise",       # (12) rising (1) / setting (-1)
    "EdotF",      # (13) edot/F (hours)
    "PkNoise",    # (14) peak-to-noise ratio
    "DelT",       # (15) delta-T (minutes)
    "MJD",        # (16) modified Julian date
    "refr_model", # (17) refraction model (0 = none)
]

# Arc-table join keys — uniquely identify one (arc, frequency) retrieval.
ARC_JOIN_KEYS = ["doy", "sat", "UTCtime", "rise", "freq"]

# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------
# The 10-feature daily model schema consumed by the Mahalanobis baseline.
MAHAL_FEATURES = [
    "amp_mean", "rh_std_raw", "rh_std_norm", "p2n_mean", "clr_med",
    "pr_med", "af_med", "gamma_med", "ms_mean", "vs_mean",
]

# Per-arc feature columns that get a per-(PRN, signal) z-score baseline.
PRN_BASELINE_FEATURES = [
    "Amp", "PkNoise", "CLR", "PR", "AF", "gamma", "MS", "VS", "RH",
]

# γ-fit R² gate: arcs whose joint-NLS damping fit is worse than this are
# dropped from gamma_med (their γ is unreliable).
GAMMA_R2_MIN = 0.3

# Number of points in the common sin(ε) grid used for AF power-curve baselines.
SIN_GRID_SIZE = 50

# ---------------------------------------------------------------------------
# gnssrefl output contract
# ---------------------------------------------------------------------------
# gnssrefl version gnssir-ice's file-format parsing was verified against.
TESTED_GNSSREFL_VERSION = "3.19.3"

# subdaily RHdot/IF match rate below which build-arc-table emits a (non-fatal)
# warning — every healthy station-year measured sits at 0.87 or above.
SUBDAILY_MATCH_WARN = 0.85
