# Feature provenance & literature adaptations

gnssir-ice scores each day from **10 features** aggregated from per-arc SNR
observables. Those observables come from the GNSS-IR ice-detection literature,
but gnssir-ice **adapts** the published methods rather than reproducing them.
This document records, per feature, where it first appears, and — for each of
the three cited methods — exactly where gnssir-ice follows the paper and where
it diverges, with the rationale and the measured effect.

## The 10 features and where they come from

| Feature | Observable | First appears / source |
|---|---|---|
| `amp_mean` | mean LSP amplitude | the SNR-interferogram amplitude — a standard GNSS-IR observable (Strandberg 2017 retrieves it as `A/C₁/C₂`; Song 2022 and Purnell 2024 both list it) |
| `rh_std_raw`, `rh_std_norm` | spread of reflector height (RH) | RH is the foundational GNSS-IR observable (Larson-lineage SNR retrieval, produced here by gnssrefl). Using the *within-day spread* of RH as an ice indicator is this project's own construction |
| `p2n_mean` | mean peak-to-noise | concept from Purnell 2024's **PTN**; the value gnssir-ice uses is gnssrefl's `PkNoise` — a sibling metric with a different (off-peak) noise denominator, **not** Purnell's PTN equation |
| `clr_med` | clarity ratio, P1 / mean(other LSP peaks) | used by Purnell 2024; the project's design notes attribute the original CLR formulation to **Kim et al. (2021)** |
| `pr_med` | peak ratio, P1 / P2 | Purnell 2024 |
| `af_med` | area factor (CWT power-curve integral) | introduced by **Song 2022**; also adopted by Purnell 2024 |
| `gamma_med` | SNR-envelope damping γ | **Strandberg 2017** (nonlinear-least-squares inversion of detrended SNR) |
| `ms_mean` | mean raw SNR | Purnell 2024 (`MS`) |
| `vs_mean` | variance of detrended SNR | Purnell 2024 (`VS`) |

## How gnssir-ice adapts the cited methods

### Strandberg et al. (2017) — damping γ → `gamma_med`

**What the paper did.** γ is extracted by nonlinear-least-squares inversion of
detrended SNR, fitting
`δSNR = [C₁·sin(4πh/λ·sinε) + C₂·cos(4πh/λ·sinε)]·exp(−4k²γ·sin²ε)`.
Strandberg fits it over a **3-day sliding window** (≈1 parameter per 24 h per
GNSS/frequency) for stable daily values, and normalises γ to an ice-free
reference (`γ_rel`, ≈0.92–1.09 over open water).

**What gnssir-ice does.** The same model form (`compute_damping_strandberg`),
but fit **per individual arc** — not pooled over 3 days — with an R² ≥ 0.30
quality gate; `gamma_med` is the daily median of the per-arc γ. There is no
`γ_rel` reference normalisation; instead γ goes through the per-(PRN, signal)
z-scoring described below.

**Why, and the effect.** gnssir-ice's architecture is per-arc and per-day, to
align with daily ground truth (GLERL, camera labels) — it does not pool
3-day windows. Per-arc γ is consequently noisier than Strandberg's pooled
estimate, which is why γ feeds a PCA-whitened **Mahalanobis** model (which
tolerates a noisy axis) rather than the hard `γ_rel` threshold of the paper.

### Song et al. (2022) — area factor → `af_med`

**What the paper did.** AF is the integral of the power curve from a continuous
wavelet transform (Morlet basis) of detrended SNR. Song removes the antenna
gain pattern by subtracting a per-satellite ice-free power-curve baseline
(Eq. 23), and reports the **daily AF as the mean across satellites**. The study
is Arctic **sea** ice (Tuktoyaktuk).

**What gnssir-ice does.** AF via a hand-rolled, scipy-faithful Morlet CWT
(`wavelet.py`, fixed centre frequency ω₀ = 5), integrating the power curve at
the dominant RH; the Eq. 23 per-(sat, freq) open-water baseline subtraction is
applied, pooled across `open_water_years` (see `docs/adr/0001`). The daily
`af_med` is the **median across arcs** (robust to a few bad arcs), computed
only for full arcs (Song's ≥ 80 % elevation-span requirement).

**Divergences.** (1) median-across-arcs vs Song's mean-across-satellites;
(2) **freshwater** Great Lakes ice vs Song's Arctic sea ice — roughly a 20×
dielectric difference, so AF baselines and even the *sign* of the ice response
differ (AF drops in Great Lakes ice, rises in Arctic sea ice — which is one
reason the polarity-agnostic Mahalanobis model is used); (3) gnssir-ice
integrates power *P* over elevation — whether this equals Song's stated
pseudo-energy `θ(t) = P(t)·t` is an **open fidelity question**, not yet
verified against the paper's reference implementation.

### Purnell et al. (2024) — `clr_med`, `pr_med`, `ms_mean`, `vs_mean`

**What the paper did.** Eight parameters from Lomb-Scargle harmonic estimation
on **2nd-order**-detrended SNR; `CLR = P1/mean(P₂…ₙ)`, `PR = P1/P2`. Purnell
applies a **4-hour moving average** (1-hour step; ≈ +5 % accuracy), normalises
each parameter **per satellite and per antenna**, and classifies with
**supervised** machine learning (neural net 93.7 %, random forest 92.2 %).

**What gnssir-ice does.** `CLR`, `PR`, `MS`, `VS` are computed per arc, then
aggregated to **one daily value** (no moving average); the detrend order is the
station's gnssrefl `polyV` (default **4th-order**, mirroring the `gnssir`
retrieval the SNR features sit on top of); each per-arc feature is
**per-(PRN, signal) z-score normalised** before aggregation; and the model is
**unsupervised** — StandardScaler → PCA-whitening → Mahalanobis distance from
an open-water baseline.

**Divergences and why.** Daily aggregation (not a 4-hour moving average)
because the ground truth and the operational unit are daily. 4th-order detrend
because gnssir-ice's per-arc detrend mirrors the gnssrefl station
configuration. An **unsupervised** model because gnssir-ice has no per-station
labelled training set — it cannot use Purnell's supervised NN/RF; the
Mahalanobis baseline needs only open-water reference days. The
per-(PRN, signal) z-scoring **is** Purnell's "normalise per satellite/antenna"
advice (also recommended by Strandberg) — the project originally omitted it and
adopting it measurably improved discrimination (next section).

## Did the adaptations help? — measured at ROSS and UMNQ

Per-arc features were labelled ice vs. open water (ROSS: GLERL ice
concentration; UMNQ: hand-labelled camera states) and the ice/water
separation measured as |Cohen's d|.

| Adaptation | Station | Feature | \|d\| before | \|d\| after |
|---|---|---|---|---|
| per-(PRN, signal) normalisation | ROSS | all 10 | — | within ±0.09 — **near-neutral** |
| per-(PRN, signal) normalisation | UMNQ | `amp_mean` | 0.66 | **0.89** |
| per-(PRN, signal) normalisation | UMNQ | `vs_mean` | 0.56 | **0.89** |
| per-(PRN, signal) normalisation | UMNQ | `af_med` | 0.78 | **0.92** |
| Song Eq. 23 AF antenna-gain correction | ROSS | `af_med` | 1.19 | **1.28** |

**Takeaway.** The per-(PRN, signal) normalisation is **station-dependent**:
near-neutral at ROSS, where the per-satellite channels are already well
behaved, but a clear gain at UMNQ (Greenland, 70°N, many constellations) — the
station whose channels carry the most per-satellite/antenna bias. Song's
Eq. 23 antenna-gain correction gives a smaller, consistent lift. This is why
both are *enabled by default but configurable* (`options.per_prn_normalization`,
`options.af_baseline`): they help where there is bias to remove and cost little
where there is not.

## One fixed 10-feature schema

The research code that preceded gnssir-ice selected features per station
(dropping `af_med` where the constellation under-supported the CWT, dropping
`rh_std_norm` where summer arcs were too few). gnssir-ice instead keeps a
**fixed 10-feature schema** and lets the model absorb the slack: PCA-whitening
collapses features that are redundant or degenerate at a given station (e.g.
`rh_std_norm` equals `rh_std_raw` exactly when per-PRN normalisation is off),
and a day missing a feature it cannot compute is dropped with per-feature
attribution. This trades the research code's per-station tuning for a simpler,
reproducible contract.

## References

- Strandberg, J., Hobiger, T., & Haas, R. (2017). Coastal sea ice detection
  using ground-based GNSS-R. *IEEE GRSL*, 14(9), 1552–1556.
  doi:10.1109/LGRS.2017.2722041
- Kim et al. (2021). *Remote Sensing*, 13(5), 976 — original clarity-ratio
  (CLR) formulation, per the project's design notes; adopted by Purnell 2024.
  (Full author list not yet confirmed against the paper.)
- Song, M., He, X., Jia, D., et al. (2022). Sea surface states detection in
  polar regions using ground-based GNSS interferometric reflectometry.
  *IEEE TGRS*, 60, 1–14. doi:10.1109/TGRS.2022.3155051
- Purnell, D., Dabboor, M., Matte, P., et al. (2024). Observations of river ice
  breakup using GNSS-IR, SAR, and machine learning. *IEEE TGRS*, 62, 5800613.
  doi:10.1109/TGRS.2024.3380854
- Larson, K. M., MacFerrin, M., & Nylen, T. (2020). Update on the GPS
  reflection technique for measuring snow accumulation in Greenland.
  *The Cryosphere*, 14, 1985–1988. doi:10.5194/tc-14-1985-2020 — representative
  GNSS-IR cryosphere reference for the underlying RH / amplitude observables.
