# gnssir-ice

Track lake/sea ice with GNSS interferometric reflectometry. `gnssir-ice` turns
gnssrefl output into **10 SNR-derived features** and a daily **Mahalanobis
distance** from an open-water baseline — the distance rises at freeze-up and
falls at break-up.

It is a thin, distributable pipeline meant to run *after* gnssrefl: point it at
one station's gnssrefl output with a single config file and get a daily
ice-tracking time series, per-azimuth-sector scores, and freeze-up / break-up
dates.

## What it computes

Ten daily features, aggregated from per-arc SNR observables:

| Feature | Meaning |
|---------|---------|
| `amp_mean`   | mean LSP amplitude |
| `rh_std_raw` / `rh_std_norm` | spread of reflector height (raw / per-PRN-normalized) |
| `p2n_mean`   | mean peak-to-noise ratio |
| `clr_med`    | clarity ratio — P1 / mean(other LSP peaks) (Kim 2021) |
| `pr_med`     | peak ratio — P1 / P2 (Purnell 2024) |
| `af_med`     | area factor — CWT power integral (Song 2022) |
| `gamma_med`  | SNR-envelope damping (Strandberg 2017) |
| `ms_mean`    | mean raw SNR (dB) |
| `vs_mean`    | variance of detrended SNR |

A StandardScaler → PCA-whitening → Mahalanobis model is fit on open-water days;
every day is then scored as a distance from that baseline — for the whole sky
and per azimuth sector. A final stage turns the distance series into discrete
ice seasons with freeze-up and break-up dates.

## Prerequisites

`gnssir-ice` consumes gnssrefl output — it does **not** download or process
RINEX. For each station-year, run gnssrefl first:

```bash
# 1. SNR files (snr66 format)
rinex2snr <station> <year> <doy> -snr 66 -orb gnss

# 2. per-arc reflector-height retrievals
#    (elevation / azimuth / frequency masks come from the gnssrefl station JSON)
gnssir <station> <year> <doy>

# 3. RHdot + inter-frequency reflector-height correction  (REQUIRED)
subdaily <station> <year> -knots 8
```

`gnssir-ice` resolves these under `$REFL_CODE`:

```
$REFL_CODE/{year}/snr/{station}/{station}{doy}0.{yy}.snr66
$REFL_CODE/{year}/results/{station}/{doy}.txt
$REFL_CODE/Files/{station}/{station}_{year}_subdaily_edit.txt.withrhdotIF
```

The `subdaily` step is **required** — `gnssir-ice` uses the RHdot/IF-corrected
reflector height, not the raw `gnssir` value. The gnssrefl output formats are a
hard contract: gnssir-ice is verified against **gnssrefl v3.19.3**, reads the
version from the subdaily header, and `build-arc-table` fails if too few arcs
receive the correction — a guard against a silent format mismatch.

Requirements: Python ≥ 3.9; numpy, pandas, scipy, scikit-learn, pyarrow, pyyaml
(installed automatically). gnssrefl itself is *not* a dependency of this
package — run it in whatever environment you already have.

## Install

`gnssir-ice` is distributed from GitHub (not on PyPI):

```bash
pip install "git+https://github.com/jjohnson2701/gnssir-ice.git"

# + matplotlib, for the `plot` command:
pip install "gnssir-ice[viz] @ git+https://github.com/jjohnson2701/gnssir-ice.git"

# or, from a checkout:
pip install -e .[dev]
```

Append `@<tag-or-commit>` to the URL to pin a version, e.g.
`...gnssir-ice.git@d8d2a24`.

## Quick start

```bash
# 1. scaffold a config and edit it
gnssir-ice init-config --station ROSS --out ross.yaml

# 2. run the full pipeline
gnssir-ice run --config ross.yaml

# 3. plot the result
gnssir-ice plot --config ross.yaml

# → results/ROSS/ROSS_daily_mahal_d.parquet  (+ .csv)
# → results/ROSS/ROSS_ice_seasons.csv
```

See [`examples/`](examples/) for a complete worked config and an explanation of
every output file.

## Pipeline

`run` walks 11 stages; each is also a standalone subcommand:

```
 1  consolidate-snr   daily snr66 files          → one SNR parquet
 2  build-arc-table   gnssir + subdaily output   → arc_table (RHdot/IF-corrected RH)
 3  extract           SNR arcs                   → arc_features + power curves
 4  af-baseline       open-water power curves    → pooled antenna-gain AF baseline
 5  af-correct        power curves + baseline    → arc_af (corrected area factor)
 6  prn-baseline      open-water arcs            → per-(PRN, signal) z-score baseline
 7  normalize         arc_af + prn-baseline      → arc_norm (*_norm columns)
 8  aggregate         arc_norm                   → 10 daily features
 9  fit-baseline      open-water daily features  → Mahalanobis model
10  score             daily features + model     → daily_mahal_d + sector_mahal_d
11  ice-seasons       scored series              → freeze-up / break-up dates
```

Run a single stage with, e.g., `gnssir-ice extract --config ross.yaml --year 2024`.

The per-arc table is an **immutable chain** — `arc_table` → `arc_features` →
`arc_af` → `arc_norm` — each stage reads the previous file and writes its own,
so a file's contents always identify the stage that produced it
([ADR-0001](docs/adr/0001-immutable-per-stage-artifacts.md)).

Two stages are optional (both default-on, toggle in the config `options`):
`per_prn_normalization` and `af_baseline`. With per-PRN normalization off,
`rh_std_norm` equals `rh_std_raw` and PCA collapses the redundant axis.

## Per-sector scoring

Alongside the whole-sky result, `score` fits a per-azimuth-sector baseline: the
scaler / PCA / threshold are shared, but each sector is recentred on its own
open-water mean — so a sector that simply faces a different surface is not
flagged as anomalous. Sector width is `baseline.sector_width_deg` (default
30°); a sector with fewer than `min_sector_baseline_days` open-water days is
skipped. Per-sector scores land in `{station}_sector_mahal_d.parquet`, and
Stage 11 derives a freeze-up / break-up date for each sector as well.

## Plotting & validation

`gnssir-ice plot` renders the scored series as one panel per ice year
(Jul–Jun) — `mahal_d` on a log axis, with vertical stripes over the open-water
months sampled for the baseline, the alarm threshold, and above-threshold days
marked (needs the `[viz]` extra). Pass `--validation` to overlay an independent
ground-truth ice series on a second axis:

```bash
gnssir-ice plot --config ross.yaml --validation ross_glerl.csv
```

The validation file is a generic two-column CSV — `date` (YYYY-MM-DD) and
`ice_fraction` (0 = open water, 1 = full ice). Hand-labelled camera states
convert to it with
[`examples/camera_labels_to_validation.py`](examples/camera_labels_to_validation.py);
GLERL or other ice-concentration products convert the same way
(`ice_fraction = ice% / 100`).

Add `--features` to also write a companion `*_features.png` — a per-feature
z-score heatmap showing which of the 10 daily features drove a high `mahal_d`.

## Configuration

One YAML (or JSON) file per station defines coordinates, the gnssrefl
processing parameters, the open-water baseline windows, path templates, and the
optional-stage toggles. `coordinates` are **optional** provenance metadata —
the model does not use them.

The baseline has **two month windows but one shared year list**:
`open_water_months` is the narrow, reliably ice-free window the Mahalanobis
baseline is fit on; `normalization_months` is a wider window feeding the
per-PRN and AF baselines (more arcs → more stable per-channel statistics).
Months have an ice-free-purity gradient, so they get two windows; a year is
simply a good open-water year or not, so `open_water_years` is shared by both.

Path templates expand `{station}`, `{station_lower}`, `{year}`, `{yy}`,
`{doy}` and `${ENV_VAR}`. See
[`examples/ross.example.yaml`](examples/ross.example.yaml).

## References

The 10 features **adapt** observables from the GNSS-IR ice-detection
literature. [`docs/features.md`](docs/features.md) gives per-feature provenance
and a precise account of where gnssir-ice follows the cited methods and where
it diverges (per-arc rather than multi-day damping, daily aggregation, an
unsupervised baseline model), with the measured effect of each adaptation.

- Strandberg et al. (2017) — SNR-envelope damping γ.
- Kim et al. (2021) — the clarity ratio (CLR).
- Song et al. (2022) — the CWT area factor and its antenna-gain correction.
- Purnell et al. (2024) — the peak-ratio / mean-SNR / SNR-variance indicators
  and per-satellite feature normalization.

## License

MIT — see [LICENSE](LICENSE).
