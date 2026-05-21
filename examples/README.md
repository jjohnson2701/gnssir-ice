# Worked example — ROSS

`ross.example.yaml` is a complete config for the ROSS station (Rossport, Lake
Superior). It assumes you have already run gnssrefl for the years in
`processing.years` (see the [prerequisites](../README.md#prerequisites) in the
main README).

## Run it

```bash
cp examples/ross.example.yaml ross.yaml
# edit paths / coordinates for your machine, then:
gnssir-ice run --config ross.yaml
```

This walks all 11 stages and writes, under `results/ROSS/`:

| File | What it is |
|------|-----------|
| `ROSS_{year}_snr.parquet`               | consolidated SNR — Stage 1 |
| `ROSS_{year}_arc_table.parquet`         | gnssir + subdaily-corrected RH — Stage 2 |
| `ROSS_{year}_arc_features.parquet`      | arc_table + per-arc SNR features — Stage 3 |
| `ROSS_{year}_arc_features.csv`          | human-readable copy of the per-arc features |
| `ROSS_{year}_power_curves.npz`          | persisted CWT power curves |
| `ROSS_af_baseline.npz`                  | pooled antenna-gain AF baseline — Stage 4 |
| `ROSS_{year}_arc_af.parquet`            | arc_features with the corrected AF — Stage 5 |
| `ROSS_prn_baseline.json`                | per-(PRN, signal) z-score baseline — Stage 6 |
| `ROSS_{year}_arc_norm.parquet`          | arc_af + per-PRN `*_norm` columns — Stage 7 |
| `ROSS_{year}_daily_features.parquet`    | the 10 daily features — Stage 8 |
| `ROSS_baseline.pkl`                     | fitted open-water Mahalanobis model — Stage 9 |
| `ROSS_daily_mahal_d.parquet` (+ `.csv`) | **the result** — daily distance from baseline — Stage 10 |
| `ROSS_sector_mahal_d.parquet`           | per-azimuth-sector daily distance — Stage 10 |
| `ROSS_ice_seasons.csv`                  | freeze-up / break-up dates per season — Stage 11 |
| `ROSS_run_manifest.json`                | run provenance — station, coordinates, gnssrefl version, windows |

The four `arc_*` tables form an **immutable chain**: each stage reads the
previous file and writes its own, so a file's contents always identify the
stage that produced it.

## Interpreting `ROSS_daily_mahal_d.parquet`

Each row is one day:

- `mahal_d` — Mahalanobis distance of that day's 10-feature vector from the
  open-water baseline. Open-water summer days sit near the baseline median;
  ice-season days rise well above it.
- `above_threshold` — True when `mahal_d` exceeds the configured percentile
  (`baseline.threshold_percentile`) of the baseline distribution.
- `{feature}_z` — the standardized value of each of the 10 features that day —
  useful for seeing *which* features drove a high `mahal_d`.

Render it with `gnssir-ice plot --config ross.yaml`: the curve steps up at
freeze-up and back down at break-up.

## `ROSS_ice_seasons.csv`

One row per detected ice season — `freeze_up`, `break_up`, `duration_days`, and
`peak_mahal_d`. `sector = -1` is the pooled all-azimuth series; `sector >= 0`
rows are the per-azimuth-sector seasons derived from `ROSS_sector_mahal_d.parquet`.

## Validation overlay

`camera_labels_to_validation.py` converts a hand-labelled camera ice-state CSV
into the generic `date, ice_fraction` validation CSV that `gnssir-ice plot
--validation` overlays:

```bash
python examples/camera_labels_to_validation.py \
    --labels labels.csv --station UMNQ --out umnq_validation.csv
gnssir-ice plot --config umnq.yaml --validation umnq_validation.csv
```

GLERL ice concentration (or any other ice product) converts the same way —
emit a two-column `date, ice_fraction` CSV with `ice_fraction = ice% / 100`.
