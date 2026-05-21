# Changelog

All notable changes to gnssir-ice are documented here.

## [0.1.0] — unreleased

Initial release. Extracted and refactored from the GNSSIRWorkflow research repo
into a standalone, distributable package.

### Pipeline
- 11-stage pipeline from gnssrefl output to a daily Mahalanobis distance:
  1 consolidate-snr → 2 build-arc-table → 3 extract → 4 af-baseline →
  5 af-correct → 6 prn-baseline → 7 normalize → 8 aggregate → 9 fit-baseline →
  10 score → 11 ice-seasons. `run` walks them all; each is also a subcommand.
- Per-arc data flows through an **immutable artifact chain** — `arc_table` →
  `arc_features` → `arc_af` → `arc_norm` — one file per stage, no in-place
  mutation (see `docs/adr/0001-immutable-per-stage-artifacts.md`).
- The 10-feature daily model schema (`amp_mean`, `rh_std_raw`, `rh_std_norm`,
  `p2n_mean`, `clr_med`, `pr_med`, `af_med`, `gamma_med`, `ms_mean`, `vs_mean`).
- StandardScaler → PCA-whitening → Mahalanobis scoring against an open-water
  baseline, pooled and **per azimuth sector** (each sector recentred on its own
  open-water mean).
- Stage 11 extracts **freeze-up / break-up dates** per ice season from the
  scored series — for the pooled result and per sector.
- Hand-rolled, scipy-faithful Morlet CWT (`wavelet.py`) for the area factor.
- Antenna-gain AF correction (Song 2022 Eq. 23), pooled across the open-water
  years, applied from persisted power curves — no second SNR read or CWT pass.
- Required RHdot + inter-frequency reflector-height correction, merged from
  gnssrefl `subdaily` output by the arc-table builder.

### Configuration & CLI
- Single per-station YAML/JSON config (`StationConfig`) with config-driven path
  templates (`PathResolver`); `coordinates` are optional provenance metadata.
- `gnssir-ice` CLI: `run` plus one subcommand per stage, `init-config` to
  scaffold a config, and `plot` to render the result (optional `[viz]` extra).

### Robustness
- Verified against gnssrefl **v3.19.3** output formats; `build-arc-table`
  parses the gnssrefl version from the subdaily header and **fails** when the
  RHdot/IF match rate falls below `options.min_subdaily_match_rate` — a guard
  against a silent format mismatch.
- Short SNR files (e.g. `snr66` missing the trailing Galileo S7/S8 columns)
  are zero-padded and used, not skipped.
- Days excluded from scoring for NaN features are attributed to the specific
  feature(s) in the run log.
- Run provenance — station, coordinates, gnssrefl version, baseline windows —
  is embedded in the daily-output parquet metadata and a
  `{station}_run_manifest.json` sidecar.
- Per-file warning spam is aggregated into one-line summaries; `run` prints an
  end-of-run summary (days scored vs calendar days, % above threshold).
