# gnssir-ice

Domain glossary for gnssir-ice — a pipeline that turns gnssrefl GNSS
interferometric-reflectometry output into daily SNR features and a Mahalanobis
distance from an open-water baseline, used to detect lake and sea ice.

This file is a glossary only. It defines what terms *are*; it carries no
implementation detail and no design rationale (rationale lives in `docs/adr/`).

## Language

**Reflector height (RH)**:
The vertical distance from the GNSS antenna down to the reflecting surface
(water or ice), in metres. The primary geometric observable — every later
quantity derives from RH or from the SNR arc that produced it.
_Avoid_: antenna height (that is the fixed hardware height above a datum),
surface height.

**Arc**:
One satellite's continuous rise or set pass across the sky within a day's SNR
record. The same physical arc observed on several carrier frequencies yields
one arc-table row per (arc, frequency).
_Avoid_: track, pass, segment.

**Water surface elevation (wse)**:
The elevation of the reflecting surface, equal to the antenna's ellipsoidal
height minus the reflector height. A diagnostic quantity only — it is not one
of the model features and does not affect the ice result.
_Avoid_: water level, surface height.

**Stage**:
One of the ten ordered steps of the pipeline (1 consolidate-snr … 10 score),
each also runnable on its own as a CLI subcommand.
_Avoid_: step, phase, layer.

**Open water**:
The ice-free reference period a station's baseline is built from. The exact
month and year windows are configurable (see `baseline` in the station config).
_Avoid_: summer, ice-free season, baseline period.

**Azimuth sector**:
A contiguous wedge of satellite azimuth (width `sector_width_deg`, default 30°)
that arcs are binned into. Each sufficiently sampled sector is scored against
its own recentred open-water baseline, alongside the pooled all-azimuth result.
_Avoid_: bin, quadrant, direction.

**Ice season**:
A sustained run of consecutive above-threshold scored days. Its **freeze-up**
date is the run's first day and its **break-up** date the run's last.
_Avoid_: freeze period, ice-on / ice-off.

## Flagged ambiguities

- **"Feature" is overloaded.** A *per-arc feature* (e.g. CLR, AF, gamma on a
  single arc) is not the same as one of the *ten daily model features*
  (`amp_mean`, `clr_med`, …) that the Mahalanobis baseline consumes. Say
  "per-arc feature" or "daily feature" — never a bare "feature".
