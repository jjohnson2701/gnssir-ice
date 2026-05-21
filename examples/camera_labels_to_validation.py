#!/usr/bin/env python3
# ABOUTME: Convert hand-labelled camera ice states to a gnssir-ice validation CSV.

"""Convert a camera ice-state label CSV to a gnssir-ice validation CSV.

The input is a hand-labelling schema with at least ``station``, ``date`` and
``state`` columns (state in open_water / partial_ice / full_ice / …). The
output is the two-column ``date, ice_fraction`` CSV that
``gnssir-ice plot --validation`` overlays.

    python examples/camera_labels_to_validation.py \\
        --labels labels.csv --station UMNQ --out umnq_validation.csv

GLERL ice concentration is converted the same way — emit a ``date,
ice_fraction`` CSV (fraction = ice % / 100) from your own GLERL tooling.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

# Map each labelled ice state to an ice fraction in [0, 1].
STATE_TO_FRACTION = {
    "open_water": 0.0,
    "freeze_onset": 0.25,
    "break_up": 0.5,
    "partial_ice": 0.5,
    "full_ice": 1.0,
    "snow_covered_ice": 1.0,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="camera label CSV")
    ap.add_argument("--station", required=True, help="station to extract")
    ap.add_argument("--out", required=True, help="output validation CSV")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.labels)
    df = df[df["station"].astype(str).str.upper() == args.station.upper()]
    df = df[df["state"].isin(STATE_TO_FRACTION)]
    if df.empty:
        print(f"no usable labels for station {args.station}", file=sys.stderr)
        return 1

    df = df.assign(ice_fraction=df["state"].map(STATE_TO_FRACTION))
    # one row per date — mean fraction when several cameras/clips share a day.
    out = (df.groupby("date")["ice_fraction"].mean()
             .reset_index().sort_values("date"))
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} dates, "
          f"{out['date'].min()} .. {out['date'].max()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
