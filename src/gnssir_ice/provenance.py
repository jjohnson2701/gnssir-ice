# ABOUTME: Run provenance — embeds station/config/version metadata into outputs.
# ABOUTME: Writes a {station}_run_manifest.json and stamps the daily parquets.

"""Run provenance.

Daily output files are otherwise not self-describing: detached from their
``{station}_``-named path they carry no station, coordinates, gnssrefl version,
or baseline window. This module records that provenance two ways — embedded in
the parquet schema metadata, and as a sidecar ``{station}_run_manifest.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gnssir_ice import __version__
from gnssir_ice.config import StationConfig
from gnssir_ice.constants import TESTED_GNSSREFL_VERSION

_META_KEY = b"gnssir_ice_provenance"


def build_provenance(config: StationConfig, **extra: Any) -> dict:
    """Assemble the run-provenance record for a station configuration."""
    c = config.coordinates
    prov = {
        "package": "gnssir-ice",
        "package_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "station": config.station,
        "coordinates": {
            "latitude_deg": c.latitude_deg,
            "longitude_deg": c.longitude_deg,
            "ellipsoidal_height_m": c.ellipsoidal_height_m,
        },
        "gnssrefl_tested_version": TESTED_GNSSREFL_VERSION,
        "baseline": {
            "open_water_months": list(config.baseline.open_water_months),
            "open_water_years": list(config.baseline.open_water_years),
            "normalization_months": list(config.baseline.normalization_months),
            "pca_variance": config.baseline.pca_variance,
            "threshold_percentile": config.baseline.threshold_percentile,
            "sector_width_deg": config.baseline.sector_width_deg,
        },
    }
    prov.update(extra)
    return prov


def write_parquet(df: pd.DataFrame, path: Path, config: StationConfig,
                  **extra: Any) -> None:
    """Write ``df`` to ``path`` as parquet, with provenance in schema metadata."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    prov = json.dumps(build_provenance(config, **extra)).encode()
    table = table.replace_schema_metadata({**existing, _META_KEY: prov})
    pq.write_table(table, path)


def read_provenance(path: Path) -> dict | None:
    """Read back the provenance record embedded in a parquet file, if any."""
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(_META_KEY)
    return json.loads(raw) if raw else None


def write_run_manifest(config: StationConfig, path: Path, **extra: Any) -> Path:
    """Write the ``{station}_run_manifest.json`` provenance sidecar."""
    path.write_text(json.dumps(build_provenance(config, **extra), indent=2))
    return path
