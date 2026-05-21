# ABOUTME: StationConfig dataclass + YAML/JSON loader + PathResolver.
# ABOUTME: One per-station config file drives every pipeline stage.

"""Configuration for one GNSS-IR station.

A single per-station file (YAML or JSON) supplies coordinates, gnssrefl
processing parameters, the open-water baseline definition, and path templates.
:class:`StationConfig` loads and validates it; :class:`PathResolver` turns the
path templates into concrete file paths for each pipeline stage.

Path templates support ``{station}``, ``{station_lower}``, ``{year}``,
``{yy}``, ``{doy}`` placeholders and ``${ENV_VAR}`` environment expansion.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a station config is missing required keys or is invalid."""


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------
@dataclass
class Coordinates:
    """Optional station provenance. ``latitude_deg`` / ``longitude_deg`` are
    recorded for traceability but unused by the model; ``ellipsoidal_height_m``
    only adds the diagnostic ``wse`` column when supplied."""
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    ellipsoidal_height_m: float | None = None


@dataclass
class GnssirParams:
    """gnssrefl ``gnssir`` processing parameters (mirror the station JSON)."""
    e1: float                       # min elevation (deg)
    e2: float                       # max elevation (deg)
    minH: float                     # reflector-height search min (m)
    maxH: float                     # reflector-height search max (m)
    polyV: int = 4                  # detrend polynomial order
    pele: tuple[float, float] = (5.0, 30.0)   # poly-fit elevation window (deg)
    desiredP: float = 0.005         # LSP RH grid step (m)


@dataclass
class BaselineParams:
    open_water_months: list[int]    # months pooled for the Mahalanobis fit
    open_water_years: list[int]     # years pooled for the Mahalanobis fit
    normalization_months: list[int] # per-PRN / AF-baseline reference window
    pca_variance: float = 0.95      # PCA variance retained
    threshold_percentile: float = 99.0   # mahal_d alarm percentile
    af_baseline_max_arcs: int | None = None  # cap on arcs per AF baseline (None = all)
    sector_width_deg: int = 30      # azimuth-sector width for per-sector scoring
    min_sector_baseline_days: int = 20   # min open-water days for a sector baseline
    ice_season_min_run_days: int = 5  # min consecutive above-threshold days/season


@dataclass
class PathTemplates:
    refl_code: str = "${REFL_CODE}"
    snr_filename: str = "{station_lower}{doy}0.{yy}.snr66"
    snr_dir: str = "{refl_code}/{year}/snr/{station_lower}"
    gnssir_output: str = "{refl_code}/{year}/results/{station_lower}"
    subdaily_file: str = ("{refl_code}/Files/{station_lower}/"
                          "{station_lower}_{year}_subdaily_edit.txt.withrhdotIF")
    output_root: str = "./results/{station}"


@dataclass
class Options:
    per_prn_normalization: bool = True
    af_baseline: bool = True
    min_subdaily_match_rate: float = 0.5  # build-arc-table raises below this


# ---------------------------------------------------------------------------
# StationConfig
# ---------------------------------------------------------------------------
@dataclass
class StationConfig:
    station: str
    coordinates: Coordinates
    gnssir: GnssirParams
    baseline: BaselineParams
    paths: PathTemplates
    options: Options
    years: list[int] = field(default_factory=list)

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "StationConfig":
        """Load a station config from a YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(f"unknown config extension {path.suffix!r} "
                              "(expected .yaml/.yml/.json)")
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top-level config must be a mapping")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StationConfig":
        def _require(d: dict, key: str, ctx: str) -> Any:
            if key not in d:
                raise ConfigError(f"missing required config key: {ctx}{key}")
            return d[key]

        station = _require(data, "station", "")
        gn = _require(data, "gnssir", "")
        bl = _require(data, "baseline", "")

        def _opt_float(d: dict, key: str) -> float | None:
            v = d.get(key)
            return float(v) if v is not None else None

        # coordinates are optional provenance metadata (see Coordinates).
        coord = data.get("coordinates", {}) or {}
        coordinates = Coordinates(
            latitude_deg=_opt_float(coord, "latitude_deg"),
            longitude_deg=_opt_float(coord, "longitude_deg"),
            ellipsoidal_height_m=_opt_float(coord, "ellipsoidal_height_m"),
        )
        gnssir = GnssirParams(
            e1=float(_require(gn, "e1", "gnssir.")),
            e2=float(_require(gn, "e2", "gnssir.")),
            minH=float(_require(gn, "minH", "gnssir.")),
            maxH=float(_require(gn, "maxH", "gnssir.")),
            polyV=int(gn.get("polyV", 4)),
            pele=tuple(float(x) for x in gn.get("pele", [5.0, 30.0])),
            desiredP=float(gn.get("desiredP", 0.005)),
        )
        norm_months = bl.get("normalization_months",
                             bl.get("open_water_months"))
        baseline = BaselineParams(
            open_water_months=list(_require(bl, "open_water_months", "baseline.")),
            open_water_years=list(_require(bl, "open_water_years", "baseline.")),
            normalization_months=list(norm_months) if norm_months else [],
            pca_variance=float(bl.get("pca_variance", 0.95)),
            threshold_percentile=float(bl.get("threshold_percentile", 99.0)),
            af_baseline_max_arcs=(int(bl["af_baseline_max_arcs"])
                                  if bl.get("af_baseline_max_arcs") else None),
            sector_width_deg=int(bl.get("sector_width_deg", 30)),
            min_sector_baseline_days=int(bl.get("min_sector_baseline_days", 20)),
            ice_season_min_run_days=int(bl.get("ice_season_min_run_days", 5)),
        )
        pt = data.get("paths", {})
        defaults = PathTemplates()
        paths = PathTemplates(
            refl_code=pt.get("refl_code", defaults.refl_code),
            snr_filename=pt.get("snr_filename", defaults.snr_filename),
            snr_dir=pt.get("snr_dir", defaults.snr_dir),
            gnssir_output=pt.get("gnssir_output", defaults.gnssir_output),
            subdaily_file=pt.get("subdaily_file", defaults.subdaily_file),
            output_root=pt.get("output_root", defaults.output_root),
        )
        op = data.get("options", {})
        options = Options(
            per_prn_normalization=bool(op.get("per_prn_normalization", True)),
            af_baseline=bool(op.get("af_baseline", True)),
            min_subdaily_match_rate=float(op.get("min_subdaily_match_rate", 0.5)),
        )
        years = list(data.get("processing", {}).get("years", []))

        cfg = cls(station=str(station), coordinates=coordinates, gnssir=gnssir,
                  baseline=baseline, paths=paths, options=options, years=years)
        cfg.validate()
        return cfg

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        """Fail fast on an unusable config."""
        if not self.station:
            raise ConfigError("station id must be non-empty")
        g = self.gnssir
        if g.e1 >= g.e2:
            raise ConfigError(f"gnssir.e1 ({g.e1}) must be < e2 ({g.e2})")
        if g.minH >= g.maxH:
            raise ConfigError(f"gnssir.minH ({g.minH}) must be < maxH ({g.maxH})")
        if g.polyV < 1:
            raise ConfigError(f"gnssir.polyV must be >= 1 (got {g.polyV})")
        if g.desiredP <= 0:
            raise ConfigError("gnssir.desiredP must be > 0")
        b = self.baseline
        if not b.open_water_months:
            raise ConfigError("baseline.open_water_months must be non-empty")
        if not b.open_water_years:
            raise ConfigError("baseline.open_water_years must be non-empty")
        if not b.normalization_months:
            raise ConfigError("baseline.normalization_months must be non-empty")
        if any(not 1 <= m <= 12 for m in b.open_water_months + b.normalization_months):
            raise ConfigError("baseline months must be in 1..12")
        if not 0 < b.pca_variance <= 1.0:
            raise ConfigError("baseline.pca_variance must be in (0, 1]")
        if not 0 < b.threshold_percentile < 100:
            raise ConfigError("baseline.threshold_percentile must be in (0, 100)")
        if not 0 < b.sector_width_deg <= 360:
            raise ConfigError("baseline.sector_width_deg must be in (0, 360]")
        if b.min_sector_baseline_days < 1:
            raise ConfigError("baseline.min_sector_baseline_days must be >= 1")
        if b.ice_season_min_run_days < 1:
            raise ConfigError("baseline.ice_season_min_run_days must be >= 1")
        if not 0 < self.options.min_subdaily_match_rate <= 1:
            raise ConfigError("options.min_subdaily_match_rate must be in (0, 1]")


# ---------------------------------------------------------------------------
# PathResolver
# ---------------------------------------------------------------------------
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PathResolver:
    """Turn a :class:`StationConfig`'s path templates into concrete paths.

    All package outputs land under ``paths.output_root``; gnssrefl inputs are
    resolved under ``paths.refl_code``.
    """

    def __init__(self, config: StationConfig):
        self.config = config
        self.station = config.station
        self.station_lower = config.station.lower()
        self.refl_code = self._expand(config.paths.refl_code, {})
        self.output_root = Path(self._expand(
            config.paths.output_root,
            {"station": self.station, "station_lower": self.station_lower},
        ))

    # -- template expansion -------------------------------------------------
    @staticmethod
    def _expand_env(template: str) -> str:
        def repl(m: re.Match) -> str:
            name = m.group(1)
            val = os.environ.get(name)
            if val is None:
                raise ConfigError(
                    f"environment variable ${{{name}}} referenced in config "
                    "is not set")
            return val
        return _ENV_RE.sub(repl, template)

    def _expand(self, template: str, ctx: dict[str, Any]) -> str:
        full = {"station": self.station, "station_lower": self.station_lower}
        full.update(ctx)
        return self._expand_env(template).format(**full)

    def _ctx(self, year: int | None = None, doy: int | None = None) -> dict:
        ctx: dict[str, Any] = {"refl_code": self.refl_code}
        if year is not None:
            ctx["year"] = year
            ctx["yy"] = f"{year % 100:02d}"
        if doy is not None:
            ctx["doy"] = f"{doy:03d}"
        return ctx

    # -- gnssrefl inputs ----------------------------------------------------
    def snr_file(self, year: int, doy: int) -> Path:
        """Path to one daily gnssrefl ``snr66`` file (the ``.gz`` form is also
        accepted by :func:`gnssir_ice.snr.read_snr_file`)."""
        ctx = self._ctx(year, doy)
        d = self._expand(self.config.paths.snr_dir, ctx)
        f = self._expand(self.config.paths.snr_filename, ctx)
        return Path(d) / f

    def gnssir_results_dir(self, year: int) -> Path:
        """Directory holding gnssrefl ``gnssir`` per-day ``.txt`` output."""
        return Path(self._expand(self.config.paths.gnssir_output,
                                 self._ctx(year)))

    def subdaily_file(self, year: int) -> Path:
        """Path to the gnssrefl ``subdaily`` ``.withrhdotIF`` output file."""
        return Path(self._expand(self.config.paths.subdaily_file,
                                 self._ctx(year)))

    # -- package outputs ----------------------------------------------------
    # The per-arc table is an immutable chain: each stage reads the previous
    # artifact and writes its own, so a file's contents identify the stage
    # that produced it (docs/adr/0001-immutable-per-stage-artifacts.md):
    #     arc_table -> arc_features -> arc_af -> arc_norm
    def _out(self, name: str) -> Path:
        return self.output_root / name

    def snr_consolidated(self, year: int) -> Path:
        return self._out(f"{self.station}_{year}_snr.parquet")

    def arc_table(self, year: int) -> Path:
        """Stage 2 output — raw gnssir + subdaily-corrected RH."""
        return self._out(f"{self.station}_{year}_arc_table.parquet")

    def arc_features(self, year: int) -> Path:
        """Stage 3 output — arc_table plus the per-arc SNR features."""
        return self._out(f"{self.station}_{year}_arc_features.parquet")

    def arc_features_csv(self, year: int) -> Path:
        """Human-readable CSV companion to the Stage 3 arc_features table."""
        return self._out(f"{self.station}_{year}_arc_features.csv")

    def arc_af(self, year: int) -> Path:
        """Stage 5 output — arc_features with the antenna-gain-corrected AF."""
        return self._out(f"{self.station}_{year}_arc_af.parquet")

    def arc_norm(self, year: int) -> Path:
        """Stage 7 output — arc_af plus the per-(PRN, signal) *_norm columns."""
        return self._out(f"{self.station}_{year}_arc_norm.parquet")

    def arc_pre_norm(self, year: int) -> Path:
        """The arc artifact feeding prn-baseline / normalize — arc_af when the
        AF correction is enabled, else arc_features."""
        return (self.arc_af(year) if self.config.options.af_baseline
                else self.arc_features(year))

    def arc_head(self, year: int) -> Path:
        """The final arc artifact for the year given the enabled options —
        the table the daily aggregation consumes."""
        if self.config.options.per_prn_normalization:
            return self.arc_norm(year)
        return self.arc_pre_norm(year)

    def power_curves(self, year: int) -> Path:
        return self._out(f"{self.station}_{year}_power_curves.npz")

    def af_baseline(self) -> Path:
        """Pooled antenna-gain AF baseline — one per station (ADR/D9)."""
        return self._out(f"{self.station}_af_baseline.npz")

    def daily_features(self, year: int) -> Path:
        return self._out(f"{self.station}_{year}_daily_features.parquet")

    def prn_baseline(self) -> Path:
        return self._out(f"{self.station}_prn_baseline.json")

    def baseline_pkl(self) -> Path:
        return self._out(f"{self.station}_baseline.pkl")

    def daily_mahal_d(self) -> Path:
        return self._out(f"{self.station}_daily_mahal_d.parquet")

    def daily_mahal_d_csv(self) -> Path:
        """CSV companion to the pooled daily Mahalanobis-distance result."""
        return self._out(f"{self.station}_daily_mahal_d.csv")

    def sector_mahal_d(self) -> Path:
        """Per-azimuth-sector daily Mahalanobis-distance result."""
        return self._out(f"{self.station}_sector_mahal_d.parquet")

    def ice_seasons(self) -> Path:
        """Freeze-up / break-up dates extracted per ice season."""
        return self._out(f"{self.station}_ice_seasons.csv")

    def run_manifest(self) -> Path:
        """Run provenance — station, coordinates, gnssrefl version, windows."""
        return self._out(f"{self.station}_run_manifest.json")

    def ensure_output_dir(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
