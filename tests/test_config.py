# ABOUTME: Tests for StationConfig loading/validation and PathResolver templating.

import json

import pytest

from gnssir_ice.config import ConfigError, PathResolver, StationConfig
from tests.conftest import config_dict


def test_load_yaml(tmp_path):
    cfg_path = tmp_path / "s.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(config_dict(tmp_path)))
    cfg = StationConfig.load(cfg_path)
    assert cfg.station == "TEST"
    assert cfg.gnssir.e1 == 5.0
    assert cfg.baseline.open_water_months == [7, 8]


def test_load_json(tmp_path):
    cfg_path = tmp_path / "s.json"
    cfg_path.write_text(json.dumps(config_dict(tmp_path)))
    cfg = StationConfig.load(cfg_path)
    assert cfg.station == "TEST"


def test_missing_required_key_raises(tmp_path):
    data = config_dict(tmp_path)
    del data["gnssir"]
    with pytest.raises(ConfigError, match="gnssir"):
        StationConfig.from_dict(data)


def test_coordinates_optional(tmp_path):
    data = config_dict(tmp_path)
    del data["coordinates"]                       # provenance only — optional
    cfg = StationConfig.from_dict(data)
    assert cfg.coordinates.latitude_deg is None
    assert cfg.coordinates.ellipsoidal_height_m is None


def test_rejects_e1_ge_e2(tmp_path):
    data = config_dict(tmp_path, gnssir={"e1": 30.0, "e2": 25.0})
    with pytest.raises(ConfigError, match="e1"):
        StationConfig.from_dict(data)


def test_rejects_minh_ge_maxh(tmp_path):
    data = config_dict(tmp_path, gnssir={"minH": 8.0, "maxH": 2.0})
    with pytest.raises(ConfigError, match="minH"):
        StationConfig.from_dict(data)


def test_rejects_empty_open_water_years(tmp_path):
    data = config_dict(tmp_path, baseline={"open_water_years": []})
    with pytest.raises(ConfigError, match="open_water_years"):
        StationConfig.from_dict(data)


def test_pathresolver_templating(tmp_path):
    cfg = StationConfig.from_dict(config_dict(tmp_path))
    r = PathResolver(cfg)
    snr = r.snr_file(2024, 7)
    assert snr.name == "test0070.24.snr66"          # {doy} zero-padded, {yy}
    assert r.arc_table(2024).name == "TEST_2024_arc_table.parquet"
    assert r.daily_mahal_d().name == "TEST_daily_mahal_d.parquet"


def test_pathresolver_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("GNSSIR_TEST_ROOT", str(tmp_path / "rc"))
    data = config_dict(tmp_path, paths={"refl_code": "${GNSSIR_TEST_ROOT}"})
    cfg = StationConfig.from_dict(data)
    r = PathResolver(cfg)
    assert str(tmp_path / "rc") in str(r.snr_file(2024, 7))


def test_pathresolver_missing_env_raises(tmp_path):
    data = config_dict(tmp_path, paths={"refl_code": "${DEFINITELY_NOT_SET_XYZ}"})
    cfg = StationConfig.from_dict(data)
    with pytest.raises(ConfigError, match="environment variable"):
        PathResolver(cfg)
