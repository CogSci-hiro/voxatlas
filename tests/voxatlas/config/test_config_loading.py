import yaml
from pathlib import Path

from voxatlas.config.config import load_and_prepare_config, load_config


def test_load_config_from_yaml(tmp_path: Path):
    cfg = {
        "features": ["acoustic.pitch.f0"]
    }

    config_file = tmp_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.safe_dump(cfg, f)

    loaded = load_config(config_file)

    assert loaded["features"] == ["acoustic.pitch.f0"]


def test_load_and_prepare_config_returns_merged_config(tmp_path: Path):
    cfg = {
        "features": ["acoustic.pitch.f0"]
    }

    config_file = tmp_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.safe_dump(cfg, f)

    loaded = load_and_prepare_config(config_file)

    assert loaded == {
        "features": ["acoustic.pitch.f0"],
        "pipeline": {
            "cache": True,
        },
    }
