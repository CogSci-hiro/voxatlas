import pytest

from voxatlas.config.exceptions import ConfigValidationError
from voxatlas.config.schema import validate_config


def test_validate_config_success():
    cfg = {
        "features": [
            "acoustic.pitch.f0",
            "phonology.duration.phoneme",
        ],
        "feature_config": {
            "acoustic.pitch.f0": {
                "frame_step": 0.01,
            },
        },
    }

    validate_config(cfg)


def test_validate_config_raises_when_features_missing():
    cfg = {
        "pipeline": {
            "cache": True,
        }
    }

    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_validate_config_raises_when_features_is_not_list():
    cfg = {
        "features": "acoustic.pitch.f0"
    }

    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_validate_config_raises_when_feature_config_has_unknown_feature():
    cfg = {
        "features": ["acoustic.pitch.f0"],
        "feature_config": {
            "phonology.duration.phoneme": {
                "window": 5,
            },
        },
    }

    with pytest.raises(ConfigValidationError):
        validate_config(cfg)
