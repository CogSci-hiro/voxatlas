from copy import deepcopy

import yaml

from .defaults import DEFAULT_CONFIG
from .schema import validate_config


def load_config(path: str) -> dict:
    """
    Load a VoxAtlas YAML configuration file.

    Parameters
    ----------
    path : str
        Filesystem path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    OSError
        Raised when the file cannot be opened.
    yaml.YAMLError
        Raised when the YAML document is invalid.

    Notes
    -----
    This function parses YAML only. It does not apply defaults or schema
    validation.

    Examples
    --------
    Usage example::

        cfg = load_config("config.yaml")
        print(cfg["features"])
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def expand_defaults(cfg: dict) -> dict:
    """
    Merge a user configuration with VoxAtlas defaults.

    Parameters
    ----------
    cfg : dict
        User-supplied configuration dictionary.

    Returns
    -------
    dict
        Configuration with top-level defaults applied.

    Notes
    -----
    The merge is shallow at the top level of the configuration dictionary.

    Examples
    --------
    Usage example::

        cfg = expand_defaults({"features": ["acoustic.pitch.f0"]})
        print(cfg["pipeline"])
    """
    final = deepcopy(DEFAULT_CONFIG)
    final.update(cfg)
    return final


def load_and_prepare_config(path: str) -> dict:
    """
    Load, validate, and normalize a VoxAtlas configuration.

    Parameters
    ----------
    path : str
        Filesystem path to a YAML configuration file.

    Returns
    -------
    dict
        Validated configuration with defaults applied.

    Raises
    ------
    ConfigValidationError
        Raised when the configuration does not satisfy the expected schema.

    Notes
    -----
    This is the recommended configuration entry point for the CLI and tutorial
    workflows.

    Examples
    --------
    Usage example::

        config = load_and_prepare_config("config.yaml")
        print(config["features"])
    """
    cfg = load_config(path)
    validate_config(cfg)
    final_cfg = expand_defaults(cfg)
    return final_cfg
