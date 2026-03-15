from copy import deepcopy

import yaml

from .defaults import DEFAULT_CONFIG
from .schema import validate_config


def load_config(path: str) -> dict:
    """
    Load a VoxAtlas YAML configuration file.

    Expected YAML Format
    --------------------
    VoxAtlas configuration files are YAML mappings (YAML "dicts") with a small
    set of conventional top-level keys. The minimal valid config contains a
    ``features`` list:

    .. code-block:: yaml

        features:
          - acoustic.pitch.dummy

    Optional keys supported by the pipeline and config layer include:

    - ``pipeline``: pipeline runtime options (mapping)
      - ``n_jobs``: number of worker processes per dependency layer (int)
      - ``cache``: enable/disable on-disk feature caching (bool)
      - ``cache_dir``: cache directory when caching is enabled (str)
    - ``feature_config``: per-feature parameter overrides (mapping)
      - keys are feature names from ``features``
      - values are extractor-specific parameter mappings

    Example with per-feature parameters and pipeline options:

    .. code-block:: yaml

        features:
          - phonology.prosody.stressed
          - acoustic.pitch.f0

        pipeline:
          n_jobs: 4
          cache: true
          cache_dir: .voxatlas_cache

        feature_config:
          phonology.prosody.stressed:
            language: fra
            resource_root: /path/to/resources/phonology

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
    validation. For the recommended entry point that validates and applies
    defaults, see :func:`load_and_prepare_config`.

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
