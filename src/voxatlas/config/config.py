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
    >>> import tempfile
    >>> from pathlib import Path
    >>> from voxatlas.config import load_config
    >>> yaml_text = "features:\\n  - acoustic.pitch.dummy\\n"
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "config.yaml"
    ...     _ = path.write_text(yaml_text, encoding="utf-8")
    ...     cfg = load_config(str(path))
    ...     cfg["features"]
    ['acoustic.pitch.dummy']
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def expand_defaults(cfg: dict) -> dict:
    """
    Merge a user configuration with VoxAtlas defaults.

    What "Expand Defaults" Means
    ----------------------------
    VoxAtlas maintains a small built-in default configuration
    (:data:`voxatlas.config.defaults.DEFAULT_CONFIG`). ``expand_defaults``
    starts from a deep copy of that default mapping and then applies the user
    configuration on top.

    This is a **shallow top-level merge**:

    - Only the first level of keys is merged (via ``dict.update``).
    - If the user provides a top-level key, it **replaces** the default value
      for that key entirely.
    - Nested mappings are **not** deep-merged. For example, providing a
      ``pipeline`` mapping replaces the whole default ``pipeline`` mapping.

    Concretely, given the default:

    .. code-block:: python

        {"features": [], "pipeline": {"cache": True}}

    The following user config:

    .. code-block:: python

        {"pipeline": {"n_jobs": 4}}

    Produces:

    .. code-block:: python

        {"features": [], "pipeline": {"n_jobs": 4}}

    (note how ``pipeline.cache`` is not preserved because nested dicts are not
    merged).

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
    If you want to override just one pipeline option while keeping other
    defaults, pass the full desired ``pipeline`` mapping (or use
    :func:`load_and_prepare_config`, which is the recommended config entry
    point for most workflows).

    Examples
    --------
    >>> from voxatlas.config import expand_defaults
    >>> cfg = expand_defaults({"features": ["acoustic.pitch.dummy"]})
    >>> cfg["features"]
    ['acoustic.pitch.dummy']
    >>> sorted(cfg["pipeline"].keys())
    ['cache']
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
    >>> import tempfile
    >>> from pathlib import Path
    >>> from voxatlas.config import load_and_prepare_config
    >>> yaml_text = "features:\\n  - acoustic.pitch.dummy\\n"
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "config.yaml"
    ...     _ = path.write_text(yaml_text, encoding="utf-8")
    ...     cfg = load_and_prepare_config(str(path))
    ...     cfg["features"]
    ['acoustic.pitch.dummy']
    """
    cfg = load_config(path)
    validate_config(cfg)
    final_cfg = expand_defaults(cfg)
    return final_cfg
