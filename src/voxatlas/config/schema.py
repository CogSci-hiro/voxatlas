from .exceptions import ConfigValidationError


def validate_config(cfg: dict) -> None:
    """
    Validate config against VoxAtlas API rules.
    
    This public function belongs to the config layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    cfg : dict
        Dictionary of configuration values, metadata, or structured intermediate results.
    
    Returns
    -------
    None
        Return value produced by ``validate_config``.
    
    Examples
    --------
        value = validate_config(cfg=...)
        print(value)
    """

    if "features" not in cfg:
        raise ConfigValidationError(
            "Config must contain 'features' field"
        )

    if not isinstance(cfg["features"], list):
        raise ConfigValidationError(
            "'features' must be a list"
        )

    feature_config = cfg.get("feature_config", {})

    if not isinstance(feature_config, dict):
        raise ConfigValidationError(
            "'feature_config' must be a dict"
        )

    unknown_features = set(feature_config.keys()) - set(cfg["features"])

    if unknown_features:
        raise ConfigValidationError(
            "'feature_config' keys must be a subset of 'features'"
        )
