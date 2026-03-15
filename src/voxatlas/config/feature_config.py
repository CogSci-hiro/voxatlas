from copy import deepcopy


def resolve_feature_config(feature_name, extractor_cls, config):
    """
    Provide the ``resolve_feature_config`` public API.
    
    This public function belongs to the config layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    feature_name : object
        Fully qualified VoxAtlas feature name, such as ``acoustic.pitch.f0``.
    extractor_cls : object
        Extractor class to validate or register with the central feature registry.
    config : object
        Pipeline or feature configuration dictionary that controls dependency resolution, execution, and algorithm parameters.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.config.feature_config import resolve_feature_config
    >>> from voxatlas.features.example_pitch import DummyPitchExtractor
    >>> resolve_feature_config(
    ...     "acoustic.pitch.dummy",
    ...     DummyPitchExtractor,
    ...     {"feature_config": {"acoustic.pitch.dummy": {"example_param": 1}}},
    ... )
    {'example_param': 1}
    """
    params = deepcopy(getattr(extractor_cls, "default_config", {}))
    user_params = config.get("feature_config", {}).get(feature_name, {})
    params.update(user_params)
    return params
