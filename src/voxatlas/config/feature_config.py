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
        value = resolve_feature_config(feature_name=..., extractor_cls=..., config=...)
        print(value)
    """
    params = deepcopy(getattr(extractor_cls, "default_config", {}))
    user_params = config.get("feature_config", {}).get(feature_name, {})
    params.update(user_params)
    return params
