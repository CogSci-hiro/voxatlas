class ConfigError(Exception):
    """
    Represent the config error concept in VoxAtlas.
    
    This public class exposes reusable state or behavior for the config layer of VoxAtlas. It is part of the supported API surface and is intended to be composed by pipelines, registries, and feature extractors.
    
    Examples
    --------
        from voxatlas.config.exceptions import ConfigError
    
        obj = ConfigError()
        print(obj)
    """
    pass


class ConfigValidationError(ConfigError):
    """
    Represent the config validation error concept in VoxAtlas.
    
    This public class exposes reusable state or behavior for the config layer of VoxAtlas. It is part of the supported API surface and is intended to be composed by pipelines, registries, and feature extractors.
    
    Examples
    --------
        from voxatlas.config.exceptions import ConfigValidationError
    
        obj = ConfigValidationError()
        print(obj)
    """
    pass