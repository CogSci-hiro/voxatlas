import re

from voxatlas.features.base_extractor import BaseExtractor


FEATURE_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,}$"
)
VALID_UNITS = {
    "frame",
    "token",
    "phoneme",
    "syllable",
    "sentence",
    "word",
    "ipu",
    "turn",
    "conversation",
}


def validate_feature_name(name):
    """
    Validate feature name against VoxAtlas API rules.
    
    This function supports the public feature registry contract used to discover, validate, and resolve extractor classes before execution.
    
    Parameters
    ----------
    name : object
        Fully qualified VoxAtlas feature name or family label.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.registry.validator import validate_feature_name
    >>> validate_feature_name("acoustic.pitch.dummy") is None
    True
    >>> validate_feature_name("NotAFeature")
    Traceback (most recent call last):
    ...
    ValueError: Feature name must follow the format 'domain.feature' with optional additional suffix segments
    """
    if not isinstance(name, str) or not FEATURE_NAME_PATTERN.match(name):
        raise ValueError(
            "Feature name must follow the format 'domain.feature' "
            "with optional additional suffix segments"
        )


def validate_units(input_units, output_units):
    """
    Validate units against VoxAtlas API rules.
    
    This function supports the public feature registry contract used to discover, validate, and resolve extractor classes before execution.
    
    Parameters
    ----------
    input_units : object
        Source unit level consumed by the feature, or ``None`` when the extractor works directly from audio or global context.
    output_units : object
        Target unit level produced by the feature, or ``None`` when the output is not aligned to a unit table.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.registry.validator import validate_units
    >>> validate_units(None, "conversation") is None
    True
    >>> validate_units("banana", "conversation")
    Traceback (most recent call last):
    ...
    ValueError: input_units must be one of ['conversation', 'frame', 'ipu', 'phoneme', 'sentence', 'syllable', 'token', 'turn', 'word'] or None
    """
    for label, value in {
        "input_units": input_units,
        "output_units": output_units,
    }.items():
        if value is not None and value not in VALID_UNITS:
            raise ValueError(
                f"{label} must be one of {sorted(VALID_UNITS)} or None"
            )


def validate_dependencies(dependencies):
    """
    Validate dependencies against VoxAtlas API rules.
    
    This function supports the public feature registry contract used to discover, validate, and resolve extractor classes before execution.
    
    Parameters
    ----------
    dependencies : object
        Ordered list of upstream feature names that must be computed before the current feature is executed.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.registry.validator import validate_dependencies
    >>> validate_dependencies(["acoustic.pitch.dummy"]) is None
    True
    """
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be a list")

    for dependency in dependencies:
        validate_feature_name(dependency)


def validate_extractor_contract(extractor_cls):
    """
    Validate extractor contract against VoxAtlas API rules.
    
    This function supports the public feature registry contract used to discover, validate, and resolve extractor classes before execution.
    
    Parameters
    ----------
    extractor_cls : object
        Extractor class to validate or register with the central feature registry.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.features.example_pitch import DummyPitchExtractor
    >>> from voxatlas.registry.validator import validate_extractor_contract
    >>> validate_extractor_contract(DummyPitchExtractor) is None
    True
    """
    if not isinstance(extractor_cls, type) or not issubclass(extractor_cls, BaseExtractor):
        raise ValueError("extractor_cls must be a subclass of BaseExtractor")

    if not getattr(extractor_cls, "name", None):
        raise ValueError("Extractor must define a name")

    if not hasattr(extractor_cls, "compute") or not callable(getattr(extractor_cls, "compute")):
        raise ValueError("Extractor must define a callable compute method")

    if not hasattr(extractor_cls, "dependencies"):
        raise ValueError("Extractor must define dependencies")

    validate_feature_name(extractor_cls.name)
    validate_dependencies(extractor_cls.dependencies)
    validate_units(
        getattr(extractor_cls, "input_units", None),
        getattr(extractor_cls, "output_units", None),
    )
