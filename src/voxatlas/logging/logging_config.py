import logging


def configure_logging(level="INFO"):
    """
    Provide the ``configure_logging`` public API.
    
    This public function belongs to the logging layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    level : object
        Logging level string such as ``INFO`` or ``DEBUG``.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> import logging
    >>> from voxatlas.logging.logging_config import configure_logging
    >>> configure_logging(level="WARNING") is None
    True
    >>> logging.getLevelName(logging.getLogger().level)
    'WARNING'
    """
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def get_logger(name):
    """
    Provide the ``get_logger`` public API.
    
    This public function belongs to the logging layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
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
    >>> from voxatlas.logging.logging_config import get_logger
    >>> logger = get_logger("voxatlas.example")
    >>> logger.name
    'voxatlas.example'
    """
    return logging.getLogger(name)
