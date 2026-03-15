from .units import Units


def load_alignment(path: str) -> Units:
    """
    Load a single alignment file into a :class:`Units` container.

    Parameters
    ----------
    path : str
        Filesystem path to an alignment file.

    Returns
    -------
    Units
        Unit container built from the alignment source.

    Notes
    -----
    This helper is currently a placeholder and returns an empty ``Units``
    object. The higher-level dataset loader is the recommended public entry
    point for working pipelines.

    Examples
    --------
    Usage example::

        units = load_alignment("alignment.TextGrid")
        print(units)
    """
    return Units()
