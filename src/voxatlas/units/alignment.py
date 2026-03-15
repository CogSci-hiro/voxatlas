from .units import Units


def load_alignment(path: str) -> Units:
    """
    Load an alignment file into a :class:`Units` container.

    This is a lightweight compatibility entry point for alignment ingestion.
    The current implementation returns an empty ``Units`` object and does not
    parse the file content yet.

    Parameters
    ----------
    path : str
        Filesystem path to an alignment file (for example, a TextGrid file).
        The path is accepted for API consistency, even though content parsing is
        not implemented in this helper yet.

    Returns
    -------
    Units
        An empty ``Units`` container.

    Notes
    -----
    For full data loading workflows, prefer higher-level input loading helpers
    that combine audio, alignment, and metadata validation.

    Examples
    --------
    >>> from voxatlas.units.alignment import load_alignment
    >>> from voxatlas.units.units import Units
    >>> units = load_alignment("alignment.TextGrid")
    >>> isinstance(units, Units)
    True
    """
    return Units()
