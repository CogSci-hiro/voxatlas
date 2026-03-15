from pathlib import Path

import pandas as pd


REQUIRED_MAPPING_COLUMNS = {
    "mapping_id",
    "surface",
    "canonical",
    "rule",
    "confidence",
}


def load_mapping_table(path):
    """
    Load mapping table for VoxAtlas processing.
    
    This public function belongs to the nlp layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    path : object
        Filesystem path pointing to an audio file, alignment file, cache file, or resource file.
    
    Returns
    -------
    object
        Loaded resource object ready for downstream VoxAtlas stages.
    
    Examples
    --------
        value = load_mapping_table(path=...)
        print(value)
    """
    mapping = pd.read_csv(Path(path))
    missing_columns = REQUIRED_MAPPING_COLUMNS - set(mapping.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Invalid mapping table file. Missing columns: {missing}"
        )

    return mapping
