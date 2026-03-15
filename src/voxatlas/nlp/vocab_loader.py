from pathlib import Path

import pandas as pd


REQUIRED_VOCAB_COLUMNS = {
    "token_id",
    "canonical",
    "lemma",
    "pos",
    "token_type",
    "source",
}


def load_canonical_vocab(path):
    """
    Load canonical vocab for VoxAtlas processing.
    
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
        value = load_canonical_vocab(path=...)
        print(value)
    """
    vocab = pd.read_csv(Path(path))
    missing_columns = REQUIRED_VOCAB_COLUMNS - set(vocab.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Invalid canonical vocabulary file. Missing columns: {missing}"
        )

    return vocab
