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
    >>> import tempfile
    >>> from pathlib import Path
    >>> from voxatlas.nlp.vocab_loader import load_canonical_vocab
    >>> csv_text = "token_id,canonical,lemma,pos,token_type,source\\n1,hello,hello,INTJ,word,example\\n"
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "vocab.csv"
    ...     _ = path.write_text(csv_text, encoding="utf-8")
    ...     vocab = load_canonical_vocab(path)
    ...     vocab.loc[0, \"canonical\"]
    'hello'
    """
    vocab = pd.read_csv(Path(path))
    missing_columns = REQUIRED_VOCAB_COLUMNS - set(vocab.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Invalid canonical vocabulary file. Missing columns: {missing}"
        )

    return vocab
