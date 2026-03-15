from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_frequency_lexicon(language=None, resource_root=None, lexicon=None):
    r"""
    Load a lemma-frequency lexicon for lexical frequency features.
    
    Parameters
    ----------
    language : str, optional
        Language identifier used to choose a packaged lexicon.
    resource_root : str or pathlib.Path, optional
        Root directory that contains VoxAtlas lexical resources.
    lexicon : pandas.DataFrame, optional
        In-memory override lexicon.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with at least ``lemma`` and ``frequency`` columns.
    
    Algorithm
    ---------
    If an explicit lexicon is supplied it is copied directly. Otherwise VoxAtlas resolves the packaged resource path ``resources/lexical/languages/<language>/lexicon.csv`` and loads the corresponding table.
    
    Examples
    --------
        lexicon = load_frequency_lexicon(language="en", resource_root="resources")
        print(lexicon.head())
    
    """
    if lexicon is not None:
        if isinstance(lexicon, pd.DataFrame):
            return lexicon.copy()
        raise ValueError("lexicon must be a pandas DataFrame")

    if language is None or resource_root is None:
        return pd.DataFrame(columns=["lemma", "frequency"])

    path = (
        Path(resource_root)
        / "resources"
        / "lexical"
        / "languages"
        / str(language)
        / "lexicon.csv"
    )

    if not path.exists():
        return pd.DataFrame(columns=["lemma", "frequency"])

    return pd.read_csv(path)


def _build_frequency_lookup(lexicon):
    lookup = {}

    for _, row in lexicon.iterrows():
        lemma = row.get("lemma", row.get("word", row.get("token")))
        frequency = row.get(
            "frequency",
            row.get("freq_per_million", row.get("word_frequency")),
        )

        if lemma is None or pd.isna(lemma):
            continue

        if frequency is None or pd.isna(frequency):
            continue

        lookup[str(lemma)] = float(frequency)

    return lookup


def _get_lemma(token_row):
    for key in ("lemma", "Lemma", "text", "token", "form", "label"):
        value = token_row.get(key)
        if value is not None and value == value:
            return str(value)
    return None


def lookup_word_frequency(tokens, lexicon):
    r"""
    Look up raw lexical frequency for each token.
    
    Parameters
    ----------
    tokens : pandas.DataFrame
        Token table whose rows provide lemma-like forms.
    lexicon : pandas.DataFrame or object
        Frequency lexicon or an object that can be resolved into one.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    Returns
    -------
    pandas.Series
        Float32 raw-frequency series indexed by token identifier.
    
    Algorithm
    ---------
    Each token contributes a lemma-like string :math:`w_i`; the returned value is
    
    .. math::
    
       f_i = L(w_i),
    
    where :math:`L` is the frequency lexicon lookup map.
    
    Examples
    --------
        freq = lookup_word_frequency(tokens, lexicon)
        print(freq.head())
    
    """
    if tokens is None:
        raise ValueError("Frequency features require token units")

    lexicon = load_frequency_lexicon(lexicon=lexicon) if not isinstance(lexicon, pd.DataFrame) else lexicon
    lookup = _build_frequency_lookup(lexicon)

    values = []
    index = []

    for _, token_row in tokens.iterrows():
        lemma = _get_lemma(token_row)
        values.append(lookup.get(lemma, np.nan))
        index.append(token_row.get("id", token_row.name))

    return pd.Series(values, index=index, dtype="float32")


def compute_zipf_frequency(raw_frequency):
    r"""
    Convert raw frequencies to the Zipf scale.
    
    Parameters
    ----------
    raw_frequency : array-like
        Raw frequency values.
    
    Returns
    -------
    pandas.Series
        Float32 Zipf-scaled frequencies.
    
    Algorithm
    ---------
    The Zipf transform is
    
    .. math::
    
       z_i = \log_{10}(f_i) + 3.
    
    Non-finite values are replaced with ``NaN``.
    
    Examples
    --------
        zipf = compute_zipf_frequency(raw_frequency)
        print(zipf.head())
    
    """
    raw_frequency = pd.Series(raw_frequency, copy=False)
    zipf = np.log10(raw_frequency.astype("float32")) + 3.0
    zipf[~np.isfinite(zipf)] = np.nan
    return zipf.astype("float32")
