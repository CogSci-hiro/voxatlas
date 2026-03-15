from __future__ import annotations

import pandas as pd


def _normalize_phone(phone):
    if phone is None:
        return None
    return str(phone).strip()


def _word_key_candidates(word_row):
    candidates = []

    for key in ("label", "text", "token", "word", "surface"):
        value = word_row.get(key)
        if value is not None and value == value:
            candidates.append(str(value))

    return candidates


def get_expected_pronunciation(word_row, pronunciation_dictionary):
    """
    Provide the ``get_expected_pronunciation`` public API.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    word_row : object
        Argument used by the phonology API.
    pronunciation_dictionary : object
        Argument used by the phonology API.
    
    Returns
    -------
    list
        List of values derived from the supplied inputs.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.phonology.reduction_utils import get_expected_pronunciation
    >>> word_row = pd.Series({"label": "hello"})
    >>> get_expected_pronunciation(word_row, {"hello": "h ə l oʊ"})
    ['h', 'ə', 'l', 'oʊ']
    """
    if pronunciation_dictionary is None:
        raise ValueError("Pronunciation dictionary is required for reduction features")

    if isinstance(pronunciation_dictionary, pd.DataFrame):
        lookup = {}
        for _, row in pronunciation_dictionary.iterrows():
            orth = row.get("word", row.get("orthography", row.get("token")))
            pronunciation = row.get(
                "pronunciation",
                row.get("phonemes", row.get("segments")),
            )
            if orth is not None and pronunciation is not None:
                lookup[str(orth)] = pronunciation
        pronunciation_dictionary = lookup

    for candidate in _word_key_candidates(word_row):
        if candidate in pronunciation_dictionary:
            value = pronunciation_dictionary[candidate]
            if isinstance(value, str):
                return [_normalize_phone(phone) for phone in value.split() if phone.strip()]
            return [_normalize_phone(phone) for phone in value if _normalize_phone(phone)]

    return []


def get_observed_pronunciation(phonemes, word_row):
    """
    Provide the ``get_observed_pronunciation`` public API.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    phonemes : object
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    word_row : object
        Argument used by the phonology API.
    
    phonemes example
    ----------------
    phoneme_id | start | end | label | word_id
    0 | 0.12 | 0.18 | h | 0
    1 | 0.18 | 0.25 | eh | 0
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.phonology.reduction_utils import get_observed_pronunciation
    >>> phonemes = pd.DataFrame([{"word_id": 1, "label": "h"}, {"word_id": 1, "label": "i"}])
    >>> word_row = pd.Series({"id": 1})
    >>> get_observed_pronunciation(phonemes, word_row)
    ['h', 'i']
    """
    if phonemes is None:
        raise ValueError("Observed phoneme alignment is required for reduction features")

    if "word_id" in phonemes.columns:
        word_id = word_row.get("id", word_row.name)
        observed = phonemes.loc[phonemes["word_id"] == word_id]
    elif {"start", "end"}.issubset(phonemes.columns) and {"start", "end"}.issubset(word_row.index):
        observed = phonemes.loc[
            (phonemes["start"] >= word_row["start"]) &
            (phonemes["end"] <= word_row["end"])
        ]
    else:
        raise ValueError(
            "Unable to map phonemes to words. Expected 'word_id' in phonemes "
            "or time-aligned 'start'/'end' columns."
        )

    labels = []
    for _, row in observed.iterrows():
        label = row.get("label", row.get("text", row.get("phoneme")))
        if label is not None and label == label:
            labels.append(_normalize_phone(label))

    return labels


def align_phoneme_sequences(expected, observed):
    """
    Provide the ``align_phoneme_sequences`` public API.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    expected : object
        Argument used by the phonology API.
    observed : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.phonology.reduction_utils import align_phoneme_sequences
    >>> align_phoneme_sequences(["a", "b"], ["a"])
    [('a', 'a'), ('b', None)]
    """
    expected = list(expected)
    observed = list(observed)
    n_expected = len(expected)
    n_observed = len(observed)

    dp = [[0] * (n_observed + 1) for _ in range(n_expected + 1)]
    back = [[None] * (n_observed + 1) for _ in range(n_expected + 1)]

    for i in range(1, n_expected + 1):
        dp[i][0] = i
        back[i][0] = "delete"

    for j in range(1, n_observed + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, n_expected + 1):
        for j in range(1, n_observed + 1):
            match_cost = 0 if expected[i - 1] == observed[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + match_cost, "match"),
                (dp[i - 1][j] + 1, "delete"),
                (dp[i][j - 1] + 1, "insert"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    alignment = []
    i = n_expected
    j = n_observed

    while i > 0 or j > 0:
        action = back[i][j]

        if action == "match":
            alignment.append((expected[i - 1], observed[j - 1]))
            i -= 1
            j -= 1
        elif action == "delete":
            alignment.append((expected[i - 1], None))
            i -= 1
        elif action == "insert":
            alignment.append((None, observed[j - 1]))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment


def detect_schwa_deletion(alignment, schwa_symbols=None):
    """
    Detect schwa deletion from aligned annotations.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    alignment : object
        Argument used by the phonology API.
    schwa_symbols : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.phonology.reduction_utils import detect_schwa_deletion
    >>> detect_schwa_deletion([("ə", None)])
    1.0
    """
    schwa_symbols = schwa_symbols or ["ə", "@", "schwa"]
    schwa_symbols = {str(symbol) for symbol in schwa_symbols}

    for expected_phone, observed_phone in alignment:
        if expected_phone in schwa_symbols and observed_phone is None:
            return 1.0

    return 0.0


def detect_vowel_reduction(alignment, central_vowels=None, schwa_symbols=None):
    """
    Detect vowel reduction from aligned annotations.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    alignment : object
        Argument used by the phonology API.
    central_vowels : object
        Argument used by the phonology API.
    schwa_symbols : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.phonology.reduction_utils import detect_vowel_reduction
    >>> detect_vowel_reduction([("i", "ə")])
    1.0
    """
    central_vowels = central_vowels or ["ə", "ɘ", "ɜ", "ɐ"]
    schwa_symbols = schwa_symbols or ["ə", "@", "schwa"]
    central_vowels = {str(vowel) for vowel in central_vowels}
    schwa_symbols = {str(symbol) for symbol in schwa_symbols}

    for expected_phone, observed_phone in alignment:
        if expected_phone is None or observed_phone is None:
            continue

        if expected_phone == observed_phone:
            continue

        if observed_phone in central_vowels and expected_phone not in schwa_symbols:
            return 1.0

    return 0.0
