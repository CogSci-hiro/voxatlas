from __future__ import annotations

import pandas as pd


def _token_text(token_row: pd.Series) -> str:
    for column in ("token_normalized", "token_canonical", "token", "label", "text", "word", "surface"):
        value = token_row.get(column)
        if value is not None and value == value:
            text = str(value).strip()
            if text:
                return text
    return ""


def _select_within_span(table: pd.DataFrame, token_row: pd.Series) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame()

    token_id = token_row.get("id", token_row.name)

    parent_columns = ("token_id", "word_id")
    for column in parent_columns:
        if column in table.columns:
            return table.loc[table[column] == token_id]

    if {"start", "end"}.issubset(table.columns) and {"start", "end"}.issubset(token_row.index):
        return table.loc[
            (table["start"] >= token_row["start"]) &
            (table["end"] <= token_row["end"])
        ]

    raise ValueError(
        "Unable to map aligned units to tokens. Expected 'token_id'/'word_id' "
        "or time-aligned 'start'/'end' columns."
    )


def compute_lexical_properties(
    tokens: pd.DataFrame,
    syllables: pd.DataFrame,
    phonemes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute lexical properties from VoxAtlas inputs.
    
    This public function belongs to the lexical layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : pd.DataFrame
        Token-level annotation table used for morphological or syntactic computation.
    syllables : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    phonemes : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    syllables example
    -----------------
    syllable_id | start | end | word_id | stress
    0 | 0.12 | 0.25 | 0 | 1
    1 | 0.46 | 0.60 | 1 | 0
    
    phonemes example
    ----------------
    phoneme_id | start | end | label | word_id
    0 | 0.12 | 0.18 | h | 0
    1 | 0.18 | 0.25 | eh | 0
    
    Returns
    -------
    pandas.DataFrame
        Tabular result aligned to a VoxAtlas unit level or registry resource.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.lexical.properties_utils import compute_lexical_properties
    >>> tokens = pd.DataFrame([{"id": 1, "token": "hello"}])
    >>> syllables = pd.DataFrame([{"id": 10, "word_id": 1}, {"id": 11, "word_id": 1}])
    >>> phonemes = pd.DataFrame([{"id": 20, "word_id": 1}, {"id": 21, "word_id": 1}, {"id": 22, "word_id": 1}])
    >>> out = compute_lexical_properties(tokens=tokens, syllables=syllables, phonemes=phonemes)
    >>> out.loc[0, ["word_length", "syllable_count", "phoneme_count"]].to_dict()
    {'word_length': 5, 'syllable_count': 2, 'phoneme_count': 3}
    """
    if tokens is None:
        raise ValueError("Lexical property features require token units")

    rows = []

    for _, token_row in tokens.iterrows():
        token_text = _token_text(token_row)
        token_syllables = _select_within_span(syllables, token_row) if syllables is not None else pd.DataFrame()
        token_phonemes = _select_within_span(phonemes, token_row) if phonemes is not None else pd.DataFrame()

        rows.append(
            {
                "id": token_row.get("id"),
                "word_length": len(token_text),
                "syllable_count": int(len(token_syllables)),
                "phoneme_count": int(len(token_phonemes)),
            }
        )

    return pd.DataFrame(rows)
