from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


FUNCTION_POS_TAGS = {
    "ADP",
    "AUX",
    "CCONJ",
    "SCONJ",
    "DET",
    "PART",
    "PRON",
    "PUNCT",
    "INTJ",
    "IN",
    "DT",
    "CC",
    "TO",
    "MD",
    "PRP",
    "PRP$",
    "WP",
    "WP$",
    "WDT",
    "UH",
}


def _default_resource_root() -> Path:
    return Path(__file__).resolve().parents[3] / "resources" / "lexical"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_lookup(table: pd.DataFrame, value_column: str) -> dict[str, object]:
    if table.empty or value_column not in table.columns:
        return {}

    lookup = {}
    for _, row in table.iterrows():
        lemma = row.get("lemma", row.get("word", row.get("token")))
        value = row.get(value_column)
        if lemma is None or pd.isna(lemma) or value is None or pd.isna(value):
            continue
        lookup[str(lemma).strip().lower()] = value
    return lookup


@lru_cache(maxsize=None)
def load_lexical_property_resources(language=None, resource_root=None):
    """
    Load lexical property resources for VoxAtlas processing.
    
    This public function belongs to the lexical layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    language : object
        Argument used by the lexical API.
    resource_root : object
        Argument used by the lexical API.
    
    Returns
    -------
    dict
        Dictionary containing structured metadata or feature values.
    
    Examples
    --------
    >>> from voxatlas.lexical.property_utils import load_lexical_property_resources
    >>> resources = load_lexical_property_resources(language="")
    >>> sorted(resources.keys())
    ['animacy', 'concreteness']
    """
    language = (language or "").strip()
    root = Path(resource_root) if resource_root is not None else _default_resource_root()

    if not language:
        return {"animacy": {}, "concreteness": {}}

    language_dir = root / "languages" / str(language)
    animacy = _read_csv_if_exists(language_dir / "animacy.csv")
    concreteness = _read_csv_if_exists(language_dir / "concreteness.csv")

    return {
        "animacy": _build_lookup(animacy, "animacy"),
        "concreteness": _build_lookup(concreteness, "concreteness"),
    }


def _get_token_key(token_row):
    for key in ("lemma", "token_normalized", "token_canonical", "text", "token", "form", "label", "surface", "word"):
        value = token_row.get(key)
        if value is not None and value == value:
            normalized = str(value).strip().lower()
            if normalized:
                return normalized
    return None


def _get_pos(token_row):
    for key in ("upos", "pos", "POS", "xpos"):
        value = token_row.get(key)
        if value is not None and value == value:
            normalized = str(value).strip()
            if normalized:
                return normalized
    return None


def classify_function_word(pos_tag):
    """
    Provide the ``classify_function_word`` public API.
    
    This public function belongs to the lexical layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    pos_tag : object
        Argument used by the lexical API.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> from voxatlas.lexical.property_utils import classify_function_word
    >>> float(classify_function_word("DET"))
    1.0
    """
    if pos_tag is None:
        return np.float32(np.nan)
    return np.float32(1.0 if str(pos_tag).upper() in FUNCTION_POS_TAGS else 0.0)


def lookup_lexical_properties(tokens, resources):
    """
    Provide the ``lookup_lexical_properties`` public API.
    
    This public function belongs to the lexical layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    resources : object
        Argument used by the lexical API.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
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
    >>> from voxatlas.lexical.property_utils import lookup_lexical_properties
    >>> tokens = pd.DataFrame([{"id": 1, "lemma": "dog", "upos": "NOUN"}])
    >>> resources = {"animacy": {"dog": 1.0}, "concreteness": {"dog": 4.5}}
    >>> out = lookup_lexical_properties(tokens, resources)
    >>> out.loc[0, ["id", "animacy", "concreteness"]].to_dict()
    {'id': 1.0, 'animacy': 1.0, 'concreteness': 4.5}
    """
    if tokens is None:
        raise ValueError("Lexical property features require token units")

    rows = []

    for _, token_row in tokens.iterrows():
        token_id = token_row.get("id", token_row.name)
        token_key = _get_token_key(token_row)
        pos_tag = _get_pos(token_row)

        rows.append(
            {
                "id": token_id,
                "pos": pos_tag,
                "function_word": classify_function_word(pos_tag),
                "animacy": resources.get("animacy", {}).get(token_key, np.nan),
                "concreteness": resources.get("concreteness", {}).get(token_key, np.nan),
            }
        )

    return pd.DataFrame(rows)
