from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_clitic_list(language=None, resource_root=None, clitic_list=None):
    """
    Load clitic list for VoxAtlas processing.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    language : object
        Argument used by the morphology API.
    resource_root : object
        Argument used by the morphology API.
    clitic_list : object
        Argument used by the morphology API.
    
    Returns
    -------
    object
        Loaded resource object ready for downstream VoxAtlas stages.
    
    Examples
    --------
    >>> from voxatlas.morphology.word_formation_utils import load_clitic_list
    >>> load_clitic_list(clitic_list=["l", "d"]) == {"l", "d"}
    True
    """
    if clitic_list is not None:
        return {str(item).strip() for item in clitic_list}

    if language is None or resource_root is None:
        return set()

    path = (
        Path(resource_root)
        / "resources"
        / "morphology"
        / "languages"
        / str(language)
        / "clitics.csv"
    )

    if not path.exists():
        return set()

    table = pd.read_csv(path)
    column = "clitic" if "clitic" in table.columns else table.columns[0]
    return {str(value).strip() for value in table[column].dropna().tolist()}


def _get_token_text(token_row):
    for key in ("text", "token", "form", "label", "surface"):
        value = token_row.get(key)
        if value is not None and value == value:
            return str(value)
    return ""


def _get_segmentation(token_row, segmentation_lookup=None):
    segmentation_lookup = segmentation_lookup or {}
    token_id = token_row.get("id", token_row.name)

    if token_id in segmentation_lookup:
        return segmentation_lookup[token_id]

    token_text = _get_token_text(token_row)
    if token_text in segmentation_lookup:
        return segmentation_lookup[token_text]

    return None


def detect_compound(token_row, lexicon_lookup=None, segmentation_lookup=None):
    """
    Detect compound from aligned annotations.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    token_row : object
        Argument used by the morphology API.
    lexicon_lookup : object
        Argument used by the morphology API.
    segmentation_lookup : object
        Argument used by the morphology API.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.morphology.word_formation_utils import detect_compound
    >>> token_row = pd.Series({"text": "ice-cream"})
    >>> detect_compound(token_row, lexicon_lookup={"ice", "cream"})
    1.0
    """
    lexicon_lookup = lexicon_lookup or set()
    token_text = _get_token_text(token_row)
    segmentation = _get_segmentation(token_row, segmentation_lookup=segmentation_lookup)

    if segmentation is None:
        for separator in ("-", "_", "+"):
            if separator in token_text:
                parts = [part for part in token_text.split(separator) if part]
                if len(parts) >= 2 and all(part in lexicon_lookup for part in parts):
                    return 1.0
        return 0.0

    if isinstance(segmentation, str):
        parts = [part.strip() for part in segmentation.replace("+", " ").split() if part.strip()]
    else:
        parts = [str(part).strip() for part in segmentation if str(part).strip()]

    if len(parts) < 2:
        return 0.0

    if not lexicon_lookup:
        return 1.0

    return 1.0 if all(part in lexicon_lookup for part in parts[:2]) else 0.0


def detect_clitic(token_row, clitic_list=None):
    """
    Detect clitic from aligned annotations.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    token_row : object
        Argument used by the morphology API.
    clitic_list : object
        Argument used by the morphology API.
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.morphology.word_formation_utils import detect_clitic
    >>> token_row = pd.Series({"text": "l'amour"})
    >>> detect_clitic(token_row, clitic_list={"l"})
    1.0
    """
    clitic_list = clitic_list or set()
    token_text = _get_token_text(token_row)

    if not token_text:
        return 0.0

    normalized = token_text.lower()

    if normalized in clitic_list:
        return 1.0

    for separator in ("'", "’", "-"):
        if separator in normalized:
            parts = [part for part in normalized.split(separator) if part]
            if any(part in clitic_list for part in parts):
                return 1.0

    return 0.0


def extract_word_formation_features(
    tokens,
    language=None,
    resource_root=None,
    clitic_list=None,
    lexicon_lookup=None,
    segmentation_lookup=None,
):
    """
    Extract word formation features from structured VoxAtlas data.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    language : object
        Argument used by the morphology API.
    resource_root : object
        Argument used by the morphology API.
    clitic_list : object
        Argument used by the morphology API.
    lexicon_lookup : object
        Argument used by the morphology API.
    segmentation_lookup : object
        Argument used by the morphology API.
    
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
    >>> from voxatlas.morphology.word_formation_utils import extract_word_formation_features
    >>> tokens = pd.DataFrame([{"id": 1, "text": "ice-cream"}])
    >>> out = extract_word_formation_features(tokens, clitic_list=["l"], lexicon_lookup=["ice", "cream"])
    >>> out.loc[0, ["Compound", "Clitic"]].to_dict()
    {'Compound': 1.0, 'Clitic': 0.0}
    """
    if tokens is None:
        raise ValueError("Word formation features require token units")

    clitic_lookup = load_clitic_list(
        language=language,
        resource_root=resource_root,
        clitic_list=clitic_list,
    )
    lexicon_lookup = set(lexicon_lookup or [])
    segmentation_lookup = segmentation_lookup or {}

    rows = []

    for _, token_row in tokens.iterrows():
        rows.append(
            {
                "id": token_row.get("id", token_row.name),
                "Compound": detect_compound(
                    token_row,
                    lexicon_lookup=lexicon_lookup,
                    segmentation_lookup=segmentation_lookup,
                ),
                "Clitic": detect_clitic(
                    token_row,
                    clitic_list=clitic_lookup,
                ),
            }
        )

    return pd.DataFrame(rows)
