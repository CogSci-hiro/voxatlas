from __future__ import annotations

import pandas as pd


UD_FEATURES = [
    "Tense",
    "Aspect",
    "Mood",
    "VerbForm",
    "Person",
    "Number",
    "Gender",
    "Case",
    "Definite",
]


def _normalize_feature_dict(value):
    if isinstance(value, dict):
        return dict(value)

    if isinstance(value, str):
        features = {}
        for item in value.split("|"):
            if "=" not in item:
                continue
            key, feature_value = item.split("=", 1)
            features[key] = feature_value
        return features

    return {}


def _find_token_text(token_row):
    for key in ("text", "token", "form", "label", "surface"):
        value = token_row.get(key)
        if value is not None and value == value:
            return str(value)
    return None


def _find_lemma(token_row):
    for key in ("lemma", "Lemma"):
        value = token_row.get(key)
        if value is not None and value == value:
            return str(value)
    return None


def _lookup_morph_features(token_row, resources):
    lemma = _find_lemma(token_row)
    token_text = _find_token_text(token_row)

    feature_map = resources.get("morphology_lexicon", {}) or {}
    analyses = resources.get("morphological_analysis", {}) or {}

    if lemma is not None and lemma in analyses:
        return _normalize_feature_dict(analyses[lemma])

    if token_text is not None and token_text in analyses:
        return _normalize_feature_dict(analyses[token_text])

    if lemma is not None and lemma in feature_map:
        return _normalize_feature_dict(feature_map[lemma])

    if token_text is not None and token_text in feature_map:
        return _normalize_feature_dict(feature_map[token_text])

    return {}


def extract_inflection_features(tokens, resources=None):
    """
    Extract inflection features from structured VoxAtlas data.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    resources : object
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
    >>> from voxatlas.morphology.inflection_utils import extract_inflection_features
    >>> tokens = pd.DataFrame([{"id": 1, "lemma": "be"}])
    >>> resources = {"morphological_analysis": {"be": "VerbForm=Fin|Tense=Pres"}}
    >>> out = extract_inflection_features(tokens, resources=resources)
    >>> out.loc[0, ["VerbForm", "Tense"]].to_dict()
    {'VerbForm': 'Fin', 'Tense': 'Pres'}
    """
    if tokens is None:
        raise ValueError("Inflection features require token units")

    resources = resources or {}
    rows = []

    for _, token_row in tokens.iterrows():
        morph_features = _lookup_morph_features(token_row, resources)
        row = {
            "id": token_row.get("id", token_row.name),
        }

        for feature_name in UD_FEATURES:
            row[feature_name] = morph_features.get(feature_name)

        rows.append(row)

    return pd.DataFrame(rows)
