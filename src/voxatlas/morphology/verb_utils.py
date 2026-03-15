import pandas as pd


def extract_verb_morphology_features(inflection_table):
    """
    Extract verb morphology features from structured VoxAtlas data.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    inflection_table : object
        Argument used by the morphology API.
    
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
    >>> from voxatlas.morphology.verb_utils import extract_verb_morphology_features
    >>> inflection = pd.DataFrame([{"id": 1, "upos": "VERB", "VerbForm": "Fin"}])
    >>> out = extract_verb_morphology_features(inflection)
    >>> out.loc[0, ["Finite", "Participle", "Infinitive"]].to_dict()
    {'Finite': 1.0, 'Participle': 0.0, 'Infinitive': 0.0}
    """
    if inflection_table is None:
        raise ValueError("Verb morphology features require inflection features")

    rows = []

    for _, row in inflection_table.iterrows():
        verb_form = row.get("VerbForm")
        pos_value = row.get("upos", row.get("pos", row.get("POS")))

        rows.append(
            {
                "id": row.get("id", row.name),
                "VerbForm": verb_form,
                "is_verb": pos_value in {"VERB", "AUX"} if pos_value is not None else True,
                "Finite": 1.0 if verb_form == "Fin" else 0.0,
                "Participle": 1.0 if verb_form == "Part" else 0.0,
                "Infinitive": 1.0 if verb_form == "Inf" else 0.0,
            }
        )

    return pd.DataFrame(rows)
