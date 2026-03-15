from __future__ import annotations

import pandas as pd


SUBJECT_RELATIONS = {"nsubj", "csubj"}
GENDER_RELATIONS = {"amod", "det"}


def _id_column(tokens):
    if "id" in tokens.columns:
        return "id"
    raise ValueError("Agreement features require an 'id' column in token units")


def _head_column(tokens):
    for column in ("head", "head_id"):
        if column in tokens.columns:
            return column
    raise ValueError(
        "Agreement features require a dependency head column: 'head' or 'head_id'"
    )


def _relation_column(tokens):
    for column in ("deprel", "dep_rel", "dependency_relation"):
        if column in tokens.columns:
            return column
    raise ValueError(
        "Agreement features require a dependency relation column: "
        "'deprel', 'dep_rel', or 'dependency_relation'"
    )


def _token_lookup(tokens):
    id_col = _id_column(tokens)
    return {row[id_col]: row for _, row in tokens.iterrows()}


def identify_agreement_relations(tokens):
    """
    Provide the ``identify_agreement_relations`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    Returns
    -------
    dict
        Dictionary containing structured metadata or feature values.
    
    Examples
    --------
        value = identify_agreement_relations(tokens=...)
        print(value)
    """
    id_col = _id_column(tokens)
    head_col = _head_column(tokens)
    rel_col = _relation_column(tokens)
    lookup = _token_lookup(tokens)

    subject_pairs = []
    gender_pairs = []

    for _, row in tokens.iterrows():
        relation = row.get(rel_col)
        head_id = row.get(head_col)

        if pd.isna(head_id) or head_id not in lookup:
            continue

        head_row = lookup[head_id]
        pair = {
            "dependent_id": row[id_col],
            "head_id": head_row[id_col],
            "relation": relation,
            "dependent": row,
            "head": head_row,
        }

        if relation in SUBJECT_RELATIONS:
            subject_pairs.append(pair)

        if relation in GENDER_RELATIONS:
            gender_pairs.append(pair)

    return {
        "subject_verb": subject_pairs,
        "gender": gender_pairs,
    }


def detect_subject_verb_agreement(tokens):
    """
    Detect subject verb agreement from aligned annotations.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
        value = detect_subject_verb_agreement(tokens=...)
        print(value)
    """
    agreement = pd.Series(0.0, index=tokens["id"], dtype="float32")
    relations = identify_agreement_relations(tokens)["subject_verb"]

    for pair in relations:
        dependent = pair["dependent"]
        head = pair["head"]

        dep_person = dependent.get("Person")
        dep_number = dependent.get("Number")
        head_person = head.get("Person")
        head_number = head.get("Number")

        matches_person = (
            pd.notna(dep_person) and pd.notna(head_person) and dep_person == head_person
        )
        matches_number = (
            pd.notna(dep_number) and pd.notna(head_number) and dep_number == head_number
        )

        if matches_person or matches_number:
            agreement.loc[dependent["id"]] = 1.0
            agreement.loc[head["id"]] = 1.0

    return agreement


def detect_gender_agreement(tokens):
    """
    Detect gender agreement from aligned annotations.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
        value = detect_gender_agreement(tokens=...)
        print(value)
    """
    agreement = pd.Series(0.0, index=tokens["id"], dtype="float32")
    relations = identify_agreement_relations(tokens)["gender"]

    for pair in relations:
        dependent = pair["dependent"]
        head = pair["head"]

        dep_gender = dependent.get("Gender")
        head_gender = head.get("Gender")

        if pd.notna(dep_gender) and pd.notna(head_gender) and dep_gender == head_gender:
            agreement.loc[dependent["id"]] = 1.0
            agreement.loc[head["id"]] = 1.0

    return agreement


def extract_agreement_features(tokens):
    """
    Extract agreement features from structured VoxAtlas data.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    
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
        value = extract_agreement_features(tokens=...)
        print(value)
    """
    if tokens is None:
        raise ValueError("Agreement features require token units with dependency annotations")

    subject_verb = detect_subject_verb_agreement(tokens)
    gender = detect_gender_agreement(tokens)

    return pd.DataFrame(
        {
            "id": tokens["id"].values,
            "SubjectVerbAgreement": subject_verb.reindex(tokens["id"]).values,
            "GenderAgreement": gender.reindex(tokens["id"]).values,
        }
    )
