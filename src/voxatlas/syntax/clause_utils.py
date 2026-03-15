from __future__ import annotations

import pandas as pd


REQUIRED_CLAUSE_COLUMNS = ("id", "deprel")


def ensure_clause_columns(table):
    """
    Provide the ``ensure_clause_columns`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : object
        Tabular annotation aligned to a VoxAtlas unit level.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
        value = ensure_clause_columns(table=...)
        print(value)
    """
    missing = [column for column in REQUIRED_CLAUSE_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            "Clause structure features require dependency columns: "
            + ", ".join(missing)
        )


def compute_clause_membership(table, labels):
    """
    Compute clause membership from VoxAtlas inputs.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : object
        Tabular annotation aligned to a VoxAtlas unit level.
    labels : object
        Collection of labels or dependency relations used to derive a higher-level annotation.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Returns
    -------
    pandas.Series
        One-dimensional values aligned to the index of the supplied table.
    
    Examples
    --------
        value = compute_clause_membership(table=..., labels=...)
        print(value)
    """
    ensure_clause_columns(table)

    normalized_labels = {label.lower() for label in labels}
    dep_labels = table["deprel"].fillna("").astype(str).str.lower()
    values = dep_labels.isin(normalized_labels).astype("int8")

    return pd.Series(values.to_numpy(), index=table["id"], dtype="int8")
