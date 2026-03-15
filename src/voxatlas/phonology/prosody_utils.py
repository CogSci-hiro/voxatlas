from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def _default_resource_root() -> Path:
    return Path(__file__).resolve().parents[3] / "resources" / "phonology"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _midpoint(row) -> float:
    return (float(row["start"]) + float(row["end"])) / 2.0


def _assign_parent_ids(
    child_table: pd.DataFrame,
    parent_table: pd.DataFrame,
    parent_name: str,
) -> pd.Series:
    values = []

    for _, child in child_table.iterrows():
        midpoint = _midpoint(child)
        match_id = pd.NA

        for _, parent in parent_table.iterrows():
            if float(parent["start"]) <= midpoint <= float(parent["end"]):
                match_id = parent["id"]
                break

        values.append(match_id)

    return pd.Series(values, index=child_table.index, dtype="Int64", name=f"{parent_name}_id")


def compute_word_positions(
    syllables: pd.DataFrame,
    words: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute word positions from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    syllables : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    words : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    
    syllables example
    -----------------
    syllable_id | start | end | word_id | stress
    0 | 0.12 | 0.25 | 0 | 1
    1 | 0.46 | 0.60 | 1 | 0
    
    words example
    -------------
    word_id | start | end | speaker | word
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
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
        value = compute_word_positions(syllables=..., words=...)
        print(value)
    """
    if syllables is None or words is None:
        return pd.DataFrame(columns=["word_id", "position_in_word"])

    word_ids = _assign_parent_ids(syllables, words, "word")
    positions = pd.Series(pd.NA, index=syllables.index, dtype="Int64", name="position_in_word")

    for word_id in word_ids.dropna().unique():
        mask = word_ids == word_id
        positions.loc[mask] = np.arange(1, int(mask.sum()) + 1, dtype=np.int64)

    return pd.DataFrame(
        {
            "word_id": word_ids,
            "position_in_word": positions,
        },
        index=syllables.index,
    )


def compute_ipu_positions(
    syllables: pd.DataFrame,
    ipus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute ipu positions from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    syllables : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    ipus : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    
    syllables example
    -----------------
    syllable_id | start | end | word_id | stress
    0 | 0.12 | 0.25 | 0 | 1
    1 | 0.46 | 0.60 | 1 | 0
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
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
        value = compute_ipu_positions(syllables=..., ipus=...)
        print(value)
    """
    if syllables is None or ipus is None:
        return pd.DataFrame(columns=["ipu_id", "position_in_ipu"])

    ipu_ids = _assign_parent_ids(syllables, ipus, "ipu")
    positions = pd.Series(pd.NA, index=syllables.index, dtype="Int64", name="position_in_ipu")

    for ipu_id in ipu_ids.dropna().unique():
        mask = ipu_ids == ipu_id
        positions.loc[mask] = np.arange(1, int(mask.sum()) + 1, dtype=np.int64)

    return pd.DataFrame(
        {
            "ipu_id": ipu_ids,
            "position_in_ipu": positions,
        },
        index=syllables.index,
    )


@lru_cache(maxsize=None)
def load_stress_rules(
    language: str | None = None,
    resource_root: str | Path | None = None,
) -> dict[str, str]:
    """
    Load stress rules for VoxAtlas processing.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    language : str | None
        Argument used by the phonology API.
    resource_root : str | Path | None
        Filesystem path used by this API.
    
    Returns
    -------
    dict[str, str]
        Return value produced by ``load_stress_rules``.
    
    Examples
    --------
        value = load_stress_rules(language=..., resource_root=...)
        print(value)
    """
    language = (language or "").strip()
    root = Path(resource_root) if resource_root is not None else _default_resource_root()

    if not language:
        return {"domain": "word", "position": "final"}

    table = _read_csv_if_exists(root / "languages" / language / "stress_rules.csv")
    if table.empty:
        return {"domain": "word", "position": "final"}

    first = table.iloc[0]
    domain = str(first.get("domain", "word")).strip() or "word"
    position = str(first.get("position", "final")).strip() or "final"
    return {"domain": domain, "position": position}


def detect_stress(
    syllables: pd.DataFrame,
    words: pd.DataFrame,
    ipus: pd.DataFrame,
    language: str | None = None,
    resource_root: str | Path | None = None,
) -> pd.Series:
    """
    Detect stress from aligned annotations.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    syllables : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    words : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    ipus : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    language : str | None
        Argument used by the phonology API.
    resource_root : str | Path | None
        Filesystem path used by this API.
    
    syllables example
    -----------------
    syllable_id | start | end | word_id | stress
    0 | 0.12 | 0.25 | 0 | 1
    1 | 0.46 | 0.60 | 1 | 0
    
    words example
    -------------
    word_id | start | end | speaker | word
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
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
        value = detect_stress(syllables=..., words=..., ipus=..., language=..., resource_root=...)
        print(value)
    """
    stress = pd.Series(np.zeros(len(syllables), dtype=np.float32), index=syllables.index, name="stressed")
    if syllables is None or len(syllables) == 0:
        return stress

    rules = load_stress_rules(language=language, resource_root=resource_root)
    domain = rules["domain"]
    position = rules["position"]

    if domain == "ipu":
        positions_df = compute_ipu_positions(syllables, ipus)
        group_key = "ipu_id"
        position_col = "position_in_ipu"
    else:
        positions_df = compute_word_positions(syllables, words)
        group_key = "word_id"
        position_col = "position_in_word"

    valid = positions_df.dropna(subset=[group_key]).copy()
    if valid.empty:
        return stress

    for group_id in valid[group_key].unique():
        group = valid.loc[valid[group_key] == group_id]
        if position == "initial":
            target_index = group.index[0]
        else:
            target_index = group.index[-1]
        stress.loc[target_index] = np.float32(1.0)

    return stress.astype(np.float32)
