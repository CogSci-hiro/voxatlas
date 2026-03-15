from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def _default_resource_root() -> Path:
    return Path(__file__).resolve().parents[3] / "resources" / "morphology"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=None)
def load_derivational_resources(
    language: str | None = None,
    resource_root: str | Path | None = None,
) -> dict[str, list[str]]:
    """
    Load derivational resources for VoxAtlas processing.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    language : str | None
        Argument used by the morphology API.
    resource_root : str | Path | None
        Filesystem path used by this API.
    
    Returns
    -------
    dict[str, list[str]]
        Return value produced by ``load_derivational_resources``.
    
    Examples
    --------
    >>> from voxatlas.morphology.derivation_utils import load_derivational_resources
    >>> load_derivational_resources(language="")  # no language resources
    {'prefixes': [], 'suffixes': []}
    """
    language = (language or "").strip()
    root = Path(resource_root) if resource_root is not None else _default_resource_root()

    if not language:
        return {"prefixes": [], "suffixes": []}

    resource_dir = root / "languages" / language
    prefixes = _read_csv_if_exists(resource_dir / "prefixes.csv")
    suffixes = _read_csv_if_exists(resource_dir / "suffixes.csv")

    prefix_list = []
    suffix_list = []

    if not prefixes.empty and "form" in prefixes.columns:
        prefix_list = sorted(
            {str(value).strip().lower() for value in prefixes["form"] if str(value).strip()},
            key=len,
            reverse=True,
        )

    if not suffixes.empty and "form" in suffixes.columns:
        suffix_list = sorted(
            {str(value).strip().lower() for value in suffixes["form"] if str(value).strip()},
            key=len,
            reverse=True,
        )

    return {
        "prefixes": prefix_list,
        "suffixes": suffix_list,
    }


def _normalized_form(row: pd.Series) -> str:
    for column in ("token_normalized", "token_canonical", "token", "label", "surface"):
        if column in row and pd.notna(row[column]):
            value = str(row[column]).strip().lower()
            if value:
                return value
    return ""


def _lemma_form(row: pd.Series) -> str:
    if "lemma" in row and pd.notna(row["lemma"]):
        value = str(row["lemma"]).strip().lower()
        if value:
            return value
    return _normalized_form(row)


def segment_token(token_row: pd.Series, resources: dict[str, list[str]]) -> dict[str, object]:
    """
    Provide the ``segment_token`` public API.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    token_row : pd.Series
        Argument used by the morphology API.
    resources : dict[str, list[str]]
        Dictionary of configuration values, metadata, or structured intermediate results.
    
    Returns
    -------
    dict[str, object]
        Return value produced by ``segment_token``.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.morphology.derivation_utils import segment_token
    >>> token_row = pd.Series({"id": 1, "token": "unhappy"})
    >>> resources = {"prefixes": ["un"], "suffixes": []}
    >>> segment_token(token_row, resources)["morphemes"]
    ['un', 'happy']
    """
    token_form = _normalized_form(token_row)
    lemma_form = _lemma_form(token_row)

    prefixes = []
    suffixes = []
    stem = token_form or lemma_form
    working = stem

    for prefix in resources.get("prefixes", []):
        if working.startswith(prefix) and len(working) > len(prefix):
            prefixes.append(prefix)
            working = working[len(prefix):]
            break

    for suffix in resources.get("suffixes", []):
        if working.endswith(suffix) and len(working) > len(suffix):
            suffixes.insert(0, suffix)
            working = working[:-len(suffix)]
            break

    if not working:
        working = lemma_form or token_form

    morphemes = [*prefixes, working, *suffixes]

    return {
        "normalized": token_form,
        "lemma": lemma_form,
        "prefixes": prefixes,
        "suffixes": suffixes,
        "stem": working,
        "morphemes": morphemes,
        "prefix_presence": np.float32(1.0 if prefixes else 0.0),
        "suffix_presence": np.float32(1.0 if suffixes else 0.0),
        "morpheme_count": np.float32(len(morphemes) if token_form or lemma_form else np.nan),
        "morphological_complexity": np.float32(max(0, len(morphemes) - 1) if token_form or lemma_form else np.nan),
    }


def segment_tokens(tokens: pd.DataFrame, resources: dict[str, list[str]]) -> pd.DataFrame:
    """
    Provide the ``segment_tokens`` public API.
    
    This public function belongs to the morphology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : pd.DataFrame
        Token-level annotation table used for morphological or syntactic computation.
    resources : dict[str, list[str]]
        Dictionary of configuration values, metadata, or structured intermediate results.
    
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
    >>> from voxatlas.morphology.derivation_utils import segment_tokens
    >>> tokens = pd.DataFrame([{"id": 1, "token": "unhappy"}])
    >>> resources = {"prefixes": ["un"], "suffixes": []}
    >>> out = segment_tokens(tokens, resources)
    >>> out.loc[0, "stem"]
    'happy'
    """
    if tokens is None:
        raise ValueError("Derivational segmentation requires token units")

    rows = []
    for _, token_row in tokens.iterrows():
        segmented = segment_token(token_row, resources)
        rows.append(
            {
                "id": token_row.get("id"),
                "normalized": segmented["normalized"],
                "lemma": segmented["lemma"],
                "stem": segmented["stem"],
                "prefixes": "|".join(segmented["prefixes"]),
                "suffixes": "|".join(segmented["suffixes"]),
                "morphemes": "|".join(segmented["morphemes"]),
                "prefix_presence": segmented["prefix_presence"],
                "suffix_presence": segmented["suffix_presence"],
                "morpheme_count": segmented["morpheme_count"],
                "morphological_complexity": segmented["morphological_complexity"],
            }
        )

    return pd.DataFrame(rows)
