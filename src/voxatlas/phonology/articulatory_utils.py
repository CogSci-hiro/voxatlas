from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "ipa",
    "category",
    "manner",
    "voice",
    "vowel",
    "consonant",
    "nasal",
    "plosive",
    "fricative",
    "approximant",
    "voiced",
    "voiceless",
]


def _default_resource_root() -> Path:
    return Path(__file__).resolve().parents[3] / "resources" / "phonology"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_mapping_table(df: pd.DataFrame, source_col: str, target_col: str) -> dict[str, str]:
    if df.empty or source_col not in df.columns or target_col not in df.columns:
        return {}

    mapping = {}
    for _, row in df.iterrows():
        source = str(row[source_col]).strip()
        target = str(row[target_col]).strip()
        if source:
            mapping[source] = target
    return mapping


def _prepare_feature_table(df: pd.DataFrame, phoneme_col: str) -> dict[str, dict[str, object]]:
    if df.empty or phoneme_col not in df.columns:
        return {}

    table = {}
    for _, row in df.iterrows():
        phoneme = str(row[phoneme_col]).strip()
        if not phoneme:
            continue

        entry = {
            "ipa": str(row.get("ipa", phoneme)).strip() or phoneme,
            "category": row.get("category", np.nan),
            "manner": row.get("manner", np.nan),
            "voice": row.get("voice", np.nan),
        }
        table[phoneme] = derive_articulatory_flags(entry)
    return table


@lru_cache(maxsize=None)
def load_phonology_resources(
    language: str | None = None,
    resource_root: str | Path | None = None,
) -> dict[str, object]:
    """
    Load phonology resources for VoxAtlas processing.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    language : str | None
        Argument used by the phonology API.
    resource_root : str | Path | None
        Filesystem path used by this API.
    
    Returns
    -------
    dict[str, object]
        Return value produced by ``load_phonology_resources``.
    
    Examples
    --------
        value = load_phonology_resources(language=..., resource_root=...)
        print(value)
    """
    language = (language or "").strip()
    root = Path(resource_root) if resource_root is not None else _default_resource_root()

    universal_dir = root / "universal"
    language_dir = root / "languages" / language if language else None

    xsampa_map = _prepare_mapping_table(
        _read_csv_if_exists(universal_dir / "xsampa_to_ipa.csv"),
        "xsampa",
        "ipa",
    )
    universal_table = _prepare_feature_table(
        _read_csv_if_exists(universal_dir / "articulatory_features.csv"),
        "ipa",
    )

    inventory_table = {}
    override_table = {}

    if language_dir is not None:
        inventory_table = _prepare_mapping_table(
            _read_csv_if_exists(language_dir / "phoneme_inventory.csv"),
            "phoneme",
            "ipa",
        )
        override_table = _prepare_feature_table(
            _read_csv_if_exists(language_dir / "articulatory_overrides.csv"),
            "phoneme",
        )

    return {
        "resource_root": str(root),
        "language": language,
        "xsampa_to_ipa": xsampa_map,
        "inventory": inventory_table,
        "universal": universal_table,
        "overrides": override_table,
    }


def normalize_phoneme_to_ipa(phoneme: object, resources: dict[str, object]) -> str | None:
    """
    Provide the ``normalize_phoneme_to_ipa`` public API.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    phoneme : object
        Argument used by the phonology API.
    resources : dict[str, object]
        Dictionary of configuration values, metadata, or structured intermediate results.
    
    Returns
    -------
    str | None
        Return value produced by ``normalize_phoneme_to_ipa``.
    
    Examples
    --------
        value = normalize_phoneme_to_ipa(phoneme=..., resources=...)
        print(value)
    """
    if phoneme is None or (isinstance(phoneme, float) and np.isnan(phoneme)):
        return None

    normalized = str(phoneme).strip()
    if not normalized:
        return None

    inventory = resources.get("inventory", {})
    xsampa_map = resources.get("xsampa_to_ipa", {})

    if normalized in inventory:
        return inventory[normalized]

    if normalized in xsampa_map:
        return xsampa_map[normalized]

    return normalized


def derive_articulatory_flags(entry: dict[str, object]) -> dict[str, object]:
    """
    Provide the ``derive_articulatory_flags`` public API.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    entry : dict[str, object]
        Dictionary of configuration values, metadata, or structured intermediate results.
    
    Returns
    -------
    dict[str, object]
        Return value produced by ``derive_articulatory_flags``.
    
    Examples
    --------
        value = derive_articulatory_flags(entry=...)
        print(value)
    """
    category = str(entry.get("category", "")).strip().lower()
    manner = str(entry.get("manner", "")).strip().lower()
    voice = str(entry.get("voice", "")).strip().lower()

    flags = {
        "ipa": entry.get("ipa", np.nan),
        "category": entry.get("category", np.nan),
        "manner": entry.get("manner", np.nan),
        "voice": entry.get("voice", np.nan),
        "vowel": np.float32(category == "vowel"),
        "consonant": np.float32(category == "consonant"),
        "nasal": np.float32(manner == "nasal"),
        "plosive": np.float32(manner == "plosive"),
        "fricative": np.float32(manner == "fricative"),
        "approximant": np.float32(manner == "approximant"),
        "voiced": np.float32(voice == "voiced"),
        "voiceless": np.float32(voice == "voiceless"),
    }
    return flags


def lookup_articulatory_features(
    phoneme: object,
    resources: dict[str, object],
) -> tuple[str | None, dict[str, object] | None]:
    """
    Provide the ``lookup_articulatory_features`` public API.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    phoneme : object
        Argument used by the phonology API.
    resources : dict[str, object]
        Dictionary of configuration values, metadata, or structured intermediate results.
    
    Returns
    -------
    tuple[str | None, dict[str, object] | None]
        Return value produced by ``lookup_articulatory_features``.
    
    Examples
    --------
        value = lookup_articulatory_features(phoneme=..., resources=...)
        print(value)
    """
    ipa = normalize_phoneme_to_ipa(phoneme, resources)
    if ipa is None:
        return None, None

    overrides = resources.get("overrides", {})
    universal = resources.get("universal", {})

    if ipa in overrides:
        return ipa, overrides[ipa]

    if ipa in universal:
        return ipa, universal[ipa]

    return ipa, None


def log_unknown_phoneme(
    phoneme: object,
    ipa: str | None,
    language: str | None,
    context: dict,
) -> None:
    """
    Provide the ``log_unknown_phoneme`` public API.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    phoneme : object
        Argument used by the phonology API.
    ipa : str | None
        Argument used by the phonology API.
    language : str | None
        Argument used by the phonology API.
    context : dict
        Additional pipeline metadata, configuration, and dependency outputs available to the current computation stage.
    
    Returns
    -------
    None
        Return value produced by ``log_unknown_phoneme``.
    
    Examples
    --------
        value = log_unknown_phoneme(phoneme=..., ipa=..., language=..., context=...)
        print(value)
    """
    suspect_entry = {
        "phoneme": phoneme,
        "ipa": ipa,
        "language": language,
        "reason": "unknown articulatory phoneme",
    }
    suspect_list = context.setdefault("suspect_phonemes", [])
    suspect_list.append(suspect_entry)
    LOGGER.warning(
        "Unknown phoneme for articulatory lookup: phoneme=%s ipa=%s language=%s",
        phoneme,
        ipa,
        language,
    )
