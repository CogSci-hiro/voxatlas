from pathlib import Path

import numpy as np
import pandas as pd

from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.phonology.articulatory.features import (
    ArticulatoryFeaturesExtractor,
)
from voxatlas.features.phonology.articulatory.voiced import (
    ArticulatoryVoicedExtractor,
)
from voxatlas.features.phonology.articulatory.vowel import (
    ArticulatoryVowelExtractor,
)
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.units.units import Units


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resource_tree(root: Path) -> Path:
    _write_csv(
        root / "universal" / "xsampa_to_ipa.csv",
        "xsampa,ipa\nA,a\nb,b\np,p\n",
    )
    _write_csv(
        root / "universal" / "articulatory_features.csv",
        (
            "ipa,category,manner,voice\n"
            "a,vowel,vowel,voiced\n"
            "b,consonant,plosive,voiced\n"
            "p,consonant,plosive,voiceless\n"
        ),
    )
    _write_csv(
        root / "languages" / "fra" / "phoneme_inventory.csv",
        "phoneme,ipa\nA,a\nb,b\np,p\n",
    )
    _write_csv(
        root / "languages" / "fra" / "articulatory_overrides.csv",
        "phoneme,ipa,category,manner,voice\nb,b,consonant,plosive,voiceless\n",
    )
    return root


def test_articulatory_features_normalize_lookup_and_suspects(tmp_path):
    resource_root = _resource_tree(tmp_path / "phonology")
    phonemes = pd.DataFrame(
        [
            {"id": 1, "start": 0.0, "end": 0.1, "label": "A"},
            {"id": 2, "start": 0.1, "end": 0.2, "label": "b"},
            {"id": 3, "start": 0.2, "end": 0.3, "label": "?"},
        ]
    )
    feature_input = FeatureInput(
        audio=None,
        units=Units(phonemes=phonemes),
        context={},
    )

    output = ArticulatoryFeaturesExtractor().compute(
        feature_input,
        {"language": "fra", "resource_root": str(resource_root)},
    )

    assert list(output.values["ipa"]) == ["a", "b", "?"]
    assert output.values.loc[0, "vowel"] == np.float32(1.0)
    assert output.values.loc[1, "voiceless"] == np.float32(1.0)
    assert np.isnan(output.values.loc[2, "vowel"])
    assert feature_input.context["suspect_phonemes"] == [
        {
            "phoneme": "?",
            "ipa": "?",
            "language": "fra",
            "reason": "unknown articulatory phoneme",
        }
    ]


def test_articulatory_derived_extractors_use_base_table(tmp_path):
    resource_root = _resource_tree(tmp_path / "phonology")
    phonemes = pd.DataFrame(
        [
            {"id": 10, "start": 0.0, "end": 0.1, "label": "A"},
            {"id": 11, "start": 0.1, "end": 0.2, "label": "p"},
        ]
    )
    feature_input = FeatureInput(
        audio=None,
        units=Units(phonemes=phonemes),
        context={"feature_store": FeatureStore()},
    )

    base_output = ArticulatoryFeaturesExtractor().compute(
        feature_input,
        {"language": "fra", "resource_root": str(resource_root)},
    )
    feature_input.context["feature_store"].add(
        "phonology.articulatory.features",
        base_output,
    )

    vowel_output = ArticulatoryVowelExtractor().compute(feature_input, {})
    voiced_output = ArticulatoryVoicedExtractor().compute(feature_input, {})

    assert list(vowel_output.values.index) == [10, 11]
    assert list(vowel_output.values.values) == [np.float32(1.0), np.float32(0.0)]
    assert list(voiced_output.values.values) == [np.float32(1.0), np.float32(0.0)]
