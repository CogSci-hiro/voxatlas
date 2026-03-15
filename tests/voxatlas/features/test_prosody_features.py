from pathlib import Path

import pandas as pd

from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.phonology.prosody.position_ipu import (
    ProsodicPositionInIPUExtractor,
)
from voxatlas.features.phonology.prosody.position_word import (
    ProsodicPositionInWordExtractor,
)
from voxatlas.features.phonology.prosody.stress import ProsodicStressExtractor
from voxatlas.features.phonology.prosody.unstressed import (
    ProsodicUnstressedExtractor,
)
from voxatlas.phonology.prosody_utils import (
    compute_ipu_positions,
    compute_word_positions,
    detect_stress,
)
from voxatlas.units.units import Units


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resource_tree(root: Path) -> Path:
    _write_csv(
        root / "languages" / "fra" / "stress_rules.csv",
        "domain,position\nword,final\n",
    )
    _write_csv(
        root / "languages" / "eng" / "stress_rules.csv",
        "domain,position\nword,initial\n",
    )
    return root


def _units():
    syllables = pd.DataFrame(
        [
            {"id": 1, "start": 0.00, "end": 0.10, "label": "sy1"},
            {"id": 2, "start": 0.10, "end": 0.20, "label": "sy2"},
            {"id": 3, "start": 0.20, "end": 0.30, "label": "sy3"},
            {"id": 4, "start": 0.30, "end": 0.40, "label": "sy4"},
        ]
    )
    words = pd.DataFrame(
        [
            {"id": 10, "start": 0.00, "end": 0.20, "label": "w1"},
            {"id": 11, "start": 0.20, "end": 0.40, "label": "w2"},
        ]
    )
    ipus = pd.DataFrame(
        [
            {"id": 20, "start": 0.00, "end": 0.40, "label": "ipu1"},
        ]
    )
    return Units(syllables=syllables, words=words, ipus=ipus, speaker="A")


def test_prosody_utils_compute_positions_and_stress(tmp_path):
    resource_root = _resource_tree(tmp_path / "phonology")
    units = _units()

    word_positions = compute_word_positions(units.get("syllable"), units.get("word"))
    ipu_positions = compute_ipu_positions(units.get("syllable"), units.get("ipu"))
    stressed_fra = detect_stress(
        units.get("syllable"),
        units.get("word"),
        units.get("ipu"),
        language="fra",
        resource_root=str(resource_root),
    )
    stressed_eng = detect_stress(
        units.get("syllable"),
        units.get("word"),
        units.get("ipu"),
        language="eng",
        resource_root=str(resource_root),
    )

    assert list(word_positions["position_in_word"].astype("Int64")) == [1, 2, 1, 2]
    assert list(ipu_positions["position_in_ipu"].astype("Int64")) == [1, 2, 3, 4]
    assert list(stressed_fra.astype("float32")) == [0.0, 1.0, 0.0, 1.0]
    assert list(stressed_eng.astype("float32")) == [1.0, 0.0, 1.0, 0.0]


def test_prosody_extractors_return_syllable_indexed_series(tmp_path):
    resource_root = _resource_tree(tmp_path / "phonology")
    units = _units()
    feature_input = FeatureInput(audio=None, units=units, context={})

    stressed = ProsodicStressExtractor().compute(
        feature_input,
        {"language": "fra", "resource_root": str(resource_root)},
    )
    unstressed = ProsodicUnstressedExtractor().compute(
        feature_input,
        {"language": "fra", "resource_root": str(resource_root)},
    )
    word_pos = ProsodicPositionInWordExtractor().compute(feature_input, {})
    ipu_pos = ProsodicPositionInIPUExtractor().compute(feature_input, {})

    assert list(stressed.values.index) == [1, 2, 3, 4]
    assert list(stressed.values.astype("float32")) == [0.0, 1.0, 0.0, 1.0]
    assert list(unstressed.values.astype("float32")) == [1.0, 0.0, 1.0, 0.0]
    assert list(word_pos.values.astype("Int64")) == [1, 2, 1, 2]
    assert list(ipu_pos.values.astype("Int64")) == [1, 2, 3, 4]
