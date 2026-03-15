from pathlib import Path

import numpy as np
import pandas as pd

from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.morphology.derivation.complexity import (
    MorphologicalComplexityExtractor,
)
from voxatlas.features.morphology.derivation.morpheme_count import (
    MorphemeCountExtractor,
)
from voxatlas.features.morphology.derivation.prefix_presence import (
    PrefixPresenceExtractor,
)
from voxatlas.features.morphology.derivation.segmentation import (
    DerivationalSegmentationExtractor,
)
from voxatlas.features.morphology.derivation.suffix_presence import (
    SuffixPresenceExtractor,
)
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.units.units import Units


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resource_tree(root: Path) -> Path:
    _write_csv(root / "languages" / "eng" / "prefixes.csv", "form\nun\nre\n")
    _write_csv(root / "languages" / "eng" / "suffixes.csv", "form\ning\nness\nable\n")
    return root


def test_derivational_segmentation_uses_language_resources(tmp_path):
    resource_root = _resource_tree(tmp_path / "morphology")
    tokens = pd.DataFrame(
        [
            {"id": 1, "token": "redoing", "token_normalized": "redoing", "lemma": "redo"},
            {"id": 2, "token": "kindness", "token_normalized": "kindness", "lemma": "kind"},
            {"id": 3, "token": "plain", "token_normalized": "plain", "lemma": "plain"},
        ]
    )
    feature_input = FeatureInput(
        audio=None,
        units=Units(tokens=tokens),
        context={},
    )

    output = DerivationalSegmentationExtractor().compute(
        feature_input,
        {"language": "eng", "resource_root": str(resource_root)},
    )

    assert list(output.values["prefix_presence"].astype("float32")) == [1.0, 0.0, 0.0]
    assert list(output.values["suffix_presence"].astype("float32")) == [1.0, 1.0, 0.0]
    assert list(output.values["morpheme_count"].astype("float32")) == [3.0, 2.0, 1.0]
    assert list(output.values["morphological_complexity"].astype("float32")) == [2.0, 1.0, 0.0]
    assert output.values.loc[0, "morphemes"] == "re|do|ing"


def test_derivational_derived_extractors_read_segmentation(tmp_path):
    resource_root = _resource_tree(tmp_path / "morphology")
    tokens = pd.DataFrame(
        [
            {"id": 10, "token": "undoable", "token_normalized": "undoable", "lemma": "do"},
            {"id": 11, "token": "plain", "token_normalized": "plain", "lemma": "plain"},
        ]
    )
    feature_input = FeatureInput(
        audio=None,
        units=Units(tokens=tokens),
        context={"feature_store": FeatureStore()},
    )

    segmentation = DerivationalSegmentationExtractor().compute(
        feature_input,
        {"language": "eng", "resource_root": str(resource_root)},
    )
    feature_input.context["feature_store"].add(
        "morphology.derivation.segmentation",
        segmentation,
    )

    prefix = PrefixPresenceExtractor().compute(feature_input, {})
    suffix = SuffixPresenceExtractor().compute(feature_input, {})
    morpheme_count = MorphemeCountExtractor().compute(feature_input, {})
    complexity = MorphologicalComplexityExtractor().compute(feature_input, {})

    assert list(prefix.values.index) == [10, 11]
    assert np.allclose(prefix.values.to_numpy(dtype=np.float32), [1.0, 0.0])
    assert np.allclose(suffix.values.to_numpy(dtype=np.float32), [1.0, 0.0])
    assert np.allclose(morpheme_count.values.to_numpy(dtype=np.float32), [3.0, 1.0])
    assert np.allclose(complexity.values.to_numpy(dtype=np.float32), [2.0, 0.0])
