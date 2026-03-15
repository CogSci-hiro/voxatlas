from pathlib import Path

import numpy as np
import pandas as pd

from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.lexical.properties.animacy import AnimacyExtractor
from voxatlas.features.lexical.properties.concreteness import ConcretenessExtractor
from voxatlas.features.lexical.properties.function_word import (
    FunctionWordExtractor,
)
from voxatlas.features.lexical.properties.lookup import (
    LexicalPropertyLookupExtractor,
)
from voxatlas.features.lexical.properties.pos import POSExtractor
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.units.units import Units


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resource_tree(root: Path) -> Path:
    _write_csv(
        root / "languages" / "eng" / "animacy.csv",
        "lemma,animacy\ndog,animate\nrock,inanimate\n",
    )
    _write_csv(
        root / "languages" / "eng" / "concreteness.csv",
        "lemma,concreteness\ndog,concrete\nidea,abstract\n",
    )
    return root


def test_lexical_property_lookup_combines_pos_and_lexicons(tmp_path):
    resource_root = _resource_tree(tmp_path / "lexical")
    tokens = pd.DataFrame(
        [
            {"id": 1, "token": "the", "lemma": "the", "pos": "DET"},
            {"id": 2, "token": "dog", "lemma": "dog", "upos": "NOUN"},
            {"id": 3, "token": "idea", "lemma": "idea", "pos": "NOUN"},
        ]
    )
    feature_input = FeatureInput(audio=None, units=Units(tokens=tokens), context={})

    output = LexicalPropertyLookupExtractor().compute(
        feature_input,
        {"language": "eng", "resource_root": str(resource_root)},
    )

    assert list(output.values["pos"]) == ["DET", "NOUN", "NOUN"]
    assert list(output.values["function_word"].astype("float32")) == [1.0, 0.0, 0.0]
    assert output.values.loc[1, "animacy"] == "animate"
    assert output.values.loc[2, "concreteness"] == "abstract"
    assert np.isnan(output.values.loc[0, "animacy"])


def test_lexical_property_derived_extractors_read_lookup_table(tmp_path):
    resource_root = _resource_tree(tmp_path / "lexical")
    tokens = pd.DataFrame(
        [
            {"id": 10, "token": "dog", "lemma": "dog", "pos": "NOUN"},
            {"id": 11, "token": "the", "lemma": "the", "pos": "DET"},
        ]
    )
    feature_input = FeatureInput(
        audio=None,
        units=Units(tokens=tokens),
        context={"feature_store": FeatureStore()},
    )

    base = LexicalPropertyLookupExtractor().compute(
        feature_input,
        {"language": "eng", "resource_root": str(resource_root)},
    )
    feature_input.context["feature_store"].add("lexical.properties.lookup", base)

    pos = POSExtractor().compute(feature_input, {})
    function_word = FunctionWordExtractor().compute(feature_input, {})
    animacy = AnimacyExtractor().compute(feature_input, {})
    concreteness = ConcretenessExtractor().compute(feature_input, {})

    assert list(pos.values.index) == [10, 11]
    assert list(pos.values.values) == ["NOUN", "DET"]
    assert np.allclose(function_word.values.to_numpy(dtype=np.float32), [0.0, 1.0])
    assert animacy.values.iloc[0] == "animate"
    assert pd.isna(animacy.values.iloc[1])
    assert concreteness.values.iloc[0] == "concrete"
    assert pd.isna(concreteness.values.iloc[1])
