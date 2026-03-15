import numpy as np
import pandas as pd

from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.syntax.dependencies import SyntaxDependenciesExtractor
from voxatlas.features.syntax.local_structure.dep_label import (
    SyntaxLocalStructureDependencyLabelExtractor,
)
from voxatlas.features.syntax.local_structure.dependency_distance import (
    SyntaxLocalStructureDependencyDistanceExtractor,
)
from voxatlas.features.syntax.local_structure.head_distance import (
    SyntaxLocalStructureHeadDistanceExtractor,
)
from voxatlas.features.syntax.local_structure.pos import (
    SyntaxLocalStructurePOSExtractor,
)
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.units.units import Units


def _annotated_tokens():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "token": "I",
                "head_id": 2,
                "dep_label": "nsubj",
                "pos": "PRON",
            },
            {
                "id": 2,
                "token": "saw",
                "head_id": pd.NA,
                "dep_label": "root",
                "pos": "VERB",
            },
            {
                "id": 3,
                "token": "her",
                "head_id": 2,
                "dep_label": "obj",
                "pos": "PRON",
            },
        ]
    )


def test_syntax_dependencies_uses_existing_token_annotations():
    feature_input = FeatureInput(
        audio=None,
        units=Units(tokens=_annotated_tokens()),
        context={},
    )

    output = SyntaxDependenciesExtractor().compute(feature_input, {})

    assert list(output.values["token_id"]) == [1, 2, 3]
    assert list(output.values["head_id"].astype("object")) == [2, pd.NA, 2]
    assert list(output.values["dep_label"]) == ["nsubj", "root", "obj"]
    assert list(output.values["pos"]) == ["PRON", "VERB", "PRON"]
    assert list(output.values["id"]) == [1, 2, 3]
    assert list(output.values["deprel"]) == ["nsubj", "root", "obj"]
    assert list(output.values["dependency_label"]) == ["nsubj", "root", "obj"]
    assert list(output.values["upos"]) == ["PRON", "VERB", "PRON"]


def test_syntax_local_structure_derived_extractors_read_base_table():
    feature_input = FeatureInput(
        audio=None,
        units=Units(tokens=_annotated_tokens()),
        context={"feature_store": FeatureStore()},
    )

    dependencies = SyntaxDependenciesExtractor().compute(feature_input, {})
    feature_input.context["feature_store"].add("syntax.dependencies", dependencies)

    pos = SyntaxLocalStructurePOSExtractor().compute(feature_input, {})
    dep_label = SyntaxLocalStructureDependencyLabelExtractor().compute(feature_input, {})
    head_distance = SyntaxLocalStructureHeadDistanceExtractor().compute(feature_input, {})
    dependency_distance = SyntaxLocalStructureDependencyDistanceExtractor().compute(
        feature_input,
        {},
    )

    assert list(pos.values.index) == [1, 2, 3]
    assert list(pos.values) == ["PRON", "VERB", "PRON"]
    assert list(dep_label.values) == ["nsubj", "root", "obj"]
    assert np.allclose(head_distance.values.to_numpy(dtype=np.float32), [1.0, 0.0, -1.0])
    assert np.allclose(
        dependency_distance.values.to_numpy(dtype=np.float32),
        [1.0, 0.0, 1.0],
    )


def test_syntax_dependencies_can_invoke_parser_when_annotations_are_missing(monkeypatch):
    tokens = pd.DataFrame(
        [
            {"id": 10, "token": "Birds", "sentence_id": 1},
            {"id": 11, "token": "sing", "sentence_id": 1},
        ]
    )

    def fake_parser(token_table, params=None, context=None):
        assert list(token_table["token"]) == ["Birds", "sing"]
        return pd.DataFrame(
            [
                {
                    "token_id": 10,
                    "head_id": 11,
                    "dep_label": "nsubj",
                    "pos": "NOUN",
                },
                {
                    "token_id": 11,
                    "head_id": pd.NA,
                    "dep_label": "root",
                    "pos": "VERB",
                },
            ]
        )

    monkeypatch.setattr(
        "voxatlas.syntax.dependency_utils.parse_dependency_annotations",
        fake_parser,
    )

    feature_input = FeatureInput(audio=None, units=Units(tokens=tokens), context={})
    output = SyntaxDependenciesExtractor().compute(feature_input, {})

    assert list(output.values["token_id"]) == [10, 11]
    assert list(output.values["dep_label"]) == ["nsubj", "root"]
