import numpy as np
import pandas as pd

from voxatlas.audio.audio import Audio
from voxatlas.features.syntax import dependencies as _syntax_dependencies_feature
from voxatlas.features.syntax.complexity import branching_factor as _branching_factor_feature
from voxatlas.features.syntax.complexity import clause_depth as _clause_depth_feature
from voxatlas.features.syntax.complexity import mean_dependency_length as _mdl_feature
from voxatlas.features.syntax.complexity import parse_tree_depth as _tree_depth_feature
from voxatlas.pipeline.pipeline import VoxAtlasPipeline
from voxatlas.syntax.complexity_utils import (
    build_dependency_tree,
    branching_factor,
    clause_depth,
    compute_branching_factor_by_sentence,
    compute_clause_depth_by_sentence,
    compute_mean_dependency_length_by_sentence,
    compute_parse_tree_depth_by_sentence,
    mean_dependency_length,
    parse_tree_depth,
)
from voxatlas.units.units import Units


def _dependency_table():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "sentence_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "head_id": [2, 0, 2, 3, 6, 0, 6, 7],
            "deprel": ["nsubj", "root", "advcl", "obj", "nsubj", "root", "ccomp", "obj"],
            "upos": ["PRON", "VERB", "VERB", "NOUN", "PRON", "VERB", "VERB", "NOUN"],
        }
    )


def test_complexity_utils_compute_sentence_metrics():
    table = _dependency_table()

    mdl = compute_mean_dependency_length_by_sentence(table)
    clause = compute_clause_depth_by_sentence(table)
    tree_depth = compute_parse_tree_depth_by_sentence(table)
    branch = compute_branching_factor_by_sentence(table)

    assert mdl.to_dict() == {1: 1.0, 2: 1.0}
    assert clause.to_dict() == {1: 1.0, 2: 1.0}
    assert tree_depth.to_dict() == {1: 3.0, 2: 3.0}
    assert branch.to_dict() == {1: 1.5, 2: 1.5}


def test_complexity_tree_metrics_support_alias_columns():
    table = pd.DataFrame(
        {
            "token_id": [1, 2, 3, 4],
            "head_id": [2, 0, 2, 3],
            "dependency_label": ["nsubj", "root", "advcl", "obj"],
        }
    )

    tree = build_dependency_tree(table)

    assert mean_dependency_length(table) == 1.0
    assert clause_depth(tree) == 1
    assert parse_tree_depth(tree) == 3
    assert branching_factor(tree) == 1.5


def test_complexity_features_run_through_pipeline():
    table = _dependency_table()
    units = Units(tokens=table)
    audio = Audio(waveform=np.array([0.0]), sample_rate=16000)
    config = {
        "features": [
            "syntax.complexity.mean_dependency_length",
            "syntax.complexity.clause_depth",
            "syntax.complexity.parse_tree_depth",
            "syntax.complexity.branching_factor",
        ],
        "pipeline": {"cache": False},
    }

    results = VoxAtlasPipeline(audio, units, config).run()

    assert results.get("syntax.complexity.mean_dependency_length").unit == "sentence"
    assert (
        results.get("syntax.complexity.mean_dependency_length").values.to_dict()
        == {1: 1.0, 2: 1.0}
    )
    assert results.get("syntax.complexity.clause_depth").values.to_dict() == {
        1: 1.0,
        2: 1.0,
    }
    assert results.get("syntax.complexity.parse_tree_depth").values.to_dict() == {
        1: 3.0,
        2: 3.0,
    }
    assert results.get("syntax.complexity.branching_factor").values.to_dict() == {
        1: 1.5,
        2: 1.5,
    }
