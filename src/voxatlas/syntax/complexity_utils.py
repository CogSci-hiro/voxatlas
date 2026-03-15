from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


TOKEN_ID_COLUMNS = ("token_id", "id")
HEAD_ID_COLUMNS = ("head_id", "head")
DEPENDENCY_LABEL_COLUMNS = (
    "dependency_label",
    "dep_label",
    "deprel",
    "dep_rel",
    "dependency_relation",
)
SENTENCE_ID_COLUMNS = ("sentence_id", "sent_id", "sentence")

CLAUSE_RELATIONS = {
    "acl",
    "acl:relcl",
    "advcl",
    "ccomp",
    "csubj",
    "csubj:pass",
    "parataxis",
    "xcomp",
}


@dataclass
class DependencyNode:
    """
    Represent the dependency node concept in VoxAtlas.
    
    This public class exposes reusable state or behavior for the syntax layer of VoxAtlas. It is part of the supported API surface and is intended to be composed by pipelines, registries, and feature extractors.
    
    Parameters
    ----------
    token_id : int
        Integer argument controlling execution or indexing.
    head_id : int | None
        Argument used by the syntax API.
    dependency_label : str | None
        Argument used by the syntax API.
    children : list['DependencyNode']
        Ordered collection used by the current computation or orchestration step.
    
    Examples
    --------
        from voxatlas.syntax.complexity_utils import DependencyNode
    
        obj = DependencyNode(... )
        print(obj)
    """
    token_id: int
    head_id: int | None
    dependency_label: str | None
    children: list["DependencyNode"] = field(default_factory=list)


@dataclass
class DependencyTree:
    """
    Represent the dependency tree concept in VoxAtlas.
    
    This public class exposes reusable state or behavior for the syntax layer of VoxAtlas. It is part of the supported API surface and is intended to be composed by pipelines, registries, and feature extractors.
    
    Parameters
    ----------
    roots : list[DependencyNode]
        Ordered collection used by the current computation or orchestration step.
    nodes : dict[int, DependencyNode]
        Dictionary of configuration values, metadata, or structured intermediate results.
    
    Examples
    --------
        from voxatlas.syntax.complexity_utils import DependencyTree
    
        obj = DependencyTree(... )
        print(obj)
    """
    roots: list[DependencyNode]
    nodes: dict[int, DependencyNode]


def _resolve_column(table: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for column in candidates:
        if column in table.columns:
            return column
    raise ValueError(f"Dependency table must include a {label} column")


def _normalize_dependency_table(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame(columns=["token_id", "head_id", "dependency_label"])

    token_col = _resolve_column(table, TOKEN_ID_COLUMNS, "token id")
    head_col = _resolve_column(table, HEAD_ID_COLUMNS, "head id")
    label_col = _resolve_column(table, DEPENDENCY_LABEL_COLUMNS, "dependency label")

    normalized = table.copy()
    normalized["token_id"] = pd.to_numeric(normalized[token_col], errors="coerce")
    normalized["head_id"] = pd.to_numeric(normalized[head_col], errors="coerce")
    normalized["dependency_label"] = normalized[label_col]
    normalized = normalized.dropna(subset=["token_id"])
    normalized["token_id"] = normalized["token_id"].astype(int)

    return normalized


def _sentence_ids(table: pd.DataFrame) -> pd.Series:
    for column in SENTENCE_ID_COLUMNS:
        if column in table.columns:
            return table[column]

    if "token_id" in table.columns:
        return pd.Series(0, index=table.index, dtype="int64")

    return pd.Series(dtype="int64")


def iter_sentence_tables(table: pd.DataFrame):
    """
    Provide the ``iter_sentence_tables`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
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
        value = iter_sentence_tables(table=...)
        print(value)
    """
    normalized = _normalize_dependency_table(table)

    if normalized.empty:
        return

    sentence_ids = _sentence_ids(table).reindex(normalized.index, fill_value=0)
    normalized = normalized.assign(sentence_id=sentence_ids.values)

    for sentence_id, sentence_table in normalized.groupby("sentence_id", sort=False):
        yield sentence_id, sentence_table.reset_index(drop=True)


def build_dependency_tree(table: pd.DataFrame) -> DependencyTree:
    """
    Build dependency tree for the VoxAtlas pipeline.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Returns
    -------
    DependencyTree
        Return value produced by ``build_dependency_tree``.
    
    Examples
    --------
        value = build_dependency_tree(table=...)
        print(value)
    """
    normalized = _normalize_dependency_table(table)

    nodes = {
        row.token_id: DependencyNode(
            token_id=int(row.token_id),
            head_id=None if pd.isna(row.head_id) else int(row.head_id),
            dependency_label=(
                None if pd.isna(row.dependency_label) else str(row.dependency_label)
            ),
        )
        for row in normalized.itertuples(index=False)
    }

    roots: list[DependencyNode] = []

    for node in nodes.values():
        if node.head_id in (None, 0, node.token_id) or node.head_id not in nodes:
            roots.append(node)
            continue

        nodes[node.head_id].children.append(node)

    if not roots and nodes:
        roots = [nodes[min(nodes)]]

    return DependencyTree(roots=roots, nodes=nodes)


def mean_dependency_length(table: pd.DataFrame) -> float:
    """
    Provide the ``mean_dependency_length`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Returns
    -------
    float
        Scalar numeric value computed from the supplied signal or annotation.
    
    Examples
    --------
        value = mean_dependency_length(table=...)
        print(value)
    """
    normalized = _normalize_dependency_table(table)

    if normalized.empty:
        return 0.0

    valid_heads = normalized["head_id"].notna() & (normalized["head_id"] != 0)
    valid_heads &= normalized["head_id"] != normalized["token_id"]

    if not valid_heads.any():
        return 0.0

    distances = (
        normalized.loc[valid_heads, "token_id"] - normalized.loc[valid_heads, "head_id"]
    ).abs()
    return float(distances.mean())


def _max_tree_depth(node: DependencyNode) -> int:
    if not node.children:
        return 1
    return 1 + max(_max_tree_depth(child) for child in node.children)


def parse_tree_depth(tree: DependencyTree) -> int:
    """
    Parse tree depth into a structured representation.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tree : DependencyTree
        Argument used by the syntax API.
    
    Returns
    -------
    int
        Integer quantity derived from the supplied signal or annotations.
    
    Examples
    --------
        value = parse_tree_depth(tree=...)
        print(value)
    """
    if not tree.roots:
        return 0
    return max(_max_tree_depth(root) for root in tree.roots)


def _max_clause_depth(node: DependencyNode, current_depth: int) -> int:
    next_depth = current_depth + int((node.dependency_label or "") in CLAUSE_RELATIONS)

    if not node.children:
        return next_depth

    return max(_max_clause_depth(child, next_depth) for child in node.children)


def clause_depth(tree: DependencyTree) -> int:
    """
    Provide the ``clause_depth`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tree : DependencyTree
        Argument used by the syntax API.
    
    Returns
    -------
    int
        Integer quantity derived from the supplied signal or annotations.
    
    Examples
    --------
        value = clause_depth(tree=...)
        print(value)
    """
    if not tree.roots:
        return 0
    return max(_max_clause_depth(root, 0) for root in tree.roots)


def branching_factor(tree: DependencyTree) -> float:
    """
    Provide the ``branching_factor`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tree : DependencyTree
        Argument used by the syntax API.
    
    Returns
    -------
    float
        Scalar numeric value computed from the supplied signal or annotation.
    
    Examples
    --------
        value = branching_factor(tree=...)
        print(value)
    """
    branching_nodes = [len(node.children) for node in tree.nodes.values() if node.children]

    if not branching_nodes:
        return 0.0

    return float(sum(branching_nodes) / len(branching_nodes))


def compute_mean_dependency_length_by_sentence(table: pd.DataFrame) -> pd.Series:
    """
    Compute mean dependency length by sentence from VoxAtlas inputs.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    
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
        value = compute_mean_dependency_length_by_sentence(table=...)
        print(value)
    """
    return pd.Series(
        {
            sentence_id: mean_dependency_length(sentence_table)
            for sentence_id, sentence_table in iter_sentence_tables(table)
        },
        dtype="float32",
    )


def compute_clause_depth_by_sentence(table: pd.DataFrame) -> pd.Series:
    """
    Compute clause depth by sentence from VoxAtlas inputs.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    
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
        value = compute_clause_depth_by_sentence(table=...)
        print(value)
    """
    return pd.Series(
        {
            sentence_id: clause_depth(build_dependency_tree(sentence_table))
            for sentence_id, sentence_table in iter_sentence_tables(table)
        },
        dtype="float32",
    )


def compute_parse_tree_depth_by_sentence(table: pd.DataFrame) -> pd.Series:
    """
    Compute parse tree depth by sentence from VoxAtlas inputs.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    
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
        value = compute_parse_tree_depth_by_sentence(table=...)
        print(value)
    """
    return pd.Series(
        {
            sentence_id: parse_tree_depth(build_dependency_tree(sentence_table))
            for sentence_id, sentence_table in iter_sentence_tables(table)
        },
        dtype="float32",
    )


def compute_branching_factor_by_sentence(table: pd.DataFrame) -> pd.Series:
    """
    Compute branching factor by sentence from VoxAtlas inputs.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    
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
        value = compute_branching_factor_by_sentence(table=...)
        print(value)
    """
    return pd.Series(
        {
            sentence_id: branching_factor(build_dependency_tree(sentence_table))
            for sentence_id, sentence_table in iter_sentence_tables(table)
        },
        dtype="float32",
    )
