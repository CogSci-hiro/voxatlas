from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd


HEAD_COLUMNS = ("head_id", "head")
DEP_LABEL_COLUMNS = ("dep_label", "deprel", "dep_rel", "dependency_relation")
POS_COLUMNS = ("pos", "upos", "POS", "xpos")
SENTENCE_ID_COLUMNS = ("sentence_id", "sentence_index")
SENTENCE_START_COLUMNS = ("sentence_start", "is_sent_start")


def require_column(table: pd.DataFrame, column: str) -> str:
    """
    Provide the ``require_column`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    column : str
        String argument consumed by this API.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Returns
    -------
    str
        String value derived from the supplied metadata or lookup resources.
    
    Examples
    --------
        value = require_column(table=..., column=...)
        print(value)
    """
    if column not in table.columns:
        raise ValueError(f"Dependency features require a '{column}' column")
    return column


def find_first_column(table: pd.DataFrame, columns: Iterable[str]) -> str | None:
    """
    Provide the ``find_first_column`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    table : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level.
    columns : Iterable[str]
        Argument used by the syntax API.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Returns
    -------
    str | None
        Return value produced by ``find_first_column``.
    
    Examples
    --------
        value = find_first_column(table=..., columns=...)
        print(value)
    """
    for column in columns:
        if column in table.columns:
            return column
    return None


def sentence_slices(tokens: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Provide the ``sentence_slices`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : pd.DataFrame
        Token-level annotation table used for morphological or syntactic computation.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    Returns
    -------
    list of pandas.DataFrame
        Sentence- or unit-level slices derived from the input table.
    
    Examples
    --------
        value = sentence_slices(tokens=...)
        print(value)
    """
    sentence_id_col = find_first_column(tokens, SENTENCE_ID_COLUMNS)
    if sentence_id_col is not None:
        return [group.copy() for _, group in tokens.groupby(sentence_id_col, sort=False)]

    sentence_start_col = find_first_column(tokens, SENTENCE_START_COLUMNS)
    if sentence_start_col is None:
        return [tokens.copy()]

    boundaries = tokens[sentence_start_col].fillna(False).astype(bool).tolist()
    if boundaries:
        boundaries[0] = True

    sentences: list[pd.DataFrame] = []
    start = 0
    for idx in range(1, len(tokens)):
        if boundaries[idx]:
            sentences.append(tokens.iloc[start:idx].copy())
            start = idx

    if len(tokens) > 0:
        sentences.append(tokens.iloc[start:].copy())

    return sentences or [tokens.copy()]


def sentence_identifier(sentence: pd.DataFrame, fallback: int) -> int:
    """
    Provide the ``sentence_identifier`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    sentence : pd.DataFrame
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    fallback : int
        Integer argument controlling execution or indexing.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Returns
    -------
    int
        Integer quantity derived from the supplied signal or annotations.
    
    Examples
    --------
        value = sentence_identifier(sentence=..., fallback=...)
        print(value)
    """
    sentence_id_col = find_first_column(sentence, SENTENCE_ID_COLUMNS)
    if sentence_id_col is None or sentence.empty:
        return fallback

    value = sentence.iloc[0][sentence_id_col]
    if pd.isna(value):
        return fallback

    return int(value)


def build_dependency_table_from_annotations(tokens: pd.DataFrame) -> pd.DataFrame:
    """
    Build dependency table from annotations for the VoxAtlas pipeline.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : pd.DataFrame
        Token-level annotation table used for morphological or syntactic computation.
    
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
        value = build_dependency_table_from_annotations(tokens=...)
        print(value)
    """
    id_col = require_column(tokens, "id")
    head_col = find_first_column(tokens, HEAD_COLUMNS)
    dep_label_col = find_first_column(tokens, DEP_LABEL_COLUMNS)
    pos_col = find_first_column(tokens, POS_COLUMNS)

    if head_col is None or dep_label_col is None or pos_col is None:
        missing = []
        if head_col is None:
            missing.append(f"head column in {HEAD_COLUMNS}")
        if dep_label_col is None:
            missing.append(f"dependency label column in {DEP_LABEL_COLUMNS}")
        if pos_col is None:
            missing.append(f"POS column in {POS_COLUMNS}")
        raise ValueError(
            "Dependency features require existing dependency annotations or a parser. "
            f"Missing {', '.join(missing)}."
        )

    table = pd.DataFrame(
        {
            "token_id": tokens[id_col].values,
            "head_id": tokens[head_col].values,
            "dep_label": tokens[dep_label_col].values,
            "pos": tokens[pos_col].values,
        }
    )
    sentence_id_col = find_first_column(tokens, SENTENCE_ID_COLUMNS)
    if sentence_id_col is not None:
        table["sentence_id"] = tokens[sentence_id_col].values
    return finalize_dependency_table(table)


def has_dependency_annotations(tokens: pd.DataFrame) -> bool:
    """
    Provide the ``has_dependency_annotations`` public API.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : pd.DataFrame
        Token-level annotation table used for morphological or syntactic computation.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    Returns
    -------
    bool
        Boolean flag indicating whether the requested condition holds.
    
    Examples
    --------
        value = has_dependency_annotations(tokens=...)
        print(value)
    """
    return (
        "id" in tokens.columns
        and find_first_column(tokens, HEAD_COLUMNS) is not None
        and find_first_column(tokens, DEP_LABEL_COLUMNS) is not None
        and find_first_column(tokens, POS_COLUMNS) is not None
    )


def _run_custom_parser(
    tokens: pd.DataFrame,
    parser: Callable[[list[pd.DataFrame]], pd.DataFrame],
) -> pd.DataFrame:
    parsed = parser(sentence_slices(tokens))
    required = {"token_id", "head_id", "dep_label", "pos"}
    if not isinstance(parsed, pd.DataFrame) or not required.issubset(parsed.columns):
        raise ValueError(
            "Custom dependency parser must return a DataFrame with "
            "'token_id', 'head_id', 'dep_label', and 'pos' columns"
        )
    parsed = parsed.copy()
    keep_columns = ["token_id", "head_id", "dep_label", "pos"]
    if "sentence_id" in parsed.columns:
        keep_columns.append("sentence_id")
    parsed = parsed.loc[:, keep_columns]
    return finalize_dependency_table(parsed)


def finalize_dependency_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Provide the ``finalize_dependency_table`` public API.
    
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
    pandas.DataFrame
        Tabular result aligned to a VoxAtlas unit level or registry resource.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Examples
    --------
        value = finalize_dependency_table(table=...)
        print(value)
    """
    normalized = table.copy()
    normalized["token_id"] = pd.to_numeric(normalized["token_id"], errors="coerce").astype(
        "Int64"
    )
    normalized["head_id"] = pd.to_numeric(normalized["head_id"], errors="coerce").astype(
        "Int64"
    )
    normalized["dep_label"] = normalized["dep_label"].astype("object")
    normalized["pos"] = normalized["pos"].astype("object")
    normalized["id"] = normalized["token_id"]
    normalized["deprel"] = normalized["dep_label"]
    normalized["dependency_label"] = normalized["dep_label"]
    normalized["upos"] = normalized["pos"]
    return normalized


def _ensure_sentence_alignment(sentence: pd.DataFrame, parsed_rows: list[dict]) -> None:
    if len(sentence) != len(parsed_rows):
        raise ValueError(
            "Dependency parser token count does not match the provided token units"
        )


def _load_spacy_pipeline(params: dict, context: dict):
    pipeline = params.get("spacy_nlp") or context.get("spacy_nlp")
    if pipeline is not None:
        return pipeline

    try:
        import spacy
    except ImportError as exc:
        raise ValueError(
            "spaCy is not installed. Install the 'syntax' extra or provide "
            "'spacy_nlp' in params/context."
        ) from exc

    model_name = params.get("spacy_model") or context.get("spacy_model") or "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise ValueError(
            f"Unable to load spaCy model '{model_name}'. Provide 'spacy_nlp' or install "
            "the model before running syntax.dependencies."
        ) from exc


def _parse_with_spacy(tokens: pd.DataFrame, params: dict, context: dict) -> pd.DataFrame:
    nlp = _load_spacy_pipeline(params, context)

    try:
        from spacy.tokens import Doc
    except ImportError as exc:
        raise ValueError("spaCy is not available for dependency parsing") from exc

    rows: list[dict] = []
    for fallback_sentence_id, sentence in enumerate(sentence_slices(tokens)):
        words = sentence["token"].astype(str).tolist()
        doc = Doc(nlp.vocab, words=words)
        for token in doc:
            token.is_sent_start = token.i == 0

        for _, component in nlp.pipeline:
            doc = component(doc)

        sentence_rows: list[dict] = []
        sentence_ids = sentence["id"].tolist()
        current_sentence_id = sentence_identifier(sentence, fallback_sentence_id)
        for parsed_token, token_id in zip(doc, sentence_ids, strict=False):
            head_id = pd.NA
            if parsed_token.head.i != parsed_token.i:
                head_id = sentence_ids[parsed_token.head.i]

            sentence_rows.append(
                {
                    "token_id": token_id,
                    "head_id": head_id,
                    "dep_label": parsed_token.dep_ or "root",
                    "pos": parsed_token.pos_ or "",
                    "sentence_id": current_sentence_id,
                }
            )

        _ensure_sentence_alignment(sentence, sentence_rows)
        rows.extend(sentence_rows)

    return finalize_dependency_table(pd.DataFrame(rows))


def _load_stanza_pipeline(params: dict, context: dict):
    pipeline = params.get("stanza_pipeline") or context.get("stanza_pipeline")
    if pipeline is not None:
        return pipeline

    try:
        import stanza
    except ImportError as exc:
        raise ValueError(
            "Stanza is not installed. Provide 'stanza_pipeline' in params/context "
            "or install stanza."
        ) from exc

    language = params.get("language") or context.get("language") or "en"
    try:
        return stanza.Pipeline(
            lang=language,
            processors="tokenize,pos,depparse",
            tokenize_pretokenized=True,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local stanza setup
        raise ValueError(
            "Unable to initialize the Stanza pipeline. Provide 'stanza_pipeline' "
            "in params/context or download the required Stanza models."
        ) from exc


def _parse_with_stanza(tokens: pd.DataFrame, params: dict, context: dict) -> pd.DataFrame:
    pipeline = _load_stanza_pipeline(params, context)

    rows: list[dict] = []
    for fallback_sentence_id, sentence in enumerate(sentence_slices(tokens)):
        words = sentence["token"].astype(str).tolist()
        document = pipeline([words])
        parsed_words = []
        for parsed_sentence in document.sentences:
            parsed_words.extend(parsed_sentence.words)

        _ensure_sentence_alignment(sentence, parsed_words)
        sentence_ids = sentence["id"].tolist()
        current_sentence_id = sentence_identifier(sentence, fallback_sentence_id)

        for parsed_word, token_id in zip(parsed_words, sentence_ids, strict=False):
            head_id = pd.NA
            if parsed_word.head not in (None, 0):
                head_id = sentence_ids[parsed_word.head - 1]

            rows.append(
                {
                    "token_id": token_id,
                    "head_id": head_id,
                    "dep_label": parsed_word.deprel or "root",
                    "pos": parsed_word.upos or "",
                    "sentence_id": current_sentence_id,
                }
            )

    return finalize_dependency_table(pd.DataFrame(rows))


def parse_dependency_annotations(tokens: pd.DataFrame, params: dict | None = None, context: dict | None = None) -> pd.DataFrame:
    """
    Parse dependency annotations into a structured representation.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : pd.DataFrame
        Token-level annotation table used for morphological or syntactic computation.
    params : dict | None
        Resolved feature configuration for this invocation. Keys are feature-specific and merged from defaults and pipeline settings.
    context : dict | None
        Additional pipeline metadata, configuration, and dependency outputs available to the current computation stage.
    
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
        value = parse_dependency_annotations(tokens=..., params=..., context=...)
        print(value)
    """
    params = params or {}
    context = context or {}

    require_column(tokens, "id")
    require_column(tokens, "token")

    custom_parser = params.get("parser") or context.get("parser")
    if custom_parser is not None:
        return _run_custom_parser(tokens, custom_parser)

    backend = (params.get("backend") or context.get("syntax_backend") or "spacy").lower()
    if backend == "spacy":
        return _parse_with_spacy(tokens, params, context)
    if backend == "stanza":
        return _parse_with_stanza(tokens, params, context)

    raise ValueError(
        f"Unsupported dependency parser backend '{backend}'. "
        "Expected 'spacy' or 'stanza'."
    )


def extract_dependency_features(tokens: pd.DataFrame, params: dict | None = None, context: dict | None = None) -> pd.DataFrame:
    """
    Extract dependency features from structured VoxAtlas data.
    
    This public function belongs to the syntax layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : pd.DataFrame
        Token-level annotation table used for morphological or syntactic computation.
    params : dict | None
        Resolved feature configuration for this invocation. Keys are feature-specific and merged from defaults and pipeline settings.
    context : dict | None
        Additional pipeline metadata, configuration, and dependency outputs available to the current computation stage.
    
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
        value = extract_dependency_features(tokens=..., params=..., context=...)
        print(value)
    """
    if tokens is None:
        raise ValueError("syntax.dependencies requires token units")

    ordered_tokens = tokens.reset_index(drop=True)
    if has_dependency_annotations(ordered_tokens):
        return build_dependency_table_from_annotations(ordered_tokens)

    return parse_dependency_annotations(ordered_tokens, params=params, context=context)


def signed_head_distance(table: pd.DataFrame) -> pd.Series:
    """
    Provide the ``signed_head_distance`` public API.
    
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
        value = signed_head_distance(table=...)
        print(value)
    """
    distances = []
    for row in table.itertuples(index=False):
        if pd.isna(row.head_id):
            distances.append(0.0)
        else:
            distances.append(float(row.head_id - row.token_id))
    return pd.Series(distances, index=table["token_id"], dtype="float32")


def absolute_dependency_distance(table: pd.DataFrame) -> pd.Series:
    """
    Provide the ``absolute_dependency_distance`` public API.
    
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
        value = absolute_dependency_distance(table=...)
        print(value)
    """
    return signed_head_distance(table).abs().astype("float32")
