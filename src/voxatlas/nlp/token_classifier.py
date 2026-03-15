import pandas as pd

from .rule_engine import apply_token_rules


def _prepare_vocab_lookup(vocab):
    lookup = {}

    for row in vocab.to_dict(orient="records"):
        lookup[str(row["canonical"]).lower()] = row

    return lookup


def _prepare_mapping_lookup(mapping):
    lookup = {}

    for row in mapping.to_dict(orient="records"):
        lookup[str(row["surface"]).lower()] = row

    return lookup


def _resolve_from_vocab(token, vocab_lookup):
    vocab_row = vocab_lookup.get(token.lower())

    if vocab_row is None:
        return None

    canonical = str(vocab_row["canonical"])
    return {
        "canonical": canonical,
        "analysis": canonical,
        "token_type": vocab_row["token_type"],
        "lemma": vocab_row["lemma"],
        "pos": vocab_row["pos"],
        "confidence": 1.0,
        "source": f"canonical_vocab:{vocab_row['source']}",
        "found_in_vocab": True,
    }


def _resolve_from_mapping(token, mapping_lookup, vocab_lookup):
    mapping_row = mapping_lookup.get(token.lower())

    if mapping_row is None:
        return None

    canonical = str(mapping_row["canonical"])
    vocab_row = vocab_lookup.get(canonical.lower())
    found_in_vocab = vocab_row is not None

    return {
        "canonical": canonical,
        "analysis": canonical,
        "token_type": vocab_row["token_type"] if vocab_row is not None else "mapped",
        "lemma": vocab_row["lemma"] if vocab_row is not None else canonical,
        "pos": vocab_row["pos"] if vocab_row is not None else None,
        "confidence": float(mapping_row["confidence"]),
        "source": f"mapping_table:{mapping_row['rule']}",
        "found_in_vocab": found_in_vocab,
    }


def _classify_token(token, rules, mapping_lookup, vocab_lookup):
    rule_match = apply_token_rules(token, rules)

    if rule_match is not None:
        return {
            "token_surface": token,
            "token_canonical": rule_match["canonical"],
            "token_analysis": rule_match["analysis"],
            "token_type": rule_match["token_type"],
            "lemma": rule_match["lemma"],
            "pos": rule_match["pos"],
            "confidence": rule_match["confidence"],
            "source": rule_match["source"],
            "found_in_vocab": False,
        }

    mapping_match = _resolve_from_mapping(token, mapping_lookup, vocab_lookup)

    if mapping_match is not None:
        return {
            "token_surface": token,
            "token_canonical": mapping_match["canonical"],
            "token_analysis": mapping_match["analysis"],
            "token_type": mapping_match["token_type"],
            "lemma": mapping_match["lemma"],
            "pos": mapping_match["pos"],
            "confidence": mapping_match["confidence"],
            "source": mapping_match["source"],
            "found_in_vocab": mapping_match["found_in_vocab"],
        }

    vocab_match = _resolve_from_vocab(token, vocab_lookup)

    if vocab_match is not None:
        return {
            "token_surface": token,
            "token_canonical": vocab_match["canonical"],
            "token_analysis": vocab_match["analysis"],
            "token_type": vocab_match["token_type"],
            "lemma": vocab_match["lemma"],
            "pos": vocab_match["pos"],
            "confidence": vocab_match["confidence"],
            "source": vocab_match["source"],
            "found_in_vocab": True,
        }

    normalized = token.strip().lower()
    return {
        "token_surface": token,
        "token_canonical": normalized,
        "token_analysis": normalized,
        "token_type": "unknown",
        "lemma": None,
        "pos": None,
        "confidence": 0.0,
        "source": "unknown",
        "found_in_vocab": False,
    }


def classify_tokens(tokens, language_resources):
    """
    Provide the ``classify_tokens`` public API.
    
    This public function belongs to the nlp layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    language_resources : object
        Argument used by the nlp API.
    
    tokens example
    --------------
    token_id | sentence_id | head | dep_rel | text
    1 | 0 | 2 | nsubj | hello
    2 | 0 | 0 | root | world
    
    Returns
    -------
    object
        Return value produced by this API.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.nlp.token_classifier import classify_tokens
    >>> vocab = pd.DataFrame(
    ...     [{"canonical": "hello", "token_type": "word", "lemma": "hello", "pos": "INTJ", "source": "example"}]
    ... )
    >>> mapping = pd.DataFrame(columns=["surface", "canonical", "confidence", "rule"])
    >>> resources = {"canonical_vocab": vocab, "mapping_table": mapping, "token_rules": []}
    >>> out = classify_tokens(["hello"], resources)
    >>> out[0]["token_canonical"]
    'hello'
    """
    vocab = language_resources["canonical_vocab"]
    mapping = language_resources["mapping_table"]
    rules = language_resources["token_rules"]

    if not isinstance(vocab, pd.DataFrame):
        raise ValueError("language_resources['canonical_vocab'] must be a pandas DataFrame")

    if not isinstance(mapping, pd.DataFrame):
        raise ValueError("language_resources['mapping_table'] must be a pandas DataFrame")

    if not isinstance(rules, list):
        raise ValueError("language_resources['token_rules'] must be a list")

    vocab_lookup = _prepare_vocab_lookup(vocab)
    mapping_lookup = _prepare_mapping_lookup(mapping)
    classified = []

    for index, token in enumerate(tokens):
        classified_token = _classify_token(
            token,
            rules,
            mapping_lookup,
            vocab_lookup,
        )
        classified_token["index"] = index
        classified.append(classified_token)

    return classified
