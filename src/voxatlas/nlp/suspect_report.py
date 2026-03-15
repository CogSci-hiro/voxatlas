def _token_context(tokens, index):
    previous_token = tokens[index - 1]["token_surface"] if index > 0 else None
    current_token = tokens[index]["token_surface"]
    next_token = tokens[index + 1]["token_surface"] if index < len(tokens) - 1 else None

    return {
        "previous": previous_token,
        "current": current_token,
        "next": next_token,
    }


def generate_suspect_report(tokens):
    """
    Provide the ``generate_suspect_report`` public API.
    
    This public function belongs to the nlp layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tokens : object
        Token-level annotation table used for morphological or syntactic computation.
    
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
        value = generate_suspect_report(tokens=...)
        print(value)
    """
    report = []

    for index, token in enumerate(tokens):
        reasons = []

        if token["token_type"] == "unknown":
            reasons.append("token classified as unknown")

        if token["source"].startswith("mapping_table:") and float(token["confidence"]) < 1.0:
            reasons.append("token mapped with confidence < 1")

        if not token.get("found_in_vocab", False):
            reasons.append("token not found in canonical vocab")

        for reason in reasons:
            report.append(
                {
                    "token_surface": token["token_surface"],
                    "reason": reason,
                    "context": _token_context(tokens, index),
                }
            )

    return report
