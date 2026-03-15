import re
from pathlib import Path

import yaml


def load_token_rules(path):
    """
    Load token rules for VoxAtlas processing.
    
    This public function belongs to the nlp layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    path : object
        Filesystem path pointing to an audio file, alignment file, cache file, or resource file.
    
    Returns
    -------
    object
        Loaded resource object ready for downstream VoxAtlas stages.
    
    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from voxatlas.nlp.rule_engine import load_token_rules
    >>> yaml_text = \"\"\"rules:
    ...   - name: greeting
    ...     token_type: interjection
    ...     pattern: [hi, hello]
    ... \"\"\"
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "rules.yaml"
    ...     _ = path.write_text(yaml_text, encoding="utf-8")
    ...     rules = load_token_rules(path)
    ...     rules[0]["token_type"]
    'interjection'
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rules = data.get("rules", [])

    if not isinstance(rules, list):
        raise ValueError("Invalid token rules file. 'rules' must be a list.")

    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"Invalid token rule at index {index}. Rule must be a dict.")

        if "token_type" not in rule:
            raise ValueError(
                f"Invalid token rule at index {index}. Missing field: token_type"
            )

        has_pattern = "pattern" in rule
        has_regex = "regex" in rule

        if has_pattern == has_regex:
            raise ValueError(
                f"Invalid token rule at index {index}. "
                "Each rule must define exactly one of: pattern, regex"
            )

        if has_pattern and not isinstance(rule["pattern"], list):
            raise ValueError(
                f"Invalid token rule at index {index}. 'pattern' must be a list."
            )

    return rules


def apply_token_rules(token, rules):
    """
    Provide the ``apply_token_rules`` public API.
    
    This public function belongs to the nlp layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    token : object
        Argument used by the nlp API.
    rules : object
        Argument used by the nlp API.
    
    Returns
    -------
    dict
        Dictionary containing structured metadata or feature values.
    
    Examples
    --------
    >>> from voxatlas.nlp.rule_engine import apply_token_rules
    >>> rules = [{"name": "greeting", "token_type": "interjection", "pattern": ["hi", "hello"]}]
    >>> apply_token_rules("Hi", rules)["token_type"]
    'interjection'
    """
    normalized = token.strip().lower()

    for rule in rules:
        if "pattern" in rule:
            pattern_values = {value.lower() for value in rule["pattern"]}
            if normalized not in pattern_values:
                continue
        else:
            if re.match(rule["regex"], token) is None:
                continue

        canonical = rule.get("canonical", normalized)
        return {
            "canonical": canonical,
            "analysis": rule.get("analysis", canonical),
            "token_type": rule["token_type"],
            "lemma": rule.get("lemma", canonical),
            "pos": rule.get("pos"),
            "confidence": float(rule.get("confidence", 1.0)),
            "source": f"rule:{rule.get('name', 'unnamed')}",
            "matched_rule": rule.get("name"),
        }

    return None
