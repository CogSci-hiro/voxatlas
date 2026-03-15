from voxatlas.core.discovery import discover_features
from voxatlas.core.registry import registry


def test_discover_features_registers_builtin_extractors():
    discover_features()

    extractor_cls = registry.get("lexical.frequency.word")

    assert extractor_cls.name == "lexical.frequency.word"
    assert extractor_cls.output_units == "token"
