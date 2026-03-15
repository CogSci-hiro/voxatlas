from voxatlas.core.discovery import discover_features
from voxatlas.core.registry import registry


def test_registry_lookup_and_family_grouping():
    discover_features()

    entry = registry.get_entry("lexical.frequency.word")
    family_entries = registry.by_family("lexical.frequency")

    assert entry.name == "lexical.frequency.word"
    assert entry.input_units == "token"
    assert entry.output_units == "token"
    assert "lexical.frequency.lookup" in entry.dependencies
    assert any(item.name == "lexical.frequency.word" for item in family_entries)
