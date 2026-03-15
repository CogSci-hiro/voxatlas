import pytest
from importlib import import_module

from voxatlas.core.registry import FeatureNotRegisteredError, FeatureRegistry
from voxatlas.features.base_extractor import BaseExtractor


class DummyExtractor(BaseExtractor):
    name = "test.family.feature"
    dependencies = []

    def compute(self, feature_input):
        return None


class AnotherDummyExtractor(BaseExtractor):
    name = "test.other.feature"
    dependencies = []

    def compute(self, feature_input):
        return None


def test_register_returns_extractor_from_registry():
    registry = FeatureRegistry()

    registry.register(DummyExtractor)

    assert registry.get("test.family.feature") is DummyExtractor
    entry = registry.get_entry("test.family.feature")

    assert entry.name == "test.family.feature"
    assert entry.cls is DummyExtractor
    assert entry.dependencies == ()
    assert entry.input_units is None
    assert entry.output_units is None


def test_register_raises_for_duplicate_feature_name():
    registry = FeatureRegistry()

    registry.register(DummyExtractor)

    class DuplicateNameExtractor(BaseExtractor):
        name = "test.family.feature"
        dependencies = []

        def compute(self, feature_input, params):
            return None

    with pytest.raises(ValueError):
        registry.register(DuplicateNameExtractor)


def test_get_raises_for_missing_feature():
    registry = FeatureRegistry()

    with pytest.raises(FeatureNotRegisteredError):
        registry.get("unknown.feature.test")


def test_list_features_returns_registered_feature_names():
    registry = FeatureRegistry()

    registry.register(AnotherDummyExtractor)
    registry.register(DummyExtractor)

    assert registry.list_features() == [
        "test.family.feature",
        "test.other.feature",
    ]


def test_by_family_returns_matching_feature_entries():
    registry = FeatureRegistry()

    registry.register(AnotherDummyExtractor)
    registry.register(DummyExtractor)

    family_entries = registry.by_family("test.family")

    assert [entry.name for entry in family_entries] == ["test.family.feature"]


def test_register_feature_decorator_registers_class(monkeypatch):
    core_registry = import_module("voxatlas.core.registry")
    local_registry = FeatureRegistry()
    monkeypatch.setattr(core_registry, "registry", local_registry)

    @core_registry.register_feature
    class DecoratedExtractor(BaseExtractor):
        name = "decorated.family.feature"
        dependencies = []

        def compute(self, feature_input, params):
            return None

    assert local_registry.get("decorated.family.feature") is DecoratedExtractor
