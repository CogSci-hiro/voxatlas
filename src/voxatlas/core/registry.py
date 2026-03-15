from __future__ import annotations

import re
from dataclasses import dataclass

from voxatlas.features.base_extractor import BaseExtractor


FEATURE_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,}$"
)
VALID_UNITS = {
    "frame",
    "token",
    "phoneme",
    "syllable",
    "sentence",
    "word",
    "ipu",
    "turn",
    "conversation",
}


def validate_feature_name(name: str) -> None:
    """
    Validate a registered feature name.

    Parameters
    ----------
    name : str
        Fully qualified feature name such as ``"acoustic.pitch.f0"``.

    Returns
    -------
    None
        The function returns successfully when the name is valid.

    Raises
    ------
    ValueError
        Raised when the name does not match the VoxAtlas naming convention.

    Notes
    -----
    Feature names must include at least two dot-separated segments.

    Examples
    --------
    Usage example::

        validate_feature_name("acoustic.pitch.f0")
    """
    if not isinstance(name, str) or not FEATURE_NAME_PATTERN.match(name):
        raise ValueError(
            "Feature name must follow the format 'domain.feature' "
            "with optional additional suffix segments"
        )


def validate_units(input_units: str | None, output_units: str | None) -> None:
    """
    Validate declared extractor unit levels.

    Parameters
    ----------
    input_units : str | None
        Declared input unit level.
    output_units : str | None
        Declared output unit level.

    Returns
    -------
    None
        The function returns successfully when both unit labels are valid.

    Raises
    ------
    ValueError
        Raised when either unit label is unsupported.

    Notes
    -----
    ``None`` is allowed for extractors that operate directly on audio or global
    context.

    Examples
    --------
    Usage example::

        validate_units(None, "frame")
    """
    for label, value in {
        "input_units": input_units,
        "output_units": output_units,
    }.items():
        if value is not None and value not in VALID_UNITS:
            raise ValueError(
                f"{label} must be one of {sorted(VALID_UNITS)} or None"
            )


def validate_dependencies(dependencies: list[str]) -> None:
    """
    Validate a list of upstream feature dependencies.

    Parameters
    ----------
    dependencies : list of str
        Upstream feature names required by an extractor.

    Returns
    -------
    None
        The function returns successfully when every dependency is valid.

    Raises
    ------
    ValueError
        Raised when the dependency container or one of its entries is invalid.

    Notes
    -----
    Dependencies are resolved through the registry before pipeline execution.

    Examples
    --------
    Usage example::

        validate_dependencies(["syntax.dependencies"])
    """
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be a list")

    for dependency in dependencies:
        validate_feature_name(dependency)


def validate_extractor_contract(extractor_cls: type[BaseExtractor]) -> None:
    """
    Validate the public contract of an extractor class.

    Parameters
    ----------
    extractor_cls : type of BaseExtractor
        Extractor class to validate.

    Returns
    -------
    None
        The function returns successfully when the class satisfies the registry
        contract.

    Raises
    ------
    ValueError
        Raised when required class attributes or methods are missing.

    Notes
    -----
    The registry checks the extractor name, dependencies, unit declarations,
    and the presence of a callable ``compute`` method.

    Examples
    --------
    Usage example::

        validate_extractor_contract(MyExtractor)
    """
    if not isinstance(extractor_cls, type) or not issubclass(extractor_cls, BaseExtractor):
        raise ValueError("extractor_cls must be a subclass of BaseExtractor")

    if not getattr(extractor_cls, "name", None):
        raise ValueError("Extractor must define a name")

    if not hasattr(extractor_cls, "compute") or not callable(getattr(extractor_cls, "compute")):
        raise ValueError("Extractor must define a callable compute method")

    if not hasattr(extractor_cls, "dependencies"):
        raise ValueError("Extractor must define dependencies")

    validate_feature_name(extractor_cls.name)
    validate_dependencies(extractor_cls.dependencies)
    validate_units(
        getattr(extractor_cls, "input_units", None),
        getattr(extractor_cls, "output_units", None),
    )


class FeatureNotRegisteredError(KeyError):
    """
    Report that a requested feature is unavailable in the registry.

    Parameters
    ----------
    feature_name : str
        Requested feature name.

    Returns
    -------
    FeatureNotRegisteredError
        Initialized exception object.

    Notes
    -----
    The exception may also be raised when a feature is known but unavailable
    because an optional dependency is missing.

    Examples
    --------
    Usage example::

        raise FeatureNotRegisteredError("acoustic.pitch.f0")
    """

    def __init__(self, feature_name: str):
        super().__init__(f"Feature '{feature_name}' is not registered")
        self.feature_name = feature_name


@dataclass(frozen=True)
class FeatureRegistryEntry:
    """
    Store metadata for one registered feature.

    Parameters
    ----------
    name : str
        Fully qualified feature name.
    cls : type of BaseExtractor | None
        Extractor class implementing the feature, or ``None`` when unavailable.
    dependencies : tuple of str
        Upstream feature dependencies.
    input_units : str | None
        Declared input unit level.
    output_units : str | None
        Declared output unit level.
    available : bool
        Whether the feature can be instantiated in the current environment.
    missing_dependency : str | None
        Optional missing dependency name for unavailable features.
    module_name : str | None
        Python module where the feature is defined.

    Returns
    -------
    FeatureRegistryEntry
        Immutable metadata entry.

    Notes
    -----
    Registry entries allow the CLI and developer tooling to inspect features
    without instantiating every extractor class.

    Examples
    --------
    Usage example::

        entry = registry.get_entry("acoustic.pitch.f0")
        print(entry.dependencies)
    """

    name: str
    cls: type[BaseExtractor] | None
    dependencies: tuple[str, ...]
    input_units: str | None
    output_units: str | None
    available: bool = True
    missing_dependency: str | None = None
    module_name: str | None = None


class FeatureRegistry:
    """
    Register, inspect, and resolve VoxAtlas feature extractors.

    The registry is the central lookup table used by discovery, the CLI, and
    the pipeline. It stores extractor metadata and optionally tracks features
    that are unavailable because optional dependencies are missing.

    Examples
    --------
    Usage example::

        registry = FeatureRegistry()
        registry.register(MyExtractor)
        print(registry.list_features())
    """

    def __init__(self) -> None:
        self._entries: dict[str, FeatureRegistryEntry] = {}

    def register(
        self,
        extractor_cls: type[BaseExtractor],
    ) -> type[BaseExtractor]:
        """
        Register an available extractor class.

        Parameters
        ----------
        extractor_cls : type of BaseExtractor
            Extractor class to register.

        Returns
        -------
        type of BaseExtractor
            The same extractor class, which makes this method usable as a
            decorator target.

        Raises
        ------
        ValueError
            Raised when the extractor contract is invalid or the feature name is
            already registered to a different class.

        Notes
        -----
        Re-registering the same class is treated as a no-op.

        Examples
        --------
        Usage example::

            registry.register(MyExtractor)
        """
        validate_extractor_contract(extractor_cls)
        name = extractor_cls.name

        if name in self._entries:
            existing = self._entries[name].cls
            if existing is extractor_cls:
                return extractor_cls
            raise ValueError(f"Feature '{name}' is already registered")

        entry = FeatureRegistryEntry(
            name=name,
            cls=extractor_cls,
            dependencies=tuple(getattr(extractor_cls, "dependencies", [])),
            input_units=getattr(extractor_cls, "input_units", None),
            output_units=getattr(extractor_cls, "output_units", None),
            available=True,
            module_name=getattr(extractor_cls, "__module__", None),
        )
        self._entries[name] = entry
        return extractor_cls

    def register_unavailable(
        self,
        *,
        name: str,
        dependencies: list[str] | tuple[str, ...] | None = None,
        input_units: str | None = None,
        output_units: str | None = None,
        missing_dependency: str | None = None,
        module_name: str | None = None,
    ) -> None:
        """
        Register metadata for a feature that cannot currently be imported.

        Parameters
        ----------
        name : str
            Feature name.
        dependencies : list of str | tuple of str | None
            Declared upstream feature dependencies.
        input_units : str | None
            Declared input unit level.
        output_units : str | None
            Declared output unit level.
        missing_dependency : str | None
            Optional missing dependency name.
        module_name : str | None
            Module that defines the feature.

        Returns
        -------
        None
            The registry is updated in place.

        Notes
        -----
        This path is used by feature discovery so the CLI can still report
        features that are unavailable in the current environment.

        Examples
        --------
        Usage example::

            registry.register_unavailable(
                name="syntax.dependencies",
                missing_dependency="spacy",
            )
        """
        validate_feature_name(name)
        validate_units(input_units, output_units)
        dependency_list = list(dependencies or [])
        validate_dependencies(dependency_list)

        if name in self._entries:
            return

        self._entries[name] = FeatureRegistryEntry(
            name=name,
            cls=None,
            dependencies=tuple(dependency_list),
            input_units=input_units,
            output_units=output_units,
            available=False,
            missing_dependency=missing_dependency,
            module_name=module_name,
        )

    def get(self, name: str) -> type[BaseExtractor]:
        """
        Resolve an extractor class from its feature name.

        Parameters
        ----------
        name : str
            Requested feature name.

        Returns
        -------
        type of BaseExtractor
            Extractor class registered for the feature.

        Raises
        ------
        FeatureNotRegisteredError
            Raised when the feature is unknown or unavailable.

        Examples
        --------
        Usage example::

            extractor_cls = registry.get("acoustic.pitch.f0")
            print(extractor_cls.__name__)
        """
        entry = self.get_entry(name)
        if entry.cls is None:
            detail = (
                f" because dependency '{entry.missing_dependency}' is not installed"
                if entry.missing_dependency
                else ""
            )
            raise FeatureNotRegisteredError(f"{name}{detail}")
        return entry.cls

    def get_entry(self, name: str) -> FeatureRegistryEntry:
        """
        Retrieve registry metadata for one feature.

        Parameters
        ----------
        name : str
            Requested feature name.

        Returns
        -------
        FeatureRegistryEntry
            Metadata entry for the feature.

        Raises
        ------
        FeatureNotRegisteredError
            Raised when the feature name is unknown.

        Examples
        --------
        Usage example::

            entry = registry.get_entry("acoustic.pitch.f0")
            print(entry.available)
        """
        validate_feature_name(name)

        try:
            return self._entries[name]
        except KeyError as exc:
            raise FeatureNotRegisteredError(name) from exc

    def list(self) -> list[FeatureRegistryEntry]:
        """
        Return every registered feature entry in sorted order.

        Returns
        -------
        list of FeatureRegistryEntry
            Registered entries sorted by feature name.

        Examples
        --------
        Usage example::

            print(registry.list())
        """
        return [self._entries[name] for name in sorted(self._entries)]

    def list_features(self) -> list[str]:
        """
        Return the sorted list of registered feature names.

        Returns
        -------
        list of str
            Registered feature names.

        Examples
        --------
        Usage example::

            print(registry.list_features())
        """
        return [entry.name for entry in self.list()]

    def by_family(self, family_name: str) -> list[FeatureRegistryEntry]:
        """
        Return all entries that belong to a feature family.

        Parameters
        ----------
        family_name : str
            Family prefix such as ``"acoustic.pitch"``.

        Returns
        -------
        list of FeatureRegistryEntry
            Entries whose names match the requested family prefix.

        Examples
        --------
        Usage example::

            entries = registry.by_family("acoustic.pitch")
            print([entry.name for entry in entries])
        """
        validate_feature_name(family_name)
        prefix = f"{family_name}."
        return [
            entry
            for entry in self.list()
            if entry.name == family_name or entry.name.startswith(prefix)
        ]

    def grouped(self) -> dict[str, list[FeatureRegistryEntry]]:
        """
        Group registered entries by their first two name segments.

        Returns
        -------
        dict
            Mapping from family prefix to registry entries.

        Examples
        --------
        Usage example::

            grouped = registry.grouped()
            print(grouped.keys())
        """
        grouped: dict[str, list[FeatureRegistryEntry]] = {}

        for entry in self.list():
            parts = entry.name.split(".")
            family = ".".join(parts[:2]) if len(parts) >= 2 else entry.name
            grouped.setdefault(family, []).append(entry)

        return grouped

    def clear(self) -> None:
        """
        Remove all registry entries.

        Returns
        -------
        None
            The registry is cleared in place.

        Examples
        --------
        Usage example::

            registry.clear()
        """
        self._entries.clear()


registry = FeatureRegistry()


def register_feature(
    extractor_cls: type[BaseExtractor],
) -> type[BaseExtractor]:
    """
    Register an extractor class in the global registry.

    Parameters
    ----------
    extractor_cls : type of BaseExtractor
        Extractor class to register.

    Returns
    -------
    type of BaseExtractor
        The same extractor class.

    Examples
    --------
    Usage example::

        register_feature(MyExtractor)
    """
    return registry.register(extractor_cls)
