from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

from .registry import registry


_DISCOVERED = False


def _literal_or_none(node: ast.AST) -> object | None:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _register_unavailable_features(module_name: str, module_path: str, missing_dependency: str) -> None:
    source = Path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=module_path)

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        base_names = {
            base.id
            for base in node.bases
            if isinstance(base, ast.Name)
        }
        if "BaseExtractor" not in base_names:
            continue

        metadata = {
            "name": None,
            "dependencies": [],
            "input_units": None,
            "output_units": None,
        }

        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                continue

            key = statement.targets[0].id
            if key not in metadata:
                continue

            metadata[key] = _literal_or_none(statement.value)

        feature_name = metadata["name"]
        if not isinstance(feature_name, str):
            continue

        dependencies = metadata["dependencies"]
        if not isinstance(dependencies, list):
            dependencies = []

        registry.register_unavailable(
            name=feature_name,
            dependencies=dependencies,
            input_units=metadata["input_units"] if isinstance(metadata["input_units"], str) or metadata["input_units"] is None else None,
            output_units=metadata["output_units"] if isinstance(metadata["output_units"], str) or metadata["output_units"] is None else None,
            missing_dependency=missing_dependency,
            module_name=module_name,
        )


def discover_features() -> None:
    """
    Provide the ``discover_features`` public API.
    
    This function supports the public feature registry contract used to discover, validate, and resolve extractor classes before execution.
    
    Returns
    -------
    None
        Return value produced by ``discover_features``.
    
    Examples
    --------
    >>> from voxatlas.core.discovery import discover_features
    >>> discover_features() is None
    True
    """
    global _DISCOVERED

    if _DISCOVERED:
        return

    package = importlib.import_module("voxatlas.features")

    for module_info in pkgutil.walk_packages(
        package.__path__,
        prefix=f"{package.__name__}.",
    ):
        module_name = module_info.name

        if any(part.startswith("_") for part in module_name.split(".")):
            continue

        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name.startswith("voxatlas"):
                raise
            if module_info.module_finder is not None:
                spec = module_info.module_finder.find_spec(module_name)
                if spec is not None and spec.origin is not None:
                    _register_unavailable_features(
                        module_name,
                        spec.origin,
                        missing_name,
                    )
            continue

    _DISCOVERED = True
