from __future__ import annotations

import argparse

from voxatlas.core.registry import registry

from ._shared import ensure_feature_discovery, extractor_description, format_dependencies


def run(args: argparse.Namespace) -> None:
    ensure_feature_discovery()

    entry = registry.get_entry(args.feature_name)
    extractor_cls = entry.cls
    default_config = getattr(extractor_cls, "default_config", {}) if extractor_cls is not None else {}
    status = (
        "available"
        if entry.available
        else f"missing dependency ({entry.missing_dependency or 'unknown'})"
    )

    print(f"name: {entry.name}")
    print(f"status: {status}")
    print(f"input_units: {entry.input_units or '-'}")
    print(f"output_units: {entry.output_units or '-'}")
    print(f"dependencies: {format_dependencies(entry.dependencies)}")
    print(f"description: {extractor_description(extractor_cls) or '-'}" if extractor_cls is not None else "description: -")
    print("configuration parameters:")

    if not default_config:
        print("  -")
        return

    for key, value in default_config.items():
        print(f"  {key}: {value!r}")


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "info",
        help="Show metadata for a feature",
    )
    parser.add_argument("feature_name", help="Feature name to inspect")
    parser.set_defaults(func=run)
