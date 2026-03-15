from __future__ import annotations

import argparse
import os

from voxatlas.core.registry import registry

from ._shared import ensure_feature_discovery, format_dependencies


RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def run(args: argparse.Namespace) -> None:
    ensure_feature_discovery()
    use_color = args.color and os.getenv("NO_COLOR") is None

    rows = [
        (
            entry.name,
            entry.input_units or "-",
            entry.output_units or "-",
            format_dependencies(entry.dependencies),
            "available" if entry.available else f"missing:{entry.missing_dependency or 'unknown'}",
        )
        for entry in registry.list()
    ]

    headers = ("feature_name", "input_units", "output_units", "dependencies", "status")
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    for row in rows:
        rendered = list(row)
        rendered[4] = colorize(
            rendered[4],
            GREEN if row[4] == "available" else YELLOW,
            use_color,
        )
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(rendered)))


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "list",
        help="List available features",
    )
    parser.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        help="Disable ANSI colors in the status column",
    )
    parser.set_defaults(color=True)
    parser.set_defaults(func=run)
