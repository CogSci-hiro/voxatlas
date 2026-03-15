"""
Feature extraction command.
"""

import argparse


def run(args: argparse.Namespace) -> None:
    """Execute the extract command."""
    print("Running feature extraction")
    print(f"Audio: {args.audio}")
    print(f"Alignment: {args.alignment}")
    print(f"Config: {args.config}")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract command with the CLI."""
    parser = subparsers.add_parser(
        "extract",
        help="Run feature extraction",
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file",
    )

    parser.add_argument(
        "--alignment",
        required=True,
        help="Alignment file (SPPAS/TextGrid)",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Feature config file",
    )

    parser.set_defaults(func=run)