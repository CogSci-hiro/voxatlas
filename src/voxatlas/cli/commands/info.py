"""
Info command.
"""

import argparse


def run(args: argparse.Namespace) -> None:
    """Show VoxAtlas info."""
    print("VoxAtlas CLI")
    print("Feature extraction toolkit")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the info command."""
    parser = subparsers.add_parser(
        "info",
        help="Show information about VoxAtlas",
    )

    parser.set_defaults(func=run)