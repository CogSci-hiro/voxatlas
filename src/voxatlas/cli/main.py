"""
VoxAtlas command-line entry point.
"""

import argparse

from voxatlas.logging.logging_config import configure_logging

from .commands import extract, features, info, run


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level VoxAtlas CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with all public subcommands.

    Notes
    -----
    Each subcommand registers its own arguments and callback function.

    Examples
    --------
    >>> import argparse
    >>> from voxatlas.cli.main import build_parser
    >>> parser = build_parser()
    >>> isinstance(parser, argparse.ArgumentParser)
    True
    """
    parser = argparse.ArgumentParser(
        prog="voxatlas",
        description="VoxAtlas feature extraction toolkit",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    features.register(subparsers)
    run.register(subparsers)
    extract.register(subparsers)
    info.register(subparsers)

    return parser


def main() -> None:
    """
    Run the VoxAtlas command-line interface.

    Returns
    -------
    None
        The selected command is executed for its side effects.

    Notes
    -----
    Logging is configured before argument parsing so command handlers share the
    same runtime logging behavior.

    Examples
    --------
    >>> from voxatlas.cli.main import build_parser
    >>> parser = build_parser()
    >>> "VoxAtlas" in parser.format_help()
    True
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
