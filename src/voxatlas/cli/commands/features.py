import argparse

from . import features_info, features_list


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "features",
        help="Discover and inspect available features",
    )
    feature_subparsers = parser.add_subparsers(
        dest="features_command",
        required=True,
    )
    features_list.register(feature_subparsers)
    features_info.register(feature_subparsers)
