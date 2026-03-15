from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxatlas.config.config import load_and_prepare_config
from voxatlas.io.input_loader import load_dataset
from voxatlas.pipeline.pipeline import VoxAtlasPipeline

from ._shared import ensure_feature_discovery, write_feature_output


def discover_conversation_ids(input_root: Path) -> list[str]:
    """
    Discover conversation identifiers from a dataset root.

    Parameters
    ----------
    input_root : Path
        Dataset root containing ``audio/`` and optional
        ``alignment/palign/`` content.

    Returns
    -------
    list of str
        Sorted conversation identifiers inferred from filenames.

    Notes
    -----
    Conversation ids are extracted from ``audio/*.wav`` and
    ``alignment/palign/*_ch*.TextGrid`` filenames.

    Examples
    --------
    Usage example::

        ids = discover_conversation_ids(Path("/path/to/dataset"))
        print(ids)
    """
    conversation_ids = set()
    audio_dir = input_root / "audio"
    palign_dir = input_root / "alignment" / "palign"

    if audio_dir.exists():
        for path in audio_dir.glob("*.wav"):
            conversation_ids.add(path.stem)

    if palign_dir.exists():
        for path in palign_dir.glob("*_ch*.TextGrid"):
            name = path.stem
            if "_ch" in name:
                conversation_ids.add(name.rsplit("_ch", 1)[0])

    return sorted(conversation_ids)


def run_conversation(
    input_root: Path,
    output_root: Path,
    conversation_id: str,
    config: dict,
) -> None:
    """
    Run the pipeline for every stream in one conversation.

    Parameters
    ----------
    input_root : Path
        Dataset root directory.
    output_root : Path
        Directory where feature outputs should be written.
    conversation_id : str
        Conversation identifier to process.
    config : dict
        Prepared runtime configuration.

    Returns
    -------
    None
        Outputs are written to disk.

    Notes
    -----
    One output directory is created per conversation stream.

    Examples
    --------
    Usage example::

        run_conversation(input_root, output_root, "conversation01", config)
    """
    dataset = load_dataset(str(input_root), conversation_id)

    for stream_index, stream in enumerate(dataset.streams()):
        pipeline = VoxAtlasPipeline(
            audio=stream.audio,
            units=stream.units,
            config=config,
        )
        results = pipeline.run()

        speaker = (
            stream.units.speaker
            if stream.units is not None and stream.units.speaker is not None
            else f"stream_{stream_index}"
        )
        stream_output_dir = output_root / conversation_id / f"stream_{stream_index}_{speaker}"
        stream_output_dir.mkdir(parents=True, exist_ok=True)

        for feature_name in config["features"]:
            write_feature_output(
                stream_output_dir,
                feature_name,
                results.get(feature_name),
            )

        summary = {
            "conversation_id": conversation_id,
            "stream_index": stream_index,
            "speaker": stream.units.speaker if stream.units is not None else None,
            "audio_channel": stream.audio.channel if stream.audio is not None else None,
            "sample_rate": stream.audio.sample_rate if stream.audio is not None else None,
            "audio_duration": stream.audio.duration if stream.audio is not None else None,
            "has_audio": stream.audio is not None,
            "has_units": stream.units is not None,
            "features": config["features"],
        }
        (stream_output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(
            f"Processed conversation={conversation_id} "
            f"stream={stream_index} speaker={speaker}"
        )


def build_runtime_config(args: argparse.Namespace) -> dict:
    """
    Build the runtime configuration used by the ``run`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Prepared configuration dictionary.

    Examples
    --------
    Usage example::

        config = build_runtime_config(args)
        print(config["features"])
    """
    config = load_and_prepare_config(args.config)

    if args.features:
        config["features"] = list(args.features)

    return config


def run(args: argparse.Namespace) -> None:
    """
    Execute the ``voxatlas run`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    None
        Outputs are written to the requested directory.

    Raises
    ------
    ValueError
        Raised when no conversations can be discovered.

    Examples
    --------
    Usage example::

        run(args)
    """
    ensure_feature_discovery()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    config = build_runtime_config(args)
    conversation_ids = (
        [args.conversation_id]
        if args.conversation_id
        else discover_conversation_ids(input_root)
    )

    if not conversation_ids:
        raise ValueError(
            f"No conversations found under: {input_root}\n"
            "Expected audio/*.wav and/or alignment/palign/*_ch1.TextGrid files."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    for conversation_id in conversation_ids:
        run_conversation(
            input_root=input_root,
            output_root=output_root,
            conversation_id=conversation_id,
            config=config,
        )

    print(f"Finished. Outputs written to: {output_root}")


def register(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the ``run`` subcommand with the CLI parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser collection created by the top-level CLI parser.

    Returns
    -------
    None
        The parser is updated in place.

    Examples
    --------
    Usage example::

        register(subparsers)
    """
    parser = subparsers.add_parser(
        "run",
        help="Run the pipeline on a dataset root",
    )
    parser.add_argument("--input-root", required=True, help="Dataset root directory")
    parser.add_argument("--output-root", required=True, help="Directory for feature outputs")
    parser.add_argument("--config", required=True, help="Path to the VoxAtlas YAML config file")
    parser.add_argument(
        "--conversation-id",
        help="Only run a single conversation id",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help="Override config features with feature names from the registry",
    )
    parser.set_defaults(func=run)
