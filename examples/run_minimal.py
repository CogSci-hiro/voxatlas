from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voxatlas.config.config import load_and_prepare_config
from voxatlas.io.input_loader import load_dataset
from voxatlas.pipeline.pipeline import VoxAtlasPipeline
from voxatlas.registry.discovery import discover_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal VoxAtlas pipeline on all discovered conversations.",
    )
    parser.add_argument(
        "input_root",
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "output_root",
        help="Directory where feature outputs will be written.",
    )
    parser.add_argument(
        "config_path",
        help="Path to the VoxAtlas YAML config file.",
    )
    return parser


def discover_conversation_ids(input_root: Path) -> list[str]:
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


def sanitize_feature_name(feature_name: str) -> str:
    return feature_name.replace(".", "_")


def write_feature_output(output_dir: Path, feature_name: str, output) -> None:
    feature_dir = output_dir / sanitize_feature_name(feature_name)
    feature_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature": getattr(output, "feature", feature_name),
        "unit": getattr(output, "unit", None),
        "output_type": type(output).__name__,
    }
    (feature_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    values = getattr(output, "values", None)

    if isinstance(values, pd.Series):
        values.to_csv(feature_dir / "values.csv", index=True)
        return

    if isinstance(values, pd.DataFrame):
        values.to_csv(feature_dir / "values.csv", index=False)
        return

    if isinstance(values, np.ndarray):
        np.save(feature_dir / "values.npy", values)
        return

    if values is None:
        return

    (feature_dir / "values.txt").write_text(str(values), encoding="utf-8")


def run_conversation(
    input_root: Path,
    output_root: Path,
    conversation_id: str,
    config: dict,
) -> None:
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


def main() -> None:
    args = build_parser().parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    config = load_and_prepare_config(args.config_path)

    pipeline_config = dict(config.get("pipeline", {}))
    pipeline_config["n_jobs"] = 1
    pipeline_config["cache"] = False
    config["pipeline"] = pipeline_config

    discover_features()

    conversation_ids = discover_conversation_ids(input_root)

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


if __name__ == "__main__":
    main()
