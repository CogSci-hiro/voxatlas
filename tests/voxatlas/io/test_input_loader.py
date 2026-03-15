from pathlib import Path

import numpy as np
import pandas as pd

from voxatlas.audio.audio import Audio
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.io import input_loader
from voxatlas.io.input_loader import load_dataset
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.pipeline.pipeline import VoxAtlasPipeline
from voxatlas.registry.feature_registry import registry


def _write_wav_placeholder(path: Path) -> None:
    path.write_bytes(b"RIFF")


def _textgrid_content(tier_names: list[str]) -> str:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        "xmax = 1",
        "tiers? <exists>",
        f"size = {len(tier_names)}",
        "item []:",
    ]

    for index, tier_name in enumerate(tier_names, start=1):
        lines.extend(
            [
                f"    item [{index}]:",
                '        class = "IntervalTier"',
                f'        name = "{tier_name}"',
                "        xmin = 0",
                "        xmax = 1",
                "        intervals: size = 1",
                "        intervals [1]:",
                "            xmin = 0",
                "            xmax = 1",
                '            text = "hello"',
            ]
        )

    return "\n".join(lines) + "\n"


def _write_textgrid(path: Path, tier_names: list[str]) -> None:
    path.write_text(_textgrid_content(tier_names), encoding="utf-8")


def _create_dataset(tmp_path: Path, conversation_id: str) -> Path:
    dataset_root = tmp_path / "dataset"
    audio_dir = dataset_root / "audio"
    palign_dir = dataset_root / "alignment" / "palign"
    syll_dir = dataset_root / "alignment" / "syll"
    ipu_dir = dataset_root / "alignment" / "ipu"

    audio_dir.mkdir(parents=True)
    palign_dir.mkdir(parents=True)
    syll_dir.mkdir(parents=True)
    ipu_dir.mkdir(parents=True)

    _write_wav_placeholder(audio_dir / f"{conversation_id}.wav")

    for channel in ("ch1", "ch2"):
        _write_textgrid(
            palign_dir / f"{conversation_id}_{channel}.TextGrid",
            ["TokensAlign", "PhonAlign"],
        )
        _write_textgrid(
            syll_dir / f"{conversation_id}_{channel}.TextGrid",
            ["SyllAlign", "SyllClassAlign"],
        )
        _write_textgrid(
            ipu_dir / f"{conversation_id}_{channel}.TextGrid",
            ["IPU"],
        )

    return dataset_root


class DummyEndToEndFeature(BaseExtractor):
    name = "test.e2e.feature"
    output_units = "word"
    dependencies = []
    default_config = {
        "scale": 1,
    }

    def compute(self, feature_input, params):
        values = pd.Series(
            [params["scale"]] * len(feature_input.units.words)
        )
        return ScalarFeatureOutput(
            feature=self.name,
            unit="word",
            values=values,
        )


if DummyEndToEndFeature.name not in registry.list_features():
    registry.register(DummyEndToEndFeature)


def test_load_dataset_and_run_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    dataset_root = _create_dataset(tmp_path, "conversation01")

    def fake_load_audio(path: str, channel_mode: str = "auto") -> list[Audio]:
        assert channel_mode == "auto"
        return [
            Audio(
                waveform=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                path=path,
                channel=0,
            ),
            Audio(
                waveform=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                path=path,
                channel=1,
            ),
        ]

    monkeypatch.setattr(input_loader, "load_audio", fake_load_audio)
    dataset = load_dataset(str(dataset_root), "conversation01")
    stream = dataset.streams()[0]

    pipeline = VoxAtlasPipeline(
        audio=stream.audio,
        units=stream.units,
        config={
            "features": ["test.e2e.feature"],
            "feature_config": {
                "test.e2e.feature": {
                    "scale": 2,
                },
            },
            "pipeline": {
                "n_jobs": 1,
                "cache": False,
            },
        },
    )

    results = pipeline.run()

    assert isinstance(results, FeatureStore)
    assert results.exists("test.e2e.feature")
    output = results.get("test.e2e.feature")
    assert list(output.values) == [2]
