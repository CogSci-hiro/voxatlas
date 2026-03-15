from dataclasses import dataclass
from pathlib import Path

from voxatlas.audio.audio import Audio
from voxatlas.audio.loader import load_audio
from voxatlas.units.alignment_loader import load_textgrid
from voxatlas.units.units import Units


@dataclass
class DatasetStream:
    """
    Represent one aligned stream from a conversation dataset.

    Parameters
    ----------
    audio : Audio | None
        Audio stream for one channel, if available.
    units : Units | None
        Hierarchical unit container for the same channel, if available.

    Returns
    -------
    DatasetStream
        Dataclass describing one multimodal stream.

    Notes
    -----
    A stream may contain audio only, units only, or both modalities.

    Examples
    --------
    >>> from voxatlas.io import DatasetStream
    >>> stream = DatasetStream(audio=None, units=None)
    >>> (stream.audio is None, stream.units is None)
    (True, True)
    """

    audio: Audio | None
    units: Units | None


@dataclass
class DatasetInput:
    """
    Store every stream loaded for one conversation.

    Parameters
    ----------
    audio_streams : list of Audio | None
        Audio streams loaded from the dataset.
    units_streams : list of Units | None
        Alignment streams loaded from TextGrid files.

    Returns
    -------
    DatasetInput
        Dataclass containing per-channel dataset inputs.

    Notes
    -----
    Audio and alignment streams are paired by channel order when both
    modalities are present.

    Examples
    --------
    >>> from voxatlas.io import DatasetInput
    >>> dataset = DatasetInput(audio_streams=None, units_streams=None)
    >>> dataset.streams()
    []
    """

    audio_streams: list[Audio] | None
    units_streams: list[Units] | None

    def streams(self) -> list[DatasetStream]:
        """
        Return paired stream objects for the conversation.

        Returns
        -------
        list of DatasetStream
            Stream objects pairing audio and alignment data where possible.

        Raises
        ------
        ValueError
            Raised when the audio and alignment channel counts differ.

        Notes
        -----
        When only one modality is present, the other field is set to ``None``.

        Examples
        --------
        >>> import numpy as np
        >>> from voxatlas.audio.audio import Audio
        >>> from voxatlas.io import DatasetInput
        >>> audio = Audio(waveform=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        >>> dataset = DatasetInput(audio_streams=[audio], units_streams=None)
        >>> streams = dataset.streams()
        >>> len(streams)
        1
        >>> (streams[0].audio is not None, streams[0].units is None)
        (True, True)
        """
        if self.audio_streams is not None and self.units_streams is not None:
            if len(self.audio_streams) != len(self.units_streams):
                raise ValueError(
                    "Audio/alignment channel mismatch.\n"
                    f"Found {len(self.audio_streams)} audio stream(s) and "
                    f"{len(self.units_streams)} alignment stream(s).\n"
                    "Please ensure audio channels match ch1/ch2 alignment files."
                )

            return [
                DatasetStream(audio=audio, units=units)
                for audio, units in zip(self.audio_streams, self.units_streams)
            ]

        if self.audio_streams is not None:
            return [
                DatasetStream(audio=audio, units=None)
                for audio in self.audio_streams
            ]

        if self.units_streams is not None:
            return [
                DatasetStream(audio=None, units=units)
                for units in self.units_streams
            ]

        return []


def _structure_error(conversation_id: str) -> str:
    return (
        "Invalid dataset structure.\n"
        "Expected:\n\n"
        f"alignment/palign/{conversation_id}_ch1.TextGrid\n"
        f"alignment/palign/{conversation_id}_ch2.TextGrid\n"
        f"alignment/syll/{conversation_id}_ch1.TextGrid\n"
        f"alignment/syll/{conversation_id}_ch2.TextGrid\n"
        f"alignment/ipu/{conversation_id}_ch1.TextGrid\n"
        f"alignment/ipu/{conversation_id}_ch2.TextGrid\n\n"
        "Please rename files accordingly."
    )


def _validate_tiers(
    tiers: dict,
    expected_tiers: list[str],
    path: Path,
    label: str,
) -> None:
    for tier_name in expected_tiers:
        if tier_name not in tiers:
            expected = "\n".join(expected_tiers)
            raise ValueError(
                f"Invalid SPPAS alignment file.\n\n"
                f"Missing tier: {tier_name}\n\n"
                f"Expected tiers for {label} file:\n\n"
                f"{expected}\n\n"
                f"Please export SPPAS {label} output again.\n"
                f"File: {path}"
            )


def _validate_structure(dataset_root: Path, conversation_id: str) -> dict[str, object]:
    if not dataset_root.exists():
        raise ValueError(
            f"Dataset root does not exist: {dataset_root}\n"
            "Please provide a valid dataset directory."
        )

    if not dataset_root.is_dir():
        raise ValueError(
            f"Dataset root is not a directory: {dataset_root}\n"
            "Please provide a valid dataset directory."
        )

    audio_dir = dataset_root / "audio"
    alignment_dir = dataset_root / "alignment"
    palign_dir = alignment_dir / "palign"
    syll_dir = alignment_dir / "syll"
    ipu_dir = alignment_dir / "ipu"

    audio_path = audio_dir / f"{conversation_id}.wav"
    alignment_paths = {
        "palign": {
            "ch1": palign_dir / f"{conversation_id}_ch1.TextGrid",
            "ch2": palign_dir / f"{conversation_id}_ch2.TextGrid",
        },
        "syll": {
            "ch1": syll_dir / f"{conversation_id}_ch1.TextGrid",
            "ch2": syll_dir / f"{conversation_id}_ch2.TextGrid",
        },
        "ipu": {
            "ch1": ipu_dir / f"{conversation_id}_ch1.TextGrid",
            "ch2": ipu_dir / f"{conversation_id}_ch2.TextGrid",
        },
    }

    has_audio = audio_path.exists()
    has_alignment = alignment_dir.exists()

    if not has_audio and not has_alignment:
        raise ValueError(
            "Invalid dataset structure.\n"
            "Expected at least one modality under:\n\n"
            "audio/\n"
            "alignment/\n\n"
            "Please provide audio and/or alignment files."
        )

    if has_alignment:
        missing_dirs = [
            path for path in (palign_dir, syll_dir, ipu_dir)
            if not path.exists()
        ]
        if missing_dirs:
            raise ValueError(
                _structure_error(conversation_id)
            )

        for group_paths in alignment_paths.values():
            for path in group_paths.values():
                if not path.exists():
                    raise ValueError(_structure_error(conversation_id))

    return {
        "audio_path": audio_path if has_audio else None,
        "alignment_paths": alignment_paths if has_alignment else None,
    }


def _load_audio(audio_path: Path | None) -> list[Audio] | None:
    if audio_path is None:
        return None

    return load_audio(str(audio_path), channel_mode="auto")


def _load_alignments(alignment_paths: dict[str, dict[str, Path]] | None) -> list[Units] | None:
    if alignment_paths is None:
        return None

    units_streams = []
    speaker_map = {
        "ch1": "A",
        "ch2": "B",
    }

    for channel, speaker in speaker_map.items():
        palign_tiers = load_textgrid(alignment_paths["palign"][channel])
        syll_tiers = load_textgrid(alignment_paths["syll"][channel])
        ipu_tiers = load_textgrid(alignment_paths["ipu"][channel])

        _validate_tiers(
            palign_tiers,
            ["TokensAlign", "PhonAlign"],
            alignment_paths["palign"][channel],
            "palign",
        )
        _validate_tiers(
            syll_tiers,
            ["SyllAlign", "SyllClassAlign"],
            alignment_paths["syll"][channel],
            "syll",
        )
        _validate_tiers(
            ipu_tiers,
            ["IPU"],
            alignment_paths["ipu"][channel],
            "ipu",
        )

        units_streams.append(
            Units(
                phonemes=palign_tiers["PhonAlign"],
                syllables=syll_tiers["SyllAlign"],
                words=palign_tiers["TokensAlign"],
                ipus=ipu_tiers["IPU"],
                speaker=speaker,
            )
        )

    return units_streams


def load_dataset(dataset_root: str, conversation_id: str) -> DatasetInput:
    """
    Load audio and alignment inputs for one conversation.

    Parameters
    ----------
    dataset_root : str
        Root directory containing ``audio/`` and ``alignment/`` subdirectories.
    conversation_id : str
        Conversation identifier shared by the audio and alignment files.

    Returns
    -------
    DatasetInput
        Loaded dataset object with channel-wise streams.

    Raises
    ------
    ValueError
        Raised when the directory layout is invalid or required files are
        missing.

    Notes
    -----
    VoxAtlas expects the SPPAS-style alignment layout used by the repository
    examples and tests.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from voxatlas.io import load_dataset
    >>>
    >>> def _write_textgrid(path: Path, tier_names: list[str]) -> None:
    ...     items = []
    ...     for idx, name in enumerate(tier_names, start=1):
    ...         items.extend(
    ...             [
    ...                 f"item [{idx}]:",
    ...                 f'    name = "{name}"',
    ...                 "    intervals [1]:",
    ...                 "        xmin = 0",
    ...                 "        xmax = 0.5",
    ...                 '        text = "x"',
    ...             ]
    ...         )
    ...     path.write_text("\\n".join(items) + "\\n", encoding="utf-8")
    >>>
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     root = Path(tmp)
    ...     (root / "alignment" / "palign").mkdir(parents=True)
    ...     (root / "alignment" / "syll").mkdir(parents=True)
    ...     (root / "alignment" / "ipu").mkdir(parents=True)
    ...     conv = "conversation01"
    ...     for ch in ("ch1", "ch2"):
    ...         _write_textgrid(
    ...             root / "alignment" / "palign" / f"{conv}_{ch}.TextGrid",
    ...             ["TokensAlign", "PhonAlign"],
    ...         )
    ...         _write_textgrid(
    ...             root / "alignment" / "syll" / f"{conv}_{ch}.TextGrid",
    ...             ["SyllAlign", "SyllClassAlign"],
    ...         )
    ...         _write_textgrid(
    ...             root / "alignment" / "ipu" / f"{conv}_{ch}.TextGrid",
    ...             ["IPU"],
    ...         )
    ...     dataset = load_dataset(str(root), conv)
    ...     streams = dataset.streams()
    ...     (len(streams), streams[0].units.speaker, streams[1].units.speaker)
    (2, 'A', 'B')
    """
    dataset_root_path = Path(dataset_root)
    validated = _validate_structure(dataset_root_path, conversation_id)
    audio_streams = _load_audio(validated["audio_path"])
    units_streams = _load_alignments(validated["alignment_paths"])

    if audio_streams is None and units_streams is None:
        raise ValueError(
            "No dataset inputs found.\n"
            "Please provide audio and/or alignment files."
        )

    if audio_streams is not None and units_streams is not None:
        if len(audio_streams) != len(units_streams):
            raise ValueError(
                "Audio/alignment channel mismatch.\n"
                f"Found {len(audio_streams)} audio stream(s) and "
                f"{len(units_streams)} alignment stream(s).\n"
                "Please ensure audio channels match ch1/ch2 alignment files."
            )

    return DatasetInput(
        audio_streams=audio_streams,
        units_streams=units_streams,
    )
