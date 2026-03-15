from pathlib import Path

import numpy as np

from voxatlas.audio.audio import Audio


def _load_waveform(path: str):
    extension = Path(path).suffix.lower()

    if extension == ".wav":
        import soundfile as sf

        waveform, sample_rate = sf.read(path)
    elif extension == ".mp4":
        from moviepy.editor import VideoFileClip

        with VideoFileClip(path) as clip:
            audio = clip.audio

            if audio is None:
                raise ValueError(f"No audio track found in file: {path}")

            waveform = audio.to_soundarray()
            sample_rate = audio.fps
    else:
        raise ValueError(f"Unsupported audio format: {extension}")

    waveform = np.asarray(waveform).astype(np.float32, copy=False)

    return waveform, sample_rate


def load_audio(path: str, channel_mode: str = "auto") -> list[Audio]:
    """
    Load an audio/video file into one or more ``Audio`` objects.

    Supported inputs are ``.wav`` and ``.mp4`` files. Multi-channel inputs can be
    kept as separate channels or mixed down to mono based on ``channel_mode``.

    Parameters
    ----------
    path : str
        Path to a ``.wav`` audio file or ``.mp4`` video file with an audio track.
    channel_mode : str
        Channel handling strategy:

        - ``"auto"``: return mono as one item; stereo as two channel-split items;
          reject inputs with more than 2 channels.
        - ``"mono"``: average all channels into a single mono waveform.
        - ``"split"``: return one ``Audio`` object per input channel.
    
    Returns
    -------
    list[Audio]
        Loaded audio streams as ``Audio`` objects with ``float32`` waveforms.
    
    Examples
    --------
    >>> from voxatlas.audio.loader import load_audio
    >>> # Let the loader infer channel behavior (mono -> 1, stereo -> 2).
    >>> streams = load_audio("samples/example.wav")
    >>> # Force mono downmix.
    >>> mono = load_audio("samples/example.wav", channel_mode="mono")
    """
    if channel_mode not in {"auto", "mono", "split"}:
        raise ValueError(
            "channel_mode must be one of: 'auto', 'mono', 'split'"
        )

    waveform, sample_rate = _load_waveform(path)

    if waveform.ndim == 1:
        return [
            Audio(
                waveform=waveform,
                sample_rate=sample_rate,
                path=path,
                channel=None,
            )
        ]

    if waveform.ndim != 2:
        raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

    n_channels = waveform.shape[1]

    if channel_mode == "mono":
        waveform = waveform.mean(axis=1)
        return [
            Audio(
                waveform=waveform.astype(np.float32, copy=False),
                sample_rate=sample_rate,
                path=path,
                channel=None,
            )
        ]

    if channel_mode == "split":
        return [
            Audio(
                waveform=waveform[:, channel].astype(np.float32, copy=False),
                sample_rate=sample_rate,
                path=path,
                channel=channel,
            )
            for channel in range(n_channels)
        ]

    if n_channels == 1:
        return [
            Audio(
                waveform=waveform[:, 0].astype(np.float32, copy=False),
                sample_rate=sample_rate,
                path=path,
                channel=None,
            )
        ]

    if n_channels == 2:
        return [
            Audio(
                waveform=waveform[:, channel].astype(np.float32, copy=False),
                sample_rate=sample_rate,
                path=path,
                channel=channel,
            )
            for channel in range(n_channels)
        ]

    raise ValueError(
        f"channel_mode='auto' only supports up to 2 channels, got {n_channels}"
    )
