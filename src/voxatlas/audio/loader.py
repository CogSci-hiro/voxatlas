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
    Load audio for VoxAtlas processing.
    
    This public function belongs to the audio layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    path : str
        Filesystem path pointing to an audio file, alignment file, cache file, or resource file.
    channel_mode : str
        String argument consumed by this API.
    
    Returns
    -------
    list[Audio]
        Return value produced by ``load_audio``.
    
    Examples
    --------
        value = load_audio(path=..., channel_mode=...)
        print(value)
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
