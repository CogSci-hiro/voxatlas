from dataclasses import dataclass

import numpy as np


@dataclass
class Audio:
    """
    Store waveform data for one VoxAtlas stream.

    Parameters
    ----------
    waveform : ndarray
        One-dimensional waveform array.
    sample_rate : int
        Sampling rate in Hertz.
    path : str | None
        Optional source path for the waveform.
    channel : int | None
        Optional channel index when the waveform came from a multichannel
        recording.

    Returns
    -------
    Audio
        Dataclass describing one audio stream.

    Notes
    -----
    The pipeline treats ``Audio`` as the canonical acoustic input object for
    feature extractors.

    Examples
    --------
    Usage example::

        audio = Audio(waveform=waveform, sample_rate=16000, path="conversation.wav")
        print(audio.duration)
    """

    waveform: np.ndarray
    sample_rate: int
    path: str | None = None
    channel: int | None = None

    @property
    def duration(self) -> float:
        """
        Return the duration of the waveform in seconds.

        Returns
        -------
        float
            Audio duration in seconds.

        Notes
        -----
        Duration is computed directly from waveform length and sample rate.

        Examples
        --------
        Usage example::

            print(audio.duration)
        """
        return len(self.waveform) / self.sample_rate
