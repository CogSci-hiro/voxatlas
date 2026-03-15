import numpy as np
from scipy.signal import stft


def _to_samples(value, sr):
    if isinstance(value, float):
        return max(1, int(round(value * sr)))
    return max(1, int(value))


def _hz_to_mel(frequency):
    frequency = np.asarray(frequency, dtype=np.float32)
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel_value):
    mel_value = np.asarray(mel_value, dtype=np.float32)
    return 700.0 * (10.0 ** (mel_value / 2595.0) - 1.0)


def compute_stft(signal, sr, frame_length, frame_step):
    """
    Compute stft from VoxAtlas inputs.
    
    This public function belongs to the acoustic layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    signal : object
        One-dimensional waveform or contour array with shape ``(n_samples,)``.
    sr : object
        Sampling rate of the waveform in Hertz.
    frame_length : object
        Analysis window duration in seconds or samples, depending on the helper.
    frame_step : object
        Hop size between successive analysis frames in seconds or samples, depending on the helper.
    
    Returns
    -------
    tuple
        Tuple of arrays or values produced by the computation.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.acoustic.spectrogram_utils import compute_stft
    >>> times, freqs, stft_matrix = compute_stft(
    ...     signal=np.zeros(1600, dtype=np.float32),
    ...     sr=16000,
    ...     frame_length=0.02,
    ...     frame_step=0.01,
    ... )
    >>> (times.ndim, freqs.ndim, stft_matrix.ndim)
    (1, 1, 2)
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError("compute_stft expects a 1D signal")

    frame_length_samples = _to_samples(frame_length, sr)
    frame_step_samples = _to_samples(frame_step, sr)

    if signal.size == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    nperseg = min(frame_length_samples, signal.size)
    noverlap = min(max(0, nperseg - frame_step_samples), max(0, nperseg - 1))

    frequencies, times, stft_matrix = stft(
        signal,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    return (
        times.astype(np.float32),
        frequencies.astype(np.float32),
        np.abs(stft_matrix).T.astype(np.float32),
    )


def compute_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """
    Compute mel filterbank from VoxAtlas inputs.
    
    This public function belongs to the acoustic layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    sr : object
        Sampling rate of the waveform in Hertz.
    n_fft : object
        Argument used by the acoustic API.
    n_mels : object
        Argument used by the acoustic API.
    fmin : object
        Lower frequency bound in Hertz used during pitch estimation.
    fmax : object
        Upper frequency bound in Hertz used during pitch estimation.
    
    Returns
    -------
    tuple
        Tuple of arrays or values produced by the computation.
    
    Examples
    --------
    >>> from voxatlas.acoustic.spectrogram_utils import compute_mel_filterbank
    >>> fb, centers = compute_mel_filterbank(sr=16000, n_fft=512, n_mels=10, fmin=0.0, fmax=8000.0)
    >>> (fb.shape[0], len(centers))
    (10, 10)
    """
    if fmax is None:
        fmax = sr / 2.0

    mel_min = _hz_to_mel(float(fmin))
    mel_max = _hz_to_mel(float(fmax))
    mel_points = np.linspace(mel_min, mel_max, int(n_mels) + 2, dtype=np.float32)
    hz_points = _mel_to_hz(mel_points)
    fft_frequencies = np.linspace(0.0, sr / 2.0, int(n_fft // 2) + 1, dtype=np.float32)

    filterbank = np.zeros((int(n_mels), len(fft_frequencies)), dtype=np.float32)

    for m in range(1, int(n_mels) + 1):
        left = hz_points[m - 1]
        center = hz_points[m]
        right = hz_points[m + 1]

        if center <= left or right <= center:
            continue

        left_mask = (fft_frequencies >= left) & (fft_frequencies <= center)
        right_mask = (fft_frequencies >= center) & (fft_frequencies <= right)

        filterbank[m - 1, left_mask] = (
            (fft_frequencies[left_mask] - left) / max(center - left, 1e-8)
        )
        filterbank[m - 1, right_mask] = (
            (right - fft_frequencies[right_mask]) / max(right - center, 1e-8)
        )

    mel_centers = hz_points[1:-1].astype(np.float32)
    return filterbank, mel_centers


def compute_mel_spectrogram(stft_matrix, mel_filterbank):
    """
    Compute mel spectrogram from VoxAtlas inputs.
    
    This public function belongs to the acoustic layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    stft_matrix : object
        Argument used by the acoustic API.
    mel_filterbank : object
        Argument used by the acoustic API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.acoustic.spectrogram_utils import compute_mel_filterbank, compute_mel_spectrogram, compute_stft
    >>> _, _, stft_matrix = compute_stft(np.zeros(1600, dtype=np.float32), 16000, 0.02, 0.01)
    >>> fb, _ = compute_mel_filterbank(sr=16000, n_fft=512, n_mels=10, fmin=0.0, fmax=8000.0)
    >>> mel = compute_mel_spectrogram(stft_matrix, fb)
    >>> mel.shape[1]
    10
    """
    stft_matrix = np.asarray(stft_matrix, dtype=np.float32)
    mel_filterbank = np.asarray(mel_filterbank, dtype=np.float32)

    if stft_matrix.size == 0:
        return np.zeros((0, mel_filterbank.shape[0]), dtype=np.float32)

    return (stft_matrix @ mel_filterbank.T).astype(np.float32)
