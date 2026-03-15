import numpy as np
from scipy.signal import stft


def _to_samples(value, sr):
    if isinstance(value, float):
        return max(1, int(round(value * sr)))
    return max(1, int(value))


def _safe_denominator(values, eps=1e-10):
    return np.maximum(values, eps)


def compute_spectrum(
    signal,
    sr,
    frame_length=0.025,
    frame_step=0.010,
    window="hann",
):
    r"""
    Compute a magnitude short-time spectrum from a waveform.
    
    This is the core spectral front end used by VoxAtlas spectral extractors.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional waveform with shape ``(n_samples,)``.
    sr : int or float
        Sampling rate in Hertz.
    frame_length : float or int, default=0.025
        Analysis-window length in seconds or samples.
    frame_step : float or int, default=0.010
        Hop size between frames in seconds or samples.
    window : str, default="hann"
        Analysis window passed to ``scipy.signal.stft``.
    
    Returns
    -------
    tuple of numpy.ndarray
        Triple ``(time, frequency, magnitude)`` where ``magnitude`` has shape ``(n_frames, n_bins)``.
    
    Algorithm
    ---------
    For each frame, the short-time Fourier transform is computed,
    
    .. math::
    
       X_t(k) = \sum_{n=0}^{N-1} x_t[n] e^{-j2\pi kn/N},
    
    and VoxAtlas stores the magnitude :math:`S_{t,k} = |X_t(k)|`.
    
    Examples
    --------
        time, frequency, spectrum = compute_spectrum(signal, sr=16000)
        print(spectrum.shape)
    
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError("compute_spectrum expects a 1D signal")

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
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(stft_matrix).T.astype(np.float32)

    return (
        times.astype(np.float32),
        frequencies.astype(np.float32),
        magnitude,
    )


def spectral_centroid(spectrum, frequencies):
    r"""
    Compute the spectral centroid of each frame.
    
    Parameters
    ----------
    spectrum : array-like
        Magnitude spectrum with shape ``(n_frames, n_bins)``.
    frequencies : array-like
        Frequency axis with shape ``(n_bins,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 centroid contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    The centroid is the first spectral moment,
    
    .. math::
    
       C_t = \frac{\sum_k S_{t,k}f_k}{\sum_k S_{t,k}}.
    
    Examples
    --------
        centroid = spectral_centroid(spectrum, frequencies)
        print(centroid[:5])
    
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float32)

    weighted_sum = spectrum @ frequencies
    magnitude_sum = _safe_denominator(np.sum(spectrum, axis=1))
    return (weighted_sum / magnitude_sum).astype(np.float32)


def spectral_spread(spectrum, frequencies, centroid=None):
    r"""
    Compute spectral spread around the centroid.
    
    Parameters
    ----------
    spectrum : array-like
        Magnitude spectrum with shape ``(n_frames, n_bins)``.
    frequencies : array-like
        Frequency axis with shape ``(n_bins,)``.
    centroid : array-like, optional
        Precomputed centroid contour.
    
    Returns
    -------
    numpy.ndarray
        Float32 spread contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    Spread is the square root of the second central moment,
    
    .. math::
    
       \mathrm{Spread}_t = \sqrt{\frac{\sum_k S_{t,k}(f_k-C_t)^2}{\sum_k S_{t,k}}}.
    
    Examples
    --------
        spread = spectral_spread(spectrum, frequencies)
        print(spread[:5])
    
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float32)

    if centroid is None:
        centroid = spectral_centroid(spectrum, frequencies)
    else:
        centroid = np.asarray(centroid, dtype=np.float32)

    centered = frequencies[np.newaxis, :] - centroid[:, np.newaxis]
    variance = np.sum(spectrum * (centered ** 2), axis=1) / _safe_denominator(
        np.sum(spectrum, axis=1)
    )
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def spectral_slope(spectrum, frequencies):
    r"""
    Compute frame-level spectral slope.
    
    Parameters
    ----------
    spectrum : array-like
        Magnitude spectrum with shape ``(n_frames, n_bins)``.
    frequencies : array-like
        Frequency axis with shape ``(n_bins,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 slope contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    For each frame, VoxAtlas fits a least-squares line relating magnitude to centered frequency bins. The slope is
    
    .. math::
    
       \hat\beta_t = \frac{\sum_k (f_k-\bar f)(S_{t,k}-\bar S_t)}{\sum_k (f_k-\bar f)^2}.
    
    Examples
    --------
        slope = spectral_slope(spectrum, frequencies)
        print(slope[:5])
    
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float32)

    if spectrum.size == 0:
        return np.array([], dtype=np.float32)

    freq_mean = np.mean(frequencies)
    freq_centered = frequencies - freq_mean
    denominator = float(np.sum(freq_centered ** 2))

    if denominator <= 0.0:
        return np.zeros(spectrum.shape[0], dtype=np.float32)

    spec_mean = np.mean(spectrum, axis=1, keepdims=True)
    numerator = np.sum(freq_centered[np.newaxis, :] * (spectrum - spec_mean), axis=1)
    return (numerator / denominator).astype(np.float32)


def spectral_rolloff(spectrum, frequencies, roll_percent=0.85):
    r"""
    Compute the roll-off frequency of each frame.
    
    Parameters
    ----------
    spectrum : array-like
        Magnitude spectrum with shape ``(n_frames, n_bins)``.
    frequencies : array-like
        Frequency axis with shape ``(n_bins,)``.
    roll_percent : float, default=0.85
        Target cumulative proportion.
    
    Returns
    -------
    numpy.ndarray
        Float32 roll-off contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    The roll-off frequency is the smallest :math:`f_k` such that
    
    .. math::
    
       \sum_{j \le k} S_{t,j} \ge \rho \sum_j S_{t,j},
    
    where :math:`\rho` is ``roll_percent``.
    
    Examples
    --------
        rolloff = spectral_rolloff(spectrum, frequencies, roll_percent=0.85)
        print(rolloff[:5])
    
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float32)

    if spectrum.size == 0:
        return np.array([], dtype=np.float32)

    cumulative = np.cumsum(spectrum, axis=1)
    thresholds = cumulative[:, -1] * float(roll_percent)
    indices = np.argmax(cumulative >= thresholds[:, np.newaxis], axis=1)
    return frequencies[indices].astype(np.float32)


def spectral_flatness(spectrum, eps=1e-10):
    r"""
    Compute spectral flatness.
    
    Parameters
    ----------
    spectrum : array-like
        Magnitude spectrum with shape ``(n_frames, n_bins)``.
    eps : float, default=1e-10
        Numerical floor.
    
    Returns
    -------
    numpy.ndarray
        Float32 flatness contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    Flatness is the ratio of geometric to arithmetic mean,
    
    .. math::
    
       F_t = \frac{\exp\left(\frac{1}{K}\sum_k \log S_{t,k}\right)}{\frac{1}{K}\sum_k S_{t,k}}.
    
    Examples
    --------
        flatness = spectral_flatness(spectrum)
        print(flatness[:5])
    
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)

    if spectrum.size == 0:
        return np.array([], dtype=np.float32)

    safe_spectrum = np.maximum(spectrum, eps)
    geometric_mean = np.exp(np.mean(np.log(safe_spectrum), axis=1))
    arithmetic_mean = _safe_denominator(np.mean(safe_spectrum, axis=1), eps=eps)
    return (geometric_mean / arithmetic_mean).astype(np.float32)


def spectral_flux(spectrum):
    r"""
    Compute inter-frame spectral flux.
    
    Parameters
    ----------
    spectrum : array-like
        Magnitude spectrum with shape ``(n_frames, n_bins)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 flux contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    Flux is the Euclidean norm of successive spectral differences,
    
    .. math::
    
       \mathrm{Flux}_t = \sqrt{\sum_k (S_{t,k} - S_{t-1,k})^2}.
    
    Examples
    --------
        flux = spectral_flux(spectrum)
        print(flux[:5])
    
    """
    spectrum = np.asarray(spectrum, dtype=np.float32)

    if spectrum.size == 0:
        return np.array([], dtype=np.float32)

    diffs = np.diff(spectrum, axis=0, prepend=spectrum[:1])
    return np.sqrt(np.sum(diffs ** 2, axis=1)).astype(np.float32)
