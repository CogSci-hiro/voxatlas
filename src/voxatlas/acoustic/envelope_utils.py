import numpy as np
from scipy.signal import find_peaks, hilbert


def _to_samples(value, sr):
    if isinstance(value, float):
        return max(1, int(round(value * sr)))
    return max(1, int(value))


def frame_signal(signal, sr, frame_length, frame_step):
    r"""
    Segment a waveform into overlapping analysis frames.
    
    This helper creates the frame grid used by VoxAtlas envelope extractors.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional waveform with shape ``(n_samples,)``.
    sr : int or float
        Sampling rate in Hertz.
    frame_length : float or int
        Analysis-window length in seconds or samples.
    frame_step : float or int
        Hop size between frames in seconds or samples.
    
    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(frames, times)`` where ``frames`` has shape ``(n_frames, frame_length_samples)`` and ``times`` has shape ``(n_frames,)``.
    
    Algorithm
    ---------
    Frame starts are generated at hop intervals and each row of the output matrix stores one local waveform slice :math:`x_t[n]`.
    
    Examples
    --------
        frames, times = frame_signal(signal, sr=16000, frame_length=0.025, frame_step=0.010)
        print(frames.shape, times.shape)
    
    """
    frame_length_samples = _to_samples(frame_length, sr)
    frame_step_samples = _to_samples(frame_step, sr)
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError("frame_signal expects a 1D signal")

    if len(signal) == 0:
        return np.zeros((0, frame_length_samples), dtype=np.float32), np.array([], dtype=np.float32)

    if len(signal) < frame_length_samples:
        padded = np.pad(signal, (0, frame_length_samples - len(signal)))
        return padded[np.newaxis, :], np.array([0.0], dtype=np.float32)

    starts = np.arange(0, len(signal) - frame_length_samples + 1, frame_step_samples)
    frames = np.stack(
        [signal[start:start + frame_length_samples] for start in starts],
        axis=0,
    )
    times = starts.astype(np.float32) / float(sr)
    return frames, times


def compute_rms(signal, sr, frame_length, frame_step):
    r"""
    Compute a frame-level RMS amplitude envelope.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional waveform with shape ``(n_samples,)``.
    sr : int or float
        Sampling rate in Hertz.
    frame_length : float or int
        Analysis-window length in seconds or samples.
    frame_step : float or int
        Hop size between frames in seconds or samples.
    
    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(time, rms_values)`` with frame times and float32 RMS amplitudes.
    
    Algorithm
    ---------
    For each frame :math:`x_t[n]`, VoxAtlas computes
    
    .. math::
    
       \mathrm{RMS}_t = \sqrt{\frac{1}{N}\sum_{n=1}^{N} x_t[n]^2}.
    
    Examples
    --------
        time, rms = compute_rms(signal, sr=16000, frame_length=0.025, frame_step=0.010)
        print(rms[:5])
    
    """
    frames, times = frame_signal(signal, sr, frame_length, frame_step)
    rms_values = np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)
    return times, rms_values


def compute_log_energy(rms_values):
    r"""
    Convert an RMS contour to log energy.
    
    Parameters
    ----------
    rms_values : array-like
        Non-negative RMS contour with shape ``(n_frames,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 array of log-energy values with the same shape as ``rms_values``.
    
    Algorithm
    ---------
    The transformation is
    
    .. math::
    
       e_t = \log(\max(r_t, \varepsilon)),
    
    where :math:`r_t` is RMS amplitude and :math:`\varepsilon` is a numerical floor.
    
    Examples
    --------
        log_energy = compute_log_energy(rms_values)
        print(log_energy[:5])
    
    """
    rms_values = np.asarray(rms_values, dtype=np.float32)
    return np.log(np.maximum(rms_values, 1e-8)).astype(np.float32)


def compute_hilbert(signal):
    r"""
    Compute the analytic-signal envelope using the Hilbert transform.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional waveform with shape ``(n_samples,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 magnitude envelope with shape ``(n_samples,)``.
    
    Algorithm
    ---------
    The analytic signal is
    
    .. math::
    
       z[n] = x[n] + j\,\mathcal{H}\{x[n]\},
    
    and the returned envelope is :math:`|z[n]|`.
    
    Examples
    --------
        envelope = compute_hilbert(signal)
        print(envelope[:5])
    
    """
    signal = np.asarray(signal, dtype=np.float32)
    return np.abs(hilbert(signal)).astype(np.float32)


def compute_derivative(signal):
    r"""
    Compute a first-order backward difference of a contour.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional contour with shape ``(n_samples,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 derivative contour with the same shape as ``signal``.
    
    Algorithm
    ---------
    The derivative is
    
    .. math::
    
       d[n] = x[n] - x[n-1],
    
    with the first sample repeated so the output length matches the input length.
    
    Examples
    --------
        derivative = compute_derivative(signal)
        print(derivative[:5])
    
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    derivative = np.diff(signal, prepend=signal[0])
    return derivative.astype(np.float32)


def detect_peaks(signal, threshold):
    r"""
    Detect local maxima above a threshold.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional contour with shape ``(n_samples,)``.
    threshold : float
        Minimum peak height.
    
    Returns
    -------
    numpy.ndarray
        Integer peak indices.
    
    Algorithm
    ---------
    Local maxima are identified and retained only when :math:`x[n] > \theta`, where :math:`\theta` is the supplied threshold.
    
    Examples
    --------
        peaks = detect_peaks(signal, threshold=0.1)
        print(peaks[:5])
    
    """
    signal = np.asarray(signal, dtype=np.float32)
    peak_indices, _ = find_peaks(signal, height=threshold)
    return peak_indices


def compute_peak_rate(peaks, sr, length):
    r"""
    Convert detected peaks into a frame-rate impulse series.
    
    Parameters
    ----------
    peaks : array-like
        Peak indices.
    sr : float
        Frame sampling rate in Hertz.
    length : int
        Desired output length.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with shape ``(length,)`` whose non-zero values mark detected peaks.
    
    Algorithm
    ---------
    The output series is
    
    .. math::
    
       y_t = f_{\mathrm{frame}}\,\mathbf{1}[t \in P],
    
    where :math:`P` is the set of detected peaks.
    
    Examples
    --------
        peak_rate = compute_peak_rate(peaks, sr=100.0, length=200)
        print(peak_rate[:10])
    
    """
    values = np.zeros(length, dtype=np.float32)
    if len(peaks) > 0:
        values[np.asarray(peaks, dtype=int)] = float(sr)
    return values


def compute_variability(signal):
    r"""
    Compute global contour variability and broadcast it across samples.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional contour with shape ``(n_samples,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with shape ``(n_samples,)`` containing the contour standard deviation.
    
    Algorithm
    ---------
    The function computes
    
    .. math::
    
       \sigma_x = \sqrt{\frac{1}{N}\sum_{n=1}^{N}(x[n]-\bar x)^2},
    
    then repeats :math:`\sigma_x` across the original index.
    
    Examples
    --------
        variability = compute_variability(signal)
        print(variability[:5])
    
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    variability = float(np.std(signal))
    return np.full(signal.shape, variability, dtype=np.float32)


def smooth_signal(signal, smoothing):
    r"""
    Smooth a contour with a moving-average kernel.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional contour with shape ``(n_samples,)``.
    smoothing : int
        Moving-average window length.
    
    Returns
    -------
    numpy.ndarray
        Smoothed float32 contour with the same shape as ``signal``.
    
    Algorithm
    ---------
    A length-``smoothing`` uniform kernel is convolved with the input signal, yielding a local average approximation to the underlying envelope.
    
    Examples
    --------
        smoothed = smooth_signal(signal, smoothing=5)
        print(smoothed[:5])
    
    """
    signal = np.asarray(signal, dtype=np.float32)
    window = max(1, int(smoothing))

    if window <= 1 or signal.size == 0:
        return signal

    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal, kernel, mode="same").astype(np.float32)
