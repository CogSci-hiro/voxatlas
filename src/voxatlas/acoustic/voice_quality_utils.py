import numpy as np
from scipy.signal import correlate


def _is_voiced(value):
    return np.isfinite(value) and value > 0.0


def _frame_signal_like(signal, n_frames):
    signal = np.asarray(signal, dtype=np.float32)
    n_frames = int(n_frames)

    if n_frames <= 0:
        return []

    edges = np.linspace(0, len(signal), n_frames + 1, dtype=int)
    frames = []

    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            end = min(len(signal), start + 1)
        frames.append(signal[start:end])

    return frames


def compute_jitter(f0_values):
    r"""
    Compute frame-to-frame relative pitch perturbation.
    
    Parameters
    ----------
    f0_values : array-like
        Frame-level F0 contour with shape ``(n_frames,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 jitter contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    For adjacent voiced frames, jitter is the absolute frame-to-frame F0
    difference normalized by the previous voiced F0 value.
    
    Examples
    --------
        jitter = compute_jitter(f0_values)
        print(jitter[:5])
    
    """
    f0_values = np.asarray(f0_values, dtype=np.float32)
    jitter = np.full(f0_values.shape, np.nan, dtype=np.float32)

    for index in range(1, len(f0_values)):
        previous = float(f0_values[index - 1])
        current = float(f0_values[index])

        if not (_is_voiced(previous) and _is_voiced(current)):
            continue

        jitter[index] = abs(current - previous) / max(previous, 1e-8)

    return jitter


def compute_shimmer(signal, f0_values):
    r"""
    Compute frame-to-frame relative amplitude perturbation.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional waveform with shape ``(n_samples,)``.
    f0_values : array-like
        Frame-level F0 contour with shape ``(n_frames,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 shimmer contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    The waveform is partitioned into frame-like segments matched to the pitch
    contour. Shimmer is the absolute change in peak amplitude between adjacent
    voiced frames, normalized by the previous amplitude.
    
    Examples
    --------
        shimmer = compute_shimmer(signal, f0_values)
        print(shimmer[:5])
    
    """
    f0_values = np.asarray(f0_values, dtype=np.float32)
    frames = _frame_signal_like(signal, len(f0_values))
    amplitudes = np.full(f0_values.shape, np.nan, dtype=np.float32)

    for index, frame in enumerate(frames):
        if frame.size == 0 or not _is_voiced(float(f0_values[index])):
            continue
        amplitudes[index] = float(np.max(np.abs(frame)))

    shimmer = np.full(f0_values.shape, np.nan, dtype=np.float32)

    for index in range(1, len(amplitudes)):
        previous = amplitudes[index - 1]
        current = amplitudes[index]

        if not (np.isfinite(previous) and np.isfinite(current)):
            continue

        shimmer[index] = abs(current - previous) / max(previous, 1e-8)

    return shimmer


def compute_hnr(signal, f0_values):
    r"""
    Compute harmonic-to-noise ratio from framewise autocorrelation.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional waveform with shape ``(n_samples,)``.
    f0_values : array-like
        Frame-level :math:`f_0` contour with shape ``(n_frames,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 HNR contour with shape ``(n_frames,)``.
    
    Algorithm
    ---------
    For each voiced frame, the largest non-zero-lag autocorrelation peak is treated as harmonic energy :math:`H`; the residual :math:`N = R(0)-H` is treated as noise. The returned value is
    
    .. math::
    
       \mathrm{HNR}_t = 10\log_{10}\left(\frac{H}{N}\right).
    
    Examples
    --------
        hnr = compute_hnr(signal, f0_values)
        print(hnr[:5])
    
    """
    f0_values = np.asarray(f0_values, dtype=np.float32)
    frames = _frame_signal_like(signal, len(f0_values))
    hnr_values = np.full(f0_values.shape, np.nan, dtype=np.float32)

    for index, frame in enumerate(frames):
        if frame.size < 2 or not _is_voiced(float(f0_values[index])):
            continue

        frame = frame.astype(np.float32)
        frame = frame - np.mean(frame)

        if np.allclose(frame, 0.0):
            hnr_values[index] = 0.0
            continue

        autocorr = correlate(frame, frame, mode="full")
        autocorr = autocorr[autocorr.size // 2:]

        if autocorr.size < 2 or autocorr[0] <= 0.0:
            continue

        harmonic = float(np.max(autocorr[1:]))
        noise = max(float(autocorr[0] - harmonic), 1e-8)
        harmonic = max(harmonic, 1e-8)
        hnr_values[index] = 10.0 * np.log10(harmonic / noise)

    return hnr_values.astype(np.float32)
