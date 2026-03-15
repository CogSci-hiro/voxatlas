import numpy as np
from scipy.signal import correlate


def _to_samples(value, sr):
    if isinstance(value, float):
        return max(1, int(round(value * sr)))
    return max(1, int(value))


def _frame_signal(signal, sr, frame_length, frame_step):
    signal = np.asarray(signal, dtype=np.float32)
    frame_length_samples = _to_samples(frame_length, sr)
    frame_step_samples = _to_samples(frame_step, sr)

    if signal.ndim != 1:
        raise ValueError("_frame_signal expects a 1D signal")

    if signal.size == 0:
        return np.zeros((0, frame_length_samples), dtype=np.float32), np.array([], dtype=np.float32)

    if signal.size < frame_length_samples:
        padded = np.pad(signal, (0, frame_length_samples - signal.size))
        return padded[np.newaxis, :], np.array([0.0], dtype=np.float32)

    starts = np.arange(0, signal.size - frame_length_samples + 1, frame_step_samples)
    frames = np.stack(
        [signal[start:start + frame_length_samples] for start in starts],
        axis=0,
    )
    times = starts.astype(np.float32) / float(sr)
    return frames, times


def _is_voiced(value):
    return np.isfinite(value) and value > 0.0


def compute_f0(
    signal,
    sr,
    fmin,
    fmax,
    frame_length=0.040,
    frame_step=0.010,
):
    r"""
    Estimate frame-level fundamental frequency by autocorrelation.
    
    This function implements the low-level pitch-tracking routine used by VoxAtlas acoustic extractors. It operates directly on a one-dimensional waveform and returns a frame-aligned time axis together with an :math:`f_0` contour that downstream pitch and voice-quality features reuse.
    
    Parameters
    ----------
    signal : array-like
        One-dimensional waveform with shape ``(n_samples,)``.
    sr : int or float
        Sampling rate in Hertz.
    fmin : float
        Lower bound of the admissible pitch range in Hertz.
    fmax : float
        Upper bound of the admissible pitch range in Hertz.
    frame_length : float or int, default=0.040
        Analysis-window length in seconds or samples.
    frame_step : float or int, default=0.010
        Hop size between successive frames in seconds or samples.
    
    Returns
    -------
    tuple of numpy.ndarray
        Pair ``(time, values)`` where ``time`` has shape ``(n_frames,)`` and ``values`` contains a float32 :math:`f_0` estimate or ``NaN`` for each frame.
    
    Algorithm
    ---------
    The implementation follows a simple autocorrelation-based estimator.
    
    1. Framing
       The waveform is segmented into overlapping windows and each frame is mean-centered.
    
    2. Candidate lag search
       The one-sided autocorrelation is computed as
    
       .. math::
    
          R(\tau) = \sum_{n=0}^{N-\tau-1} x[n]x[n+\tau].
    
       Only lags corresponding to the admissible period interval
       :math:`[f_s/f_{\max},\ f_s/f_{\min}]` are considered.
    
    3. Voicing and frequency estimation
       Let :math:`\tau^*` be the lag with the largest autocorrelation peak in the candidate region. The frame is accepted as voiced only when the normalized peak exceeds the implementation threshold. The final estimate is
    
       .. math::
    
          \hat f_0 = \frac{f_s}{\tau^*}.
    
    4. Packaging
       Low-energy or unvoiced frames are stored as ``NaN`` so later stages can preserve missingness explicitly.
    
    Examples
    --------
        time, f0 = compute_f0(signal, sr=16000, fmin=75.0, fmax=300.0)
        print(time.shape, f0.shape)
    
    """
    frames, times = _frame_signal(signal, sr, frame_length, frame_step)
    f0_values = np.full(times.shape, np.nan, dtype=np.float32)

    if frames.size == 0:
        return times, f0_values

    min_period = max(1, int(sr / max(float(fmax), 1.0)))
    max_period = max(min_period + 1, int(sr / max(float(fmin), 1.0)))

    for index, frame in enumerate(frames):
        frame = frame.astype(np.float32)
        frame = frame - np.mean(frame)

        energy = float(np.sqrt(np.mean(frame ** 2)))
        if energy < 1e-3 or np.allclose(frame, 0.0):
            continue

        autocorr = correlate(frame, frame, mode="full")
        autocorr = autocorr[autocorr.size // 2:]

        upper = min(max_period, autocorr.size - 1)
        if upper <= min_period:
            continue

        candidate_region = autocorr[min_period:upper + 1]
        if candidate_region.size == 0:
            continue

        peak_offset = int(np.argmax(candidate_region))
        peak_index = min_period + peak_offset
        peak_value = float(candidate_region[peak_offset])
        normalized_peak = peak_value / max(float(autocorr[0]), 1e-8)

        if normalized_peak < 0.3:
            continue

        f0_values[index] = float(sr) / float(peak_index)

    return times.astype(np.float32), f0_values.astype(np.float32)


def compute_f0_derivative(f0_values):
    r"""
    Compute the first temporal difference of a pitch contour.
    
    This helper is used by VoxAtlas pitch-shape features to summarize how
    rapidly F0 changes from one frame to the next.
    
    Parameters
    ----------
    f0_values : array-like
        Frame-level pitch contour with shape ``(n_frames,)``. Unvoiced frames should be encoded as non-positive values or ``NaN``.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with shape ``(n_frames,)`` containing first differences
        for valid adjacent voiced frames and ``NaN`` elsewhere.
    
    Algorithm
    ---------
    For each frame after the first, the function subtracts the previous voiced
    F0 value from the current voiced F0 value. The difference is defined only
    when both adjacent frames satisfy the VoxAtlas voicing rule.
    
    Examples
    --------
        delta = compute_f0_derivative(f0_values)
        print(delta[:5])
    
    """
    f0_values = np.asarray(f0_values, dtype=np.float32)
    derivative = np.full(f0_values.shape, np.nan, dtype=np.float32)

    for index in range(1, len(f0_values)):
        previous = float(f0_values[index - 1])
        current = float(f0_values[index])

        if not (_is_voiced(previous) and _is_voiced(current)):
            continue

        derivative[index] = current - previous

    return derivative


def compute_f0_slope(f0_values, window=5):
    r"""
    Estimate local linear slope of a pitch contour.
    
    This routine provides the frame-level pitch-trend statistic used by VoxAtlas intonational features.
    
    Parameters
    ----------
    f0_values : array-like
        Frame-level pitch contour with shape ``(n_frames,)``.
    window : int, default=5
        Number of neighboring frames used for the local regression window.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with shape ``(n_frames,)`` containing a local least-squares slope for voiced frames.
    
    Algorithm
    ---------
    For each voiced frame, voiced neighbors inside the local window are fit
    with a first-order polynomial. Frames with fewer than two voiced
    observations in the window remain undefined.
    
    Examples
    --------
        slope = compute_f0_slope(f0_values, window=7)
        print(slope[:5])
    
    """
    f0_values = np.asarray(f0_values, dtype=np.float32)
    slopes = np.full(f0_values.shape, np.nan, dtype=np.float32)
    half_window = max(1, int(window) // 2)

    for index, value in enumerate(f0_values):
        if not _is_voiced(float(value)):
            continue

        start = max(0, index - half_window)
        end = min(len(f0_values), index + half_window + 1)
        window_values = f0_values[start:end]
        voiced_mask = np.isfinite(window_values) & (window_values > 0.0)

        if np.count_nonzero(voiced_mask) < 2:
            continue

        x = np.arange(start, end, dtype=np.float32)[voiced_mask]
        y = window_values[voiced_mask].astype(np.float32)
        slope, _ = np.polyfit(x, y, 1)
        slopes[index] = np.float32(slope)

    return slopes


def compute_f0_variability(f0_values):
    r"""
    Compute global pitch dispersion and broadcast it to voiced frames.
    
    Parameters
    ----------
    f0_values : array-like
        Frame-level pitch contour with shape ``(n_frames,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with shape ``(n_frames,)`` containing the global standard deviation of voiced :math:`f_0` values on voiced frames.
    
    Algorithm
    ---------
    The function computes the population standard deviation over voiced frames,
    
    .. math::
    
       \sigma_{f_0} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(f_i-\bar f)^2},
    
    and writes the same scalar back to each voiced frame position.
    
    Examples
    --------
        variability = compute_f0_variability(f0_values)
        print(np.nanmean(variability))
    
    """
    f0_values = np.asarray(f0_values, dtype=np.float32)
    variability = np.full(f0_values.shape, np.nan, dtype=np.float32)
    voiced_mask = np.isfinite(f0_values) & (f0_values > 0.0)

    if not np.any(voiced_mask):
        return variability

    variability[voiced_mask] = np.float32(np.std(f0_values[voiced_mask]))
    return variability


def compute_f0_range(f0_values):
    r"""
    Compute voiced pitch range and broadcast it to voiced frames.
    
    Parameters
    ----------
    f0_values : array-like
        Frame-level pitch contour with shape ``(n_frames,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with shape ``(n_frames,)`` containing the global voiced pitch range on voiced frames.
    
    Algorithm
    ---------
    The range is computed as
    
    .. math::
    
       \mathrm{Range}(f_0) = \max_i f_i - \min_i f_i,
    
    using only voiced frames.
    
    Examples
    --------
        pitch_range = compute_f0_range(f0_values)
        print(np.nanmax(pitch_range))
    
    """
    f0_values = np.asarray(f0_values, dtype=np.float32)
    f0_range = np.full(f0_values.shape, np.nan, dtype=np.float32)
    voiced_mask = np.isfinite(f0_values) & (f0_values > 0.0)

    if not np.any(voiced_mask):
        return f0_range

    range_value = np.max(f0_values[voiced_mask]) - np.min(f0_values[voiced_mask])
    f0_range[voiced_mask] = np.float32(range_value)
    return f0_range


def compute_contour_shape(f0_values):
    r"""
    Discretize pitch movement into rising, level, and falling states.
    
    Parameters
    ----------
    f0_values : array-like
        Frame-level pitch contour with shape ``(n_frames,)``.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with shape ``(n_frames,)`` whose values are ``1`` for rising, ``0`` for level, ``-1`` for falling, and ``NaN`` for undefined frames.
    
    Algorithm
    ---------
    The pitch derivative is thresholded into three states:
    
    .. math::
    
       c_t = \begin{cases}
           1 & \Delta f_0[t] > \theta, \\
           0 & |\Delta f_0[t]| \le \theta, \\
          -1 & \Delta f_0[t] < -\theta.
       \end{cases}
    
    The implementation uses a fixed derivative threshold and leaves missing derivatives undefined.
    
    Examples
    --------
        contour = compute_contour_shape(f0_values)
        print(contour[:10])
    
    """
    derivatives = compute_f0_derivative(f0_values)
    contour = np.full(np.asarray(f0_values).shape, np.nan, dtype=np.float32)

    for index, value in enumerate(derivatives):
        if not np.isfinite(value):
            continue
        if value > 0.5:
            contour[index] = 1.0
        elif value < -0.5:
            contour[index] = -1.0
        else:
            contour[index] = 0.0

    return contour
