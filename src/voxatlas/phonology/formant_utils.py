from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from voxatlas.phonology.articulatory_utils import (
    load_phonology_resources,
    lookup_articulatory_features,
)


FORMANT_COLUMNS = ["F1", "F2", "F3"]


def _to_samples(value, sr):
    if isinstance(value, float):
        return max(1, int(round(value * sr)))
    return max(1, int(value))


def _safe_float32(value):
    if value is None or not np.isfinite(value):
        return np.float32(np.nan)
    return np.float32(value)


def _pre_emphasize(signal, coeff=0.97):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)


def _autocorrelation(signal, order):
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[autocorr.size // 2:]
    if autocorr.size <= order:
        return None
    return autocorr[:order + 1]


def _lpc_coefficients(signal, order):
    autocorr = _autocorrelation(signal, order)
    if autocorr is None or autocorr[0] <= 0.0:
        return None

    matrix = np.empty((order, order), dtype=np.float64)
    rhs = autocorr[1:order + 1].astype(np.float64)

    for row in range(order):
        for col in range(order):
            matrix[row, col] = autocorr[abs(row - col)]

    try:
        coeffs = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None

    return np.concatenate(([1.0], -coeffs))


def _estimate_formants_lpc(
    frame,
    sr,
    lpc_order=10,
    max_formant=5500.0,
    min_formant=90.0,
    n_formants=3,
):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.size < max(lpc_order + 1, 16):
        return [np.float32(np.nan)] * n_formants

    windowed = _pre_emphasize(frame - np.mean(frame))
    windowed = windowed * np.hamming(windowed.size)

    coeffs = _lpc_coefficients(windowed, lpc_order)
    if coeffs is None:
        return [np.float32(np.nan)] * n_formants

    roots = np.roots(coeffs)
    roots = roots[np.imag(roots) >= 0.0]

    angles = np.angle(roots)
    freqs = angles * (sr / (2.0 * np.pi))
    bandwidths = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.maximum(np.abs(roots), 1e-8))

    valid = []
    for freq, bandwidth in zip(freqs, bandwidths):
        if min_formant <= freq <= max_formant and 0.0 < bandwidth < 700.0:
            valid.append(float(freq))

    valid = sorted(valid)[:n_formants]
    while len(valid) < n_formants:
        valid.append(np.nan)

    return [np.float32(value) for value in valid]


def _compute_segment_tracks_lpc(
    segment,
    sr,
    start_time,
    phoneme_id,
    label,
    ipa,
    is_vowel,
    frame_length,
    frame_step,
    lpc_order,
    max_formant,
):
    frame_length_samples = _to_samples(frame_length, sr)
    frame_step_samples = _to_samples(frame_step, sr)
    frame_centers = np.arange(0, max(1, len(segment) - frame_length_samples + 1), frame_step_samples)

    if segment.size < frame_length_samples:
        padded = np.pad(segment, (0, frame_length_samples - segment.size))
        frame_centers = np.array([0], dtype=int)
        frames = [padded]
    else:
        frames = [
            segment[start:start + frame_length_samples]
            for start in frame_centers
        ]

    rows = []
    for index, (relative_start, frame) in enumerate(zip(frame_centers, frames), start=1):
        frame_start = start_time + (relative_start / float(sr))
        frame_mid = frame_start + (len(frame) / float(sr) / 2.0)

        if is_vowel:
            f1, f2, f3 = _estimate_formants_lpc(
                frame,
                sr,
                lpc_order=lpc_order,
                max_formant=max_formant,
            )
        else:
            f1 = f2 = f3 = np.float32(np.nan)

        rows.append(
            {
                "frame_id": index,
                "start": np.float32(frame_start),
                "end": np.float32(frame_start + len(frame) / float(sr)),
                "time": np.float32(frame_mid),
                "phoneme_id": phoneme_id,
                "label": label,
                "ipa": ipa,
                "is_vowel": np.float32(1.0 if is_vowel else 0.0),
                "F1": _safe_float32(f1),
                "F2": _safe_float32(f2),
                "F3": _safe_float32(f3),
            }
        )

    return rows


def _try_parselmouth_segment_tracks(
    segment,
    sr,
    start_time,
    phoneme_id,
    label,
    ipa,
    is_vowel,
    frame_length,
    frame_step,
    max_formant,
):
    try:
        import parselmouth
    except ImportError:
        return None

    if segment.size == 0:
        return []

    sound = parselmouth.Sound(np.asarray(segment, dtype=np.float64), sampling_frequency=sr)
    formant = sound.to_formant_burg(
        time_step=frame_step,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=frame_length,
        pre_emphasis_from=50.0,
    )

    duration = segment.size / float(sr)
    sample_times = np.arange(frame_length / 2.0, max(duration, frame_length / 2.0) + 1e-8, frame_step)
    if sample_times.size == 0:
        sample_times = np.array([min(duration / 2.0, frame_length / 2.0)], dtype=np.float32)

    rows = []
    for index, local_time in enumerate(sample_times, start=1):
        global_time = start_time + float(local_time)

        if is_vowel:
            values = []
            for formant_index in (1, 2, 3):
                value = formant.get_value_at_time(formant_index, local_time)
                values.append(_safe_float32(value if value and value > 0 else np.nan))
            f1, f2, f3 = values
        else:
            f1 = f2 = f3 = np.float32(np.nan)

        rows.append(
            {
                "frame_id": index,
                "start": np.float32(max(start_time, global_time - frame_length / 2.0)),
                "end": np.float32(min(start_time + duration, global_time + frame_length / 2.0)),
                "time": np.float32(global_time),
                "phoneme_id": phoneme_id,
                "label": label,
                "ipa": ipa,
                "is_vowel": np.float32(1.0 if is_vowel else 0.0),
                "F1": f1,
                "F2": f2,
                "F3": f3,
            }
        )

    return rows


def compute_formant_tracks(
    signal,
    sr,
    phonemes,
    language=None,
    resource_root=None,
    frame_length=0.025,
    frame_step=0.010,
    lpc_order=10,
    max_formant=5500.0,
    use_parselmouth=True,
):
    """
    Compute formant tracks from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    signal : object
        One-dimensional waveform or contour array with shape ``(n_samples,)``.
    sr : object
        Sampling rate of the waveform in Hertz.
    phonemes : object
        Tabular annotation aligned to a VoxAtlas unit level and used as structured pipeline input.
    language : object
        Argument used by the phonology API.
    resource_root : object
        Argument used by the phonology API.
    frame_length : object
        Analysis window duration in seconds or samples, depending on the helper.
    frame_step : object
        Hop size between successive analysis frames in seconds or samples, depending on the helper.
    lpc_order : object
        Argument used by the phonology API.
    max_formant : object
        Argument used by the phonology API.
    use_parselmouth : object
        Argument used by the phonology API.
    
    phonemes example
    ----------------
    phoneme_id | start | end | label | word_id
    0 | 0.12 | 0.18 | h | 0
    1 | 0.18 | 0.25 | eh | 0
    
    Returns
    -------
    pandas.DataFrame
        Tabular result aligned to a VoxAtlas unit level or registry resource.
    
    table example
    -------------
    unit_id | start | end | speaker | label
    0 | 0.12 | 0.45 | A | hello
    1 | 0.46 | 0.80 | A | world
    
    Examples
    --------
        value = compute_formant_tracks(signal=..., sr=..., phonemes=..., language=..., resource_root=..., frame_length=..., frame_step=..., lpc_order=..., max_formant=..., use_parselmouth=...)
        print(value)
    """
    if phonemes is None or len(phonemes) == 0:
        return pd.DataFrame(
            columns=["frame_id", "start", "end", "time", "phoneme_id", "label", "ipa", "is_vowel", *FORMANT_COLUMNS]
        )

    resources = load_phonology_resources(language=language, resource_root=resource_root)
    rows = []

    for _, phoneme in phonemes.iterrows():
        start = float(phoneme["start"])
        end = float(phoneme["end"])
        label = phoneme.get("label")
        ipa, features = lookup_articulatory_features(label, resources)
        is_vowel = bool(features is not None and np.isfinite(features.get("vowel", np.nan)) and float(features["vowel"]) == 1.0)

        start_sample = max(0, int(round(start * sr)))
        end_sample = min(len(signal), int(round(end * sr)))
        segment = np.asarray(signal[start_sample:end_sample], dtype=np.float32)

        segment_rows = None
        if use_parselmouth:
            segment_rows = _try_parselmouth_segment_tracks(
                segment=segment,
                sr=sr,
                start_time=start,
                phoneme_id=phoneme.get("id"),
                label=label,
                ipa=ipa,
                is_vowel=is_vowel,
                frame_length=frame_length,
                frame_step=frame_step,
                max_formant=max_formant,
            )

        if segment_rows is None:
            segment_rows = _compute_segment_tracks_lpc(
                segment=segment,
                sr=sr,
                start_time=start,
                phoneme_id=phoneme.get("id"),
                label=label,
                ipa=ipa,
                is_vowel=is_vowel,
                frame_length=frame_length,
                frame_step=frame_step,
                lpc_order=lpc_order,
                max_formant=max_formant,
            )

        rows.extend(segment_rows)

    if not rows:
        return pd.DataFrame(
            columns=["frame_id", "start", "end", "time", "phoneme_id", "label", "ipa", "is_vowel", *FORMANT_COLUMNS]
        )

    return pd.DataFrame(rows)


def _valid_vowel_tracks(tracks_df):
    if tracks_df is None or tracks_df.empty:
        return pd.DataFrame(columns=["phoneme_id", "label", "ipa", "time", *FORMANT_COLUMNS])

    df = tracks_df.copy()
    mask = df["is_vowel"].astype(float) == 1.0
    for column in FORMANT_COLUMNS:
        mask &= np.isfinite(df[column].astype(float))
    return df.loc[mask].reset_index(drop=True)


def _vowel_token_rows(tracks_df, reducer: Callable[[pd.DataFrame], dict[str, object]]):
    valid = _valid_vowel_tracks(tracks_df)
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for phoneme_id, group in valid.groupby("phoneme_id", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        base = {
            "phoneme_id": phoneme_id,
            "label": group.loc[0, "label"],
            "ipa": group.loc[0, "ipa"],
            "start": np.float32(group["start"].min()),
            "end": np.float32(group["end"].max()),
            "n_frames": int(len(group)),
        }
        base.update(reducer(group))
        rows.append(base)

    return pd.DataFrame(rows)


def _sample_row(group, proportion):
    index = int(round((len(group) - 1) * proportion))
    return group.iloc[index]


def compute_vowel_midpoint(tracks_df):
    """
    Compute vowel midpoint from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vowel_midpoint(tracks_df=...)
        print(value)
    """
    return _vowel_token_rows(
        tracks_df,
        lambda group: {
            column: np.float32(_sample_row(group, 0.5)[column])
            for column in FORMANT_COLUMNS
        },
    )


def compute_vowel_trajectory(tracks_df):
    """
    Compute vowel trajectory from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vowel_trajectory(tracks_df=...)
        print(value)
    """
    return _vowel_token_rows(
        tracks_df,
        lambda group: {
            f"{column}_{label}": np.float32(_sample_row(group, proportion)[column])
            for column in FORMANT_COLUMNS
            for label, proportion in (("20", 0.2), ("50", 0.5), ("80", 0.8))
        },
    )


def compute_vowel_mean(tracks_df):
    """
    Compute vowel mean from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vowel_mean(tracks_df=...)
        print(value)
    """
    return _vowel_token_rows(
        tracks_df,
        lambda group: {
            column: np.float32(group[column].mean())
            for column in FORMANT_COLUMNS
        },
    )


def compute_vowel_median(tracks_df):
    """
    Compute vowel median from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vowel_median(tracks_df=...)
        print(value)
    """
    return _vowel_token_rows(
        tracks_df,
        lambda group: {
            column: np.float32(group[column].median())
            for column in FORMANT_COLUMNS
        },
    )


def compute_vowel_variance(tracks_df):
    """
    Compute vowel variance from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vowel_variance(tracks_df=...)
        print(value)
    """
    return _vowel_token_rows(
        tracks_df,
        lambda group: {
            column: np.float32(group[column].var(ddof=0))
            for column in FORMANT_COLUMNS
        },
    )


def compute_onset_mid_offset(tracks_df):
    """
    Compute onset mid offset from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_onset_mid_offset(tracks_df=...)
        print(value)
    """
    return _vowel_token_rows(
        tracks_df,
        lambda group: {
            f"{column}_{label}": np.float32(_sample_row(group, proportion)[column])
            for column in FORMANT_COLUMNS
            for label, proportion in (("onset", 0.0), ("mid", 0.5), ("offset", 1.0))
        },
    )


def compute_polynomial_coefficients(tracks_df, degree=2):
    """
    Compute polynomial coefficients from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    degree : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_polynomial_coefficients(tracks_df=..., degree=...)
        print(value)
    """
    def reducer(group):
        x = np.linspace(-1.0, 1.0, len(group), dtype=np.float32)
        values = {}
        for column in FORMANT_COLUMNS:
            if len(group) <= degree:
                coeffs = [np.nan] * (degree + 1)
            else:
                coeffs = np.polyfit(x, group[column].to_numpy(dtype=np.float32), degree)
            for index, coefficient in enumerate(coeffs):
                values[f"{column}_c{index}"] = np.float32(coefficient)
        return values

    return _vowel_token_rows(tracks_df, reducer)


def _dct_coefficients(values, n_coeffs):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.full(n_coeffs, np.nan, dtype=np.float32)

    n = values.size
    indices = np.arange(n, dtype=np.float32)
    coeffs = []
    for k in range(n_coeffs):
        basis = np.cos((np.pi / n) * (indices + 0.5) * k)
        coeffs.append(np.float32(np.sum(values * basis)))
    return np.asarray(coeffs, dtype=np.float32)


def compute_dct_coefficients(tracks_df, n_coeffs=3):
    """
    Compute dct coefficients from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    n_coeffs : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_dct_coefficients(tracks_df=..., n_coeffs=...)
        print(value)
    """
    def reducer(group):
        values = {}
        for column in FORMANT_COLUMNS:
            coeffs = _dct_coefficients(group[column].to_numpy(dtype=np.float32), n_coeffs)
            for index, coefficient in enumerate(coeffs):
                values[f"{column}_dct{index}"] = np.float32(coefficient)
        return values

    return _vowel_token_rows(tracks_df, reducer)


def compute_vowel_slope(tracks_df):
    """
    Compute vowel slope from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vowel_slope(tracks_df=...)
        print(value)
    """
    def reducer(group):
        x = np.linspace(0.0, 1.0, len(group), dtype=np.float32)
        values = {}
        for column in FORMANT_COLUMNS:
            if len(group) < 2:
                values[f"{column}_slope"] = np.float32(np.nan)
                continue
            slope, _ = np.polyfit(x, group[column].to_numpy(dtype=np.float32), 1)
            values[f"{column}_slope"] = np.float32(slope)
        return values

    return _vowel_token_rows(tracks_df, reducer)


def _speaker_vowel_means(tracks_df):
    midpoint = compute_vowel_midpoint(tracks_df)
    if midpoint.empty:
        return {}

    means = {}
    for ipa, group in midpoint.groupby("ipa"):
        means[str(ipa)] = {
            "F1": float(group["F1"].mean()),
            "F2": float(group["F2"].mean()),
        }
    return means


def _triangle_area(points):
    (x1, y1), (x2, y2), (x3, y3) = points
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0


def compute_vsa(tracks_df):
    """
    Compute vsa from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vsa(tracks_df=...)
        print(value)
    """
    means = _speaker_vowel_means(tracks_df)
    required = {"i", "a", "u"}
    if not required.issubset(means):
        return np.float32(np.nan)

    points = [
        (means["i"]["F2"], means["i"]["F1"]),
        (means["a"]["F2"], means["a"]["F1"]),
        (means["u"]["F2"], means["u"]["F1"]),
    ]
    return np.float32(_triangle_area(points))


def _convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) / 2.0


def compute_tvsa(tracks_df):
    """
    Compute tvsa from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_tvsa(tracks_df=...)
        print(value)
    """
    midpoint = compute_vowel_midpoint(tracks_df)
    if midpoint.empty:
        return np.float32(np.nan)

    points = [
        (float(row["F2"]), float(row["F1"]))
        for _, row in midpoint.iterrows()
        if np.isfinite(row["F1"]) and np.isfinite(row["F2"])
    ]
    if len(points) < 3:
        return np.float32(np.nan)

    return np.float32(_polygon_area(_convex_hull(points)))


def compute_vai(tracks_df):
    """
    Compute vai from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_vai(tracks_df=...)
        print(value)
    """
    means = _speaker_vowel_means(tracks_df)
    required = {"i", "a", "u"}
    if not required.issubset(means):
        return np.float32(np.nan)

    numerator = means["i"]["F2"] + means["a"]["F1"]
    denominator = (
        means["i"]["F1"]
        + means["u"]["F1"]
        + means["u"]["F2"]
        + means["a"]["F2"]
    )
    if denominator <= 0.0:
        return np.float32(np.nan)
    return np.float32(numerator / denominator)


def compute_centralization(tracks_df):
    """
    Compute centralization from VoxAtlas inputs.
    
    This public function belongs to the phonology layer of VoxAtlas and can be reused by higher-level pipeline stages or feature extractors.
    
    Parameters
    ----------
    tracks_df : object
        Argument used by the phonology API.
    
    Returns
    -------
    object
        Computed value or structure produced from the supplied inputs.
    
    Examples
    --------
        value = compute_centralization(tracks_df=...)
        print(value)
    """
    means = _speaker_vowel_means(tracks_df)
    required = {"i", "a", "u"}
    if not required.issubset(means):
        return np.float32(np.nan)

    numerator = (
        means["u"]["F2"]
        + means["a"]["F2"]
        + means["i"]["F1"]
        + means["u"]["F1"]
    )
    denominator = means["i"]["F2"] + means["a"]["F1"]
    if denominator <= 0.0:
        return np.float32(np.nan)
    return np.float32(numerator / denominator)
