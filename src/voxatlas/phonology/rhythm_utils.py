from __future__ import annotations

import numpy as np
import pandas as pd


def _midpoint(row) -> float:
    return (float(row["start"]) + float(row["end"])) / 2.0


def _assign_ipu_ids(table: pd.DataFrame, ipus: pd.DataFrame) -> pd.Series:
    if table is None or ipus is None:
        return pd.Series(dtype="Int64")

    values = []
    for _, row in table.iterrows():
        midpoint = _midpoint(row)
        match_id = pd.NA

        for _, ipu in ipus.iterrows():
            if float(ipu["start"]) <= midpoint <= float(ipu["end"]):
                match_id = ipu["id"]
                break

        values.append(match_id)

    return pd.Series(values, index=table.index, dtype="Int64")


def compute_rhythm_intervals(
    phonemes: pd.DataFrame,
    vowel_flags: pd.Series,
    ipus: pd.DataFrame,
) -> pd.DataFrame:
    r"""
    Construct vowel and consonant rhythm intervals within IPUs.
    
    This function is the basis for all VoxAtlas rhythm metrics. It converts phoneme-level segment annotations into alternating vocalic and consonantal interval spans.
    
    Parameters
    ----------
    phonemes : pandas.DataFrame
        Phoneme table. Expected columns include ``id``, ``start``, ``end``, and ``label``.
    vowel_flags : pandas.Series
        Boolean or numeric series indexed by phoneme identifier, where vowel phonemes are marked by ``1``.
    ipus : pandas.DataFrame
        IPU table with at least ``id``, ``start``, and ``end`` columns.
    
    phonemes example
    ----------------
    phoneme_id | start | end | label | word_id
    0 | 0.12 | 0.18 | h | 0
    1 | 0.18 | 0.25 | eh | 0
    
    Returns
    -------
    pandas.DataFrame
        Interval table with columns ``interval_id``, ``ipu_id``, ``type``, ``start``, ``end``, ``duration``, and ``n_phonemes``.
    
    Algorithm
    ---------
    1. Phoneme-to-IPU assignment
       Each phoneme is assigned to the IPU containing its temporal midpoint.
    
    2. Class collapse
       Consecutive phonemes with the same class are merged into one interval of type ``v`` or ``c``.
    
    3. Duration computation
       Each interval duration is
    
       .. math::
    
          d = t^{end} - t^{start}.
    
    These interval durations are then reused by all subsequent rhythm metrics.
    
    Examples
    --------
        intervals = compute_rhythm_intervals(phonemes, vowel_flags, ipus)
        print(intervals.head())
    
    """
    if phonemes is None or len(phonemes) == 0 or ipus is None or len(ipus) == 0:
        return pd.DataFrame(
            columns=[
                "interval_id",
                "ipu_id",
                "type",
                "start",
                "end",
                "duration",
                "n_phonemes",
            ]
        )

    phoneme_table = phonemes.copy().reset_index(drop=True)
    vowel_map = vowel_flags.to_dict()
    phoneme_table["is_vowel"] = [
        vowel_map.get(phoneme_id, np.nan)
        for phoneme_id in phoneme_table["id"]
    ]
    phoneme_table["ipu_id"] = _assign_ipu_ids(phoneme_table, ipus)
    phoneme_table = phoneme_table.dropna(subset=["ipu_id"]).reset_index(drop=True)

    rows = []
    interval_id = 1

    for ipu_id, group in phoneme_table.groupby("ipu_id", sort=False):
        group = group.sort_values("start").reset_index(drop=True)
        current = None

        for _, phoneme in group.iterrows():
            is_vowel = phoneme["is_vowel"]
            if not np.isfinite(is_vowel):
                continue

            interval_type = "v" if float(is_vowel) == 1.0 else "c"
            start = float(phoneme["start"])
            end = float(phoneme["end"])

            if current is None or current["type"] != interval_type:
                if current is not None:
                    rows.append(current)
                    interval_id += 1

                current = {
                    "interval_id": interval_id,
                    "ipu_id": int(ipu_id),
                    "type": interval_type,
                    "start": np.float32(start),
                    "end": np.float32(end),
                    "duration": np.float32(end - start),
                    "n_phonemes": 1,
                }
            else:
                current["end"] = np.float32(end)
                current["duration"] = np.float32(end - float(current["start"]))
                current["n_phonemes"] += 1

        if current is not None:
            rows.append(current)
            interval_id += 1

    return pd.DataFrame(rows)


def compute_syllable_durations(syllables: pd.DataFrame) -> pd.Series:
    r"""
    Compute syllable durations from aligned syllable spans.
    
    Parameters
    ----------
    syllables : pandas.DataFrame
        Syllable table with ``start`` and ``end`` columns.
    
    Returns
    -------
    pandas.Series
        Float32 duration series indexed like ``syllables``.
    
    Algorithm
    ---------
    For each syllable :math:`i`, the duration is
    
    .. math::
    
       d_i = t_i^{end} - t_i^{start}.
    
    Examples
    --------
        durations = compute_syllable_durations(syllables)
        print(durations.head())
    
    """
    if syllables is None:
        return pd.Series(dtype="float32")
    return (syllables["end"] - syllables["start"]).astype("float32")


def compute_syllable_rate(
    syllables: pd.DataFrame,
    ipus: pd.DataFrame,
) -> pd.Series:
    r"""
    Compute syllable rate per IPU.
    
    Parameters
    ----------
    syllables : pandas.DataFrame
        Syllable table.
    ipus : pandas.DataFrame
        IPU table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level syllable-rate series.
    
    Algorithm
    ---------
    After assigning each syllable to an IPU, the rate for IPU :math:`j` is
    
    .. math::
    
       r_j = \frac{N_j^{\mathrm{syll}}}{T_j},
    
    where :math:`T_j` is IPU duration.
    
    Examples
    --------
        rate = compute_syllable_rate(syllables, ipus)
        print(rate.head())
    
    """
    if syllables is None or ipus is None:
        return pd.Series(dtype="float32")

    syllable_ipus = _assign_ipu_ids(syllables, ipus)
    rates = []
    index = []

    for _, ipu in ipus.iterrows():
        ipu_id = int(ipu["id"])
        ipu_duration = max(float(ipu["end"]) - float(ipu["start"]), 1e-8)
        count = int((syllable_ipus == ipu_id).sum())
        index.append(ipu_id)
        rates.append(np.float32(count / ipu_duration))

    return pd.Series(rates, index=index, dtype="float32")


def compute_pause_rate(
    syllables: pd.DataFrame,
    ipus: pd.DataFrame,
    pause_threshold: float = 0.05,
) -> pd.Series:
    r"""
    Compute pause rate per IPU from inter-syllabic gaps.
    
    Parameters
    ----------
    syllables : pandas.DataFrame
        Syllable table.
    ipus : pandas.DataFrame
        IPU table.
    pause_threshold : float, default=0.05
        Minimum gap duration in seconds that counts as a pause.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level pause-rate series.
    
    Algorithm
    ---------
    Within each IPU, gaps :math:`g_i = t_{i+1}^{start} - t_i^{end}` are counted whenever :math:`g_i > \theta`. The rate is then
    
    .. math::
    
       p_j = \frac{N_j^{\mathrm{pause}}}{T_j}.
    
    Examples
    --------
        pause_rate = compute_pause_rate(syllables, ipus, pause_threshold=0.05)
        print(pause_rate.head())
    
    """
    if syllables is None or ipus is None:
        return pd.Series(dtype="float32")

    syllable_table = syllables.copy().reset_index(drop=True)
    syllable_table["ipu_id"] = _assign_ipu_ids(syllable_table, ipus)
    rates = []
    index = []

    for _, ipu in ipus.iterrows():
        ipu_id = int(ipu["id"])
        group = syllable_table.loc[syllable_table["ipu_id"] == ipu_id].sort_values("start")
        pause_count = 0

        for current, nxt in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            gap = float(nxt.start) - float(current.end)
            if gap > pause_threshold:
                pause_count += 1

        ipu_duration = max(float(ipu["end"]) - float(ipu["start"]), 1e-8)
        index.append(ipu_id)
        rates.append(np.float32(pause_count / ipu_duration))

    return pd.Series(rates, index=index, dtype="float32")


def _interval_durations(intervals: pd.DataFrame, interval_type: str) -> list[np.ndarray]:
    if intervals is None or intervals.empty:
        return []

    duration_sets = []
    for _, group in intervals.groupby("ipu_id", sort=False):
        values = group.loc[group["type"] == interval_type, "duration"].to_numpy(dtype=np.float32)
        duration_sets.append(values)
    return duration_sets


def _metric_by_ipu(
    intervals: pd.DataFrame,
    reducer,
) -> pd.Series:
    if intervals is None or intervals.empty:
        return pd.Series(dtype="float32")

    values = []
    index = []
    for ipu_id, group in intervals.groupby("ipu_id", sort=False):
        index.append(int(ipu_id))
        values.append(np.float32(reducer(group)))
    return pd.Series(values, index=index, dtype="float32")


def _npvi(durations: np.ndarray) -> float:
    durations = np.asarray(durations, dtype=np.float32)
    if durations.size < 2:
        return np.nan

    pair_means = (durations[1:] + durations[:-1]) / 2.0
    valid = pair_means > 0.0
    if not np.any(valid):
        return np.nan

    normalized = np.abs(durations[1:] - durations[:-1])[valid] / pair_means[valid]
    return float(100.0 * np.mean(normalized))


def compute_npvi(intervals: pd.DataFrame) -> pd.Series:
    r"""
    Compute the normalized pairwise variability index of vowel intervals.
    
    Parameters
    ----------
    intervals : pandas.DataFrame
        Rhythm interval table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level nPVI series.
    
    Algorithm
    ---------
    For the vowel-interval durations in an IPU, VoxAtlas computes the mean
    absolute difference between adjacent durations and normalizes each
    difference by the local pair mean before averaging.
    
    Examples
    --------
        npvi = compute_npvi(intervals)
        print(npvi.head())
    
    """
    return _metric_by_ipu(
        intervals,
        lambda group: _npvi(group.loc[group["type"] == "v", "duration"].to_numpy(dtype=np.float32)),
    )


def compute_percent_v(intervals: pd.DataFrame) -> pd.Series:
    r"""
    Compute percentage of vocalic interval time per IPU.
    
    Parameters
    ----------
    intervals : pandas.DataFrame
        Rhythm interval table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level ``%V`` series.
    
    Algorithm
    ---------
    Vocalic proportion is
    
    .. math::
    
       \%V = 100\frac{\sum_i d_i^{(v)}}{\sum_i d_i^{(all)}}.
    
    Examples
    --------
        percent_v = compute_percent_v(intervals)
        print(percent_v.head())
    
    """
    return _metric_by_ipu(
        intervals,
        lambda group: (
            100.0
            * float(group.loc[group["type"] == "v", "duration"].sum())
            / max(float(group["duration"].sum()), 1e-8)
        ),
    )


def compute_percent_c(intervals: pd.DataFrame) -> pd.Series:
    r"""
    Compute percentage of consonantal interval time per IPU.
    
    Parameters
    ----------
    intervals : pandas.DataFrame
        Rhythm interval table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level ``%C`` series.
    
    Algorithm
    ---------
    Consonantal proportion is
    
    .. math::
    
       \%C = 100\frac{\sum_i d_i^{(c)}}{\sum_i d_i^{(all)}}.
    
    Examples
    --------
        percent_c = compute_percent_c(intervals)
        print(percent_c.head())
    
    """
    return _metric_by_ipu(
        intervals,
        lambda group: (
            100.0
            * float(group.loc[group["type"] == "c", "duration"].sum())
            / max(float(group["duration"].sum()), 1e-8)
        ),
    )


def _delta(group: pd.DataFrame, interval_type: str) -> float:
    values = group.loc[group["type"] == interval_type, "duration"].to_numpy(dtype=np.float32)
    if values.size == 0:
        return np.nan
    return float(np.std(values, ddof=0))


def compute_delta_v(intervals: pd.DataFrame) -> pd.Series:
    r"""
    Compute standard deviation of vowel-interval durations per IPU.
    
    Parameters
    ----------
    intervals : pandas.DataFrame
        Rhythm interval table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level ``Delta V`` series.
    
    Algorithm
    ---------
    The metric is the population standard deviation of vowel durations,
    
    .. math::
    
       \Delta V = \mathrm{sd}(d^{(v)}).
    
    Examples
    --------
        delta_v = compute_delta_v(intervals)
        print(delta_v.head())
    
    """
    return _metric_by_ipu(intervals, lambda group: _delta(group, "v"))


def compute_delta_c(intervals: pd.DataFrame) -> pd.Series:
    r"""
    Compute standard deviation of consonant-interval durations per IPU.
    
    Parameters
    ----------
    intervals : pandas.DataFrame
        Rhythm interval table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level ``Delta C`` series.
    
    Algorithm
    ---------
    The metric is the population standard deviation of consonant durations,
    
    .. math::
    
       \Delta C = \mathrm{sd}(d^{(c)}).
    
    Examples
    --------
        delta_c = compute_delta_c(intervals)
        print(delta_c.head())
    
    """
    return _metric_by_ipu(intervals, lambda group: _delta(group, "c"))


def _varco(group: pd.DataFrame, interval_type: str) -> float:
    values = group.loc[group["type"] == interval_type, "duration"].to_numpy(dtype=np.float32)
    if values.size == 0:
        return np.nan
    mean_value = float(np.mean(values))
    if mean_value <= 0.0:
        return np.nan
    return float(100.0 * np.std(values, ddof=0) / mean_value)


def compute_varco_v(intervals: pd.DataFrame) -> pd.Series:
    r"""
    Compute normalized variability of vowel-interval durations per IPU.
    
    Parameters
    ----------
    intervals : pandas.DataFrame
        Rhythm interval table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level VarcoV series.
    
    Algorithm
    ---------
    VarcoV is
    
    .. math::
    
       \mathrm{VarcoV} = 100\frac{\mathrm{sd}(d^{(v)})}{\overline{d^{(v)}}}.
    
    Examples
    --------
        varco_v = compute_varco_v(intervals)
        print(varco_v.head())
    
    """
    return _metric_by_ipu(intervals, lambda group: _varco(group, "v"))


def compute_varco_c(intervals: pd.DataFrame) -> pd.Series:
    r"""
    Compute normalized variability of consonant-interval durations per IPU.
    
    Parameters
    ----------
    intervals : pandas.DataFrame
        Rhythm interval table.
    
    Returns
    -------
    pandas.Series
        Float32 IPU-level VarcoC series.
    
    Algorithm
    ---------
    VarcoC is
    
    .. math::
    
       \mathrm{VarcoC} = 100\frac{\mathrm{sd}(d^{(c)})}{\overline{d^{(c)}}}.
    
    Examples
    --------
        varco_c = compute_varco_c(intervals)
        print(varco_c.head())
    
    """
    return _metric_by_ipu(intervals, lambda group: _varco(group, "c"))
