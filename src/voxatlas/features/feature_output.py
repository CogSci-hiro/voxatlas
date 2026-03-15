from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScalarFeatureOutput:
    """
    Store one scalar value per aligned unit.

    Parameters
    ----------
    feature : str
        Registered feature name.
    unit : str
        Unit level associated with the values.
    values : pandas.Series
        Scalar values indexed by unit identifier.

    Returns
    -------
    ScalarFeatureOutput
        Dataclass wrapping scalar outputs.

    Notes
    -----
    Scalar outputs are typically used for token-, sentence-, or conversation-
    level summary values.

    Examples
    --------
    Usage example::

        output = ScalarFeatureOutput(feature="lexical.frequency.word", unit="token", values=series)
        print(output.values.head())
    """

    feature: str
    unit: str
    values: pd.Series


@dataclass
class VectorFeatureOutput:
    """
    Store a time-aligned one-dimensional feature sequence.

    Parameters
    ----------
    feature : str
        Registered feature name.
    unit : str
        Unit level associated with the vector.
    time : ndarray
        Time axis for the vector.
    values : ndarray
        Vector values aligned to ``time``.

    Returns
    -------
    VectorFeatureOutput
        Dataclass wrapping frame-aligned outputs.

    Notes
    -----
    Acoustic envelope and pitch features commonly use this output type.

    Examples
    --------
    Usage example::

        output = VectorFeatureOutput(feature="acoustic.pitch.f0", unit="frame", time=time, values=values)
        print(output.values.shape)
    """

    feature: str
    unit: str
    time: np.ndarray
    values: np.ndarray


@dataclass
class MatrixFeatureOutput:
    """
    Store a time-frequency matrix output.

    Parameters
    ----------
    feature : str
        Registered feature name.
    unit : str
        Unit level associated with the matrix.
    time : ndarray
        Time axis of the matrix.
    frequency : ndarray
        Frequency axis of the matrix.
    values : ndarray
        Matrix values aligned to the time and frequency axes.

    Returns
    -------
    MatrixFeatureOutput
        Dataclass wrapping matrix-valued outputs.

    Notes
    -----
    Spectrogram-style features use this container.

    Examples
    --------
    Usage example::

        output = MatrixFeatureOutput(feature="acoustic.spectrogram.stft", unit="frame", time=time, frequency=freq, values=matrix)
        print(output.values.shape)
    """

    feature: str
    unit: str
    time: np.ndarray
    frequency: np.ndarray
    values: np.ndarray


@dataclass
class TableFeatureOutput:
    """
    Store a tabular feature representation.

    Parameters
    ----------
    feature : str
        Registered feature name.
    unit : str
        Unit level represented by the rows.
    values : pandas.DataFrame
        Tabular feature values.

    Returns
    -------
    TableFeatureOutput
        Dataclass wrapping DataFrame outputs.

    Notes
    -----
    Lookup tables and dependency annotations commonly use this output type.

    Examples
    --------
    Usage example::

        output = TableFeatureOutput(feature="syntax.dependencies", unit="token", values=table)
        print(output.values.columns)
    """

    feature: str
    unit: str
    values: pd.DataFrame


@dataclass
class ArrayFeatureOutput:
    """
    Store an unlabeled NumPy array result.

    Parameters
    ----------
    feature : str
        Registered feature name.
    values : ndarray
        Raw NumPy array produced by the feature.

    Returns
    -------
    ArrayFeatureOutput
        Dataclass wrapping an unlabeled array output.

    Notes
    -----
    This container is useful for results that do not naturally fit the scalar,
    vector, matrix, or table forms.

    Examples
    --------
    Usage example::

        output = ArrayFeatureOutput(feature="custom.array", values=array)
        print(output.values.shape)
    """

    feature: str
    values: np.ndarray
