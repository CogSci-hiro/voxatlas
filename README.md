# VoxAtlas

VoxAtlas is a modular Python toolkit for feature extraction and analysis workflows.

## Documentation

The project includes a Sphinx documentation setup styled after MNE-Python and NumPy conventions:

- NumPy-style docstrings
- API documentation generated from Python docstrings
- autosummary-based API tables
- `pydata-sphinx-theme`
- `numpydoc` formatting

Build the documentation from the repository root:

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

The generated site is written to:

```text
docs/_build/html/index.html
```

## Docstring Guidelines

Public API docstrings should follow NumPy style and include:

- `Parameters`
- `Returns`
- `Notes` when useful
- `Usage example` for public API functions, classes, and methods

Example:

```python
def compute_trf(signal: np.ndarray, stimulus: np.ndarray) -> np.ndarray:
    """
    Compute temporal response function.

    Parameters
    ----------
    signal : ndarray
        Neural signal time series.
    stimulus : ndarray
        Stimulus feature matrix.

    Returns
    -------
    ndarray
        TRF kernel.

    Notes
    -----
    Used for speech-brain alignment analyses.

    Usage example
    -------------
        kernel = compute_trf(signal, stimulus)
    """
```
