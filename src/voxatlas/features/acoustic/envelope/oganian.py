import numpy as np

from voxatlas.acoustic.envelope_utils import compute_hilbert, frame_signal, smooth_signal
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class OganianEnvelope(BaseExtractor):
    r"""
    Extract the ``acoustic.envelope.oganian`` feature within the VoxAtlas pipeline.

    Computes a smoothed, frame-aligned amplitude envelope derived from the
    Hilbert analytic signal. This envelope is the precursor contour used by
    Oganian & Chang (2019) to define speech envelope landmarks, and is intended
    as a base contour for downstream onset/peak-rate style features.

    Algorithm
    ---------
    The implementation mirrors the code path.

    1. Analytic-signal envelope
       Given waveform :math:`x[n]`, the extractor forms

       .. math::

          z[n] = x[n] + j\,\mathcal{H}\{x[n]\},

       where :math:`\mathcal{H}` is the Hilbert transform, and computes the
       magnitude envelope :math:`a[n] = |z[n]|`.

    2. Frame pooling
       :math:`a[n]` is segmented into overlapping analysis frames and converted
       into a frame-level contour by taking the mean amplitude per frame.

    3. Smoothing
       The resulting frame contour is smoothed with a short moving-average
       window of length ``smoothing`` frames.

    Attributes
    ----------
    name : str
        Registry key for this extractor (``"acoustic.envelope.oganian"``).
    input_units : str | None
        Required input unit level. ``None`` means this extractor operates
        directly on waveform audio.
    output_units : str | None
        Output alignment unit (``"frame"``).
    dependencies : list[str]
        Upstream features required before execution. Empty for this extractor.
    default_config : dict
        Default runtime parameters:
        ``frame_length=0.025``, ``frame_step=0.01``,
        ``peak_threshold=0.1``, ``smoothing=7``.

    References
    ----------
    Oganian, Y., & Chang, E. F. (2019). A speech envelope landmark for syllable
        encoding in human superior temporal gyrus. *Science Advances, 5*(11),
        eaay6279. https://doi.org/10.1126/sciadv.aay6279

    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.audio.audio import Audio
    >>> from voxatlas.features.acoustic.envelope.oganian import OganianEnvelope
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
    >>> feature_input = FeatureInput(audio=audio, units=None, context={})
    >>> params = OganianEnvelope.default_config.copy()
    >>> out = OganianEnvelope().compute(feature_input, params)
    >>> out.unit
    'frame'
    """
    name: str = "acoustic.envelope.oganian"
    input_units: str | None = None
    output_units: str | None = "frame"
    dependencies: list[str] = []
    default_config: dict = {
        "frame_length": 0.025,
        "frame_step": 0.010,
        "smoothing": 7,
        "peak_threshold": 0.1,
    }

    def compute(self, feature_input, params):
        """
        Compute the extractor output for a single pipeline invocation.
        
        This method is the reusable execution entry point for the extractor. It receives the standard ``FeatureInput`` bundle, applies the configured algorithm, and returns feature values aligned to the extractor output units for storage in the pipeline feature store.
        
        Parameters
        ----------
        feature_input : object
            Structured extractor input bundling audio, hierarchical units, and execution context for this feature computation.
        params : object
            Resolved feature configuration for this invocation. Keys are feature-specific and merged from defaults and pipeline settings.
        
        Returns
        -------
        FeatureOutput
            Structured output aligned to the ``frame`` unit level when applicable.
        
        Examples
        --------
        >>> import numpy as np
        >>> from voxatlas.audio.audio import Audio
        >>> from voxatlas.features.acoustic.envelope.oganian import OganianEnvelope
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
        >>> feature_input = FeatureInput(audio=audio, units=None, context={})
        >>> params = OganianEnvelope.default_config.copy()
        >>> result = OganianEnvelope().compute(feature_input, params)
        >>> result.values.shape[0] > 0
        True
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        sample_envelope = compute_hilbert(feature_input.audio.waveform)
        frames, time = frame_signal(
            sample_envelope,
            feature_input.audio.sample_rate,
            params["frame_length"],
            params["frame_step"],
        )
        values = np.mean(frames, axis=1).astype(np.float32)
        values = smooth_signal(values, params.get("smoothing", 7))

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(OganianEnvelope)
