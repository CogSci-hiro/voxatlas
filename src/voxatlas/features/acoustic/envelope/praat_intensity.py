import numpy as np

from voxatlas.acoustic.envelope_utils import compute_rms, smooth_signal
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class PraatIntensityEnvelope(BaseExtractor):
    r"""
    Extract the ``acoustic.envelope.praat_intensity`` feature within the VoxAtlas pipeline.

    Computes a smoothed, frame-aligned intensity-like contour intended to serve
    as a lightweight proxy for Praat-style intensity tracking.

    Algorithm
    ---------
    The implementation mirrors the code path.

    1. RMS amplitude
       The waveform is framed and converted to RMS values :math:`r_t`.

    2. Smoothing
       The RMS contour is smoothed with a moving-average window of length
       ``smoothing`` frames.

    Notes
    -----
    This extractor does not attempt to reproduce Praat's full intensity
    computation (e.g., frequency weighting and dB scaling). It provides a
    consistent, frame-aligned contour suitable for downstream transforms.

    Attributes
    ----------
    name : str
        Registry key for this extractor (``"acoustic.envelope.praat_intensity"``).
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
        ``peak_threshold=0.1``, ``smoothing=5``.

    References
    ----------
    Boersma, P. (2001). Praat, a system for doing phonetics by computer.
    *Glot International, 5*(9/10), 341–345.

    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.audio.audio import Audio
    >>> from voxatlas.features.acoustic.envelope.praat_intensity import PraatIntensityEnvelope
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
    >>> feature_input = FeatureInput(audio=audio, units=None, context={})
    >>> params = PraatIntensityEnvelope.default_config.copy()
    >>> out = PraatIntensityEnvelope().compute(feature_input, params)
    >>> out.unit
    'frame'
    """
    name: str = "acoustic.envelope.praat_intensity"
    input_units: str | None = None
    output_units: str | None = "frame"
    dependencies: list[str] = []
    default_config: dict = {
        "frame_length": 0.025,
        "frame_step": 0.010,
        "smoothing": 5,
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
        >>> from voxatlas.features.acoustic.envelope.praat_intensity import PraatIntensityEnvelope
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
        >>> feature_input = FeatureInput(audio=audio, units=None, context={})
        >>> params = PraatIntensityEnvelope.default_config.copy()
        >>> result = PraatIntensityEnvelope().compute(feature_input, params)
        >>> result.values.shape[0] > 0
        True
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        time, values = compute_rms(
            feature_input.audio.waveform,
            feature_input.audio.sample_rate,
            params["frame_length"],
            params["frame_step"],
        )
        values = smooth_signal(values, params.get("smoothing", 5))

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(PraatIntensityEnvelope)
