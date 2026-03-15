import numpy as np

from voxatlas.acoustic.envelope_utils import compute_hilbert, frame_signal, smooth_signal
from voxatlas.core.registry import register_feature
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput


@register_feature
class HilbertEnvelope(BaseExtractor):
    r"""
    Extract the ``acoustic.envelope.hilbert`` feature within the VoxAtlas pipeline.
    
    Computes a Hilbert-based amplitude envelope from the raw audio waveform.
    It does not require linguistic units as input and returns a frame-aligned
    time series.
    
    Algorithm
    ---------
    The extractor computes the analytic-signal envelope using the Hilbert transform.
    
    1. Analytic signal
       Given waveform :math:`x[n]`, the code forms
    
       .. math::
    
          z[n] = x[n] + j\,\mathcal{H}\{x[n]\},
    
       where :math:`\mathcal{H}` denotes the Hilbert transform.
    
    2. Magnitude envelope
       The returned contour is the magnitude :math:`a[n] = |z[n]|` and is then aligned to frame times as required by the output container.

    Attributes
    ----------
    name : str
        Registry key for this extractor (``"acoustic.envelope.hilbert"``).
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
        ``peak_threshold=0.1``, ``smoothing=1``.

    Examples
    --------
        from voxatlas.features.acoustic.envelope.hilbert import HilbertEnvelope
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = HilbertEnvelope()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.envelope.hilbert"
    input_units = None
    output_units = "frame"
    dependencies = []
    default_config = {
        "frame_length": 0.025,
        "frame_step": 0.010,
        "smoothing": 1,
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
            extractor = HilbertEnvelope()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
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
        values = smooth_signal(values, params.get("smoothing", 1))

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )
