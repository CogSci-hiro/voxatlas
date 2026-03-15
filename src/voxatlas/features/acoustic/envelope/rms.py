import numpy as np

from voxatlas.acoustic.envelope_utils import compute_rms, smooth_signal
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class RMSEnvelope(BaseExtractor):
    r"""
    Extract the ``acoustic.envelope.rms`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.envelope.rms`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor computes a frame-level root-mean-square amplitude envelope from the waveform.
    
    1. Frame extraction
       The signal is segmented into overlapping windows determined by ``frame_length`` and ``frame_step``.
    
    2. Energy accumulation
       For each frame :math:`x_t[n]`, the envelope value is
    
       .. math::
    
          \mathrm{RMS}_t = \sqrt{\frac{1}{N}\sum_{n=1}^{N} x_t[n]^2}.
    
    3. Output alignment
       RMS values are returned with frame midpoints so that derived features such as derivatives, onsets, and peak rate can reuse the same temporal grid.
    
    Examples
    --------
        from voxatlas.features.acoustic.envelope.rms import RMSEnvelope
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = RMSEnvelope()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.envelope.rms"
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
            extractor = RMSEnvelope()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        time, values = compute_rms(
            feature_input.audio.waveform,
            feature_input.audio.sample_rate,
            params["frame_length"],
            params["frame_step"],
        )
        values = smooth_signal(values, params.get("smoothing", 1))

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(RMSEnvelope)
