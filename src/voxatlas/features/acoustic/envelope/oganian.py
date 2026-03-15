import numpy as np

from voxatlas.acoustic.envelope_utils import compute_hilbert, frame_signal, smooth_signal
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class OganianEnvelope(BaseExtractor):
    r"""
    Extract the ``acoustic.envelope.oganian`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.envelope.oganian`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor follows the standard VoxAtlas feature-computation pattern.
    
    1. Input preparation
       Structured audio, unit tables, and dependency outputs are gathered from ``feature_input``.
    
    2. Feature-specific computation
       The implementation applies the domain-specific transformation required by this extractor.
    
    3. Packaging
       Results are aligned to ``frame`` units and returned as a ``FeatureOutput`` object.
    
    Examples
    --------
        from voxatlas.features.acoustic.envelope.oganian import OganianEnvelope
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = OganianEnvelope()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.envelope.oganian"
    input_units = None
    output_units = "frame"
    dependencies = []
    default_config = {
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
            extractor = OganianEnvelope()
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
        values = smooth_signal(values, params.get("smoothing", 7))

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(OganianEnvelope)
