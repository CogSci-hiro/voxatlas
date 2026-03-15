import numpy as np

from voxatlas.acoustic.voice_quality_utils import compute_hnr
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class HNRExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.voice_quality.hnr`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.voice_quality.hnr`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor estimates harmonic-to-noise ratio from voiced waveform frames aligned with the :math:`f_0` contour.
    
    1. Frame partitioning
       The waveform is split into frame-like segments synchronized with the pitch track.
    
    2. Autocorrelation analysis
       For each voiced frame, the strongest non-zero-lag autocorrelation peak is treated as harmonic energy :math:`H`, and residual energy is treated as noise :math:`N`.
    
    3. Ratio in decibels
       The returned value is
    
       .. math::
    
          \mathrm{HNR}_t = 10\log_{10}\left(\frac{H}{N}\right).
    
    4. Packaging
       Invalid or unvoiced frames remain undefined to preserve methodological transparency.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.pitch.f0'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.acoustic.voice_quality.hnr import HNRExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = HNRExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.voice_quality.hnr"
    input_units = None
    output_units = "frame"
    dependencies = ["acoustic.pitch.f0"]
    default_config = {}

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
            extractor = HNRExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        f0_output = feature_input.context["feature_store"].get("acoustic.pitch.f0")
        values = compute_hnr(
            feature_input.audio.waveform,
            f0_output.values,
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(f0_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(HNRExtractor)
