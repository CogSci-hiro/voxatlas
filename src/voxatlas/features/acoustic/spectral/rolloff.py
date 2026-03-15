import numpy as np

from voxatlas.acoustic.spectral_utils import spectral_rolloff
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class SpectralRolloffExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectral.rolloff`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectral.rolloff`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor finds the frequency below which a fixed proportion of cumulative spectral magnitude is contained.
    
    1. Cumulative energy
       For each frame, cumulative sums over frequency bins are computed.
    
    2. Threshold crossing
       The roll-off frequency :math:`f_r` is the smallest :math:`f_k` satisfying
    
       .. math::
    
          \sum_{j \le k} S_{t,j} \ge \rho \sum_j S_{t,j},
    
       where :math:`\rho` is the configured roll-off proportion.
    
    3. Packaging
       The selected frequency is returned as a frame-level scalar contour.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.spectral.spectrum'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.acoustic.spectral.rolloff import SpectralRolloffExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = SpectralRolloffExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.spectral.rolloff"
    input_units = None
    output_units = "frame"
    dependencies = ["acoustic.spectral.spectrum"]
    default_config = {
        "roll_percent": 0.85,
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
            extractor = SpectralRolloffExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        spectrum_output = feature_input.context["feature_store"].get(
            "acoustic.spectral.spectrum"
        )
        values = spectral_rolloff(
            spectrum_output.values,
            spectrum_output.frequency,
            roll_percent=params.get("roll_percent", 0.85),
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(spectrum_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(SpectralRolloffExtractor)
