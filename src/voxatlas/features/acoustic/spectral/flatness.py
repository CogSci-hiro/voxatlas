import numpy as np

from voxatlas.acoustic.spectral_utils import spectral_flatness
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class SpectralFlatnessExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectral.flatness`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectral.flatness`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor quantifies how noise-like each frame spectrum is by comparing geometric and arithmetic means.
    
    1. Flooring
       Spectral magnitudes are floored by a small :math:`\varepsilon` for numerical stability.
    
    2. Ratio computation
       Spectral flatness is
    
       .. math::
    
          F_t = \frac{\exp\left(\frac{1}{K}\sum_k \log S_{t,k}\right)}{\frac{1}{K}\sum_k S_{t,k}}.
    
    3. Packaging
       The resulting ratio is emitted as a frame-level contour.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.spectral.spectrum'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.acoustic.spectral.flatness import SpectralFlatnessExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = SpectralFlatnessExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.spectral.flatness"
    input_units = None
    output_units = "frame"
    dependencies = ["acoustic.spectral.spectrum"]
    default_config = {
        "eps": 1e-10,
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
            extractor = SpectralFlatnessExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        spectrum_output = feature_input.context["feature_store"].get(
            "acoustic.spectral.spectrum"
        )
        values = spectral_flatness(
            spectrum_output.values,
            eps=params.get("eps", 1e-10),
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(spectrum_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(SpectralFlatnessExtractor)
