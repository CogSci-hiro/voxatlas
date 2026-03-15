import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry


class DummyPitchExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.pitch.dummy`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.pitch.dummy`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``conversation`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor follows the standard VoxAtlas feature-computation pattern.
    
    1. Input preparation
       Structured audio, unit tables, and dependency outputs are gathered from ``feature_input``.
    
    2. Feature-specific computation
       The implementation applies the domain-specific transformation required by this extractor.
    
    3. Packaging
       Results are aligned to ``conversation`` units and returned as a ``FeatureOutput`` object.
    
    Examples
    --------
        from voxatlas.features.example_pitch import DummyPitchExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = DummyPitchExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.pitch.dummy"
    output_units = "conversation"

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
            Structured output aligned to the ``conversation`` unit level when applicable.
        
        Examples
        --------
            extractor = DummyPitchExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.audio is None and feature_input.units is None:
            raise ValueError(
                "acoustic.pitch.dummy requires at least one modality: audio or units"
            )

        values = pd.Series([0.0], dtype="float32")

        return ScalarFeatureOutput(
            feature=self.name,
            unit="conversation",
            values=values,
        )


registry.register(DummyPitchExtractor)
