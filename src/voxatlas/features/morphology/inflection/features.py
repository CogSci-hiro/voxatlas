import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.morphology.inflection_utils import extract_inflection_features
from voxatlas.registry.feature_registry import registry


class InflectionFeaturesExtractor(BaseExtractor):
    r"""
    Extract the ``morphology.inflection.features`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``morphology.inflection.features`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor projects morphological annotations or derived segmentation features onto the token index.
    
    1. Morphological preparation
       Token-level annotations or derived morphological resources are loaded from the dependency graph.
    
    2. Feature computation
       Depending on the extractor, the output is a categorical label, a binary indicator :math:`\mathbf{1}[\cdot]`, or a count such as :math:`N_i^{morpheme}`.
    
    3. Packaging
       The result is returned as a token-aligned scalar series so later discourse-level aggregation can preserve speaker and timing metadata.
    
    Examples
    --------
        from voxatlas.features.morphology.inflection.features import InflectionFeaturesExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = InflectionFeaturesExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "morphology.inflection.features"
    input_units = "token"
    output_units = "token"
    dependencies = []
    default_config = {
        "morphology_lexicon": None,
        "morphological_analysis": None,
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
            Structured output aligned to the ``token`` unit level when applicable.
        
        Examples
        --------
            extractor = InflectionFeaturesExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires token units")

        try:
            tokens = feature_input.units.get("token").copy()
        except Exception as exc:
            raise ValueError(f"{self.name} requires token units") from exc

        resources = {
            "morphology_lexicon": params.get("morphology_lexicon"),
            "morphological_analysis": params.get("morphological_analysis"),
        }
        inflection_table = extract_inflection_features(tokens, resources=resources)
        merged = pd.concat(
            [
                tokens.reset_index(drop=True),
                inflection_table.drop(columns=["id"], errors="ignore"),
            ],
            axis=1,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=merged,
        )


registry.register(InflectionFeaturesExtractor)
