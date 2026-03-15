import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.lexical.properties_utils import compute_lexical_properties
from voxatlas.registry.feature_registry import registry


class LexicalPropertiesExtractor(BaseExtractor):
    r"""
    Extract the ``lexical.properties.features`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``lexical.properties.features`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor computes a lookup-based lexical property from a resource table or token annotation.
    
    1. Token preparation
       Token rows are normalized so that text, lemma, or aligned subunit identifiers can be queried consistently.
    
    2. Property computation
       The feature value follows
    
       .. math::
    
          x_i = L(w_i).
    
    3. Packaging
       The resulting token-level series is returned without altering the original unit index.
    
    Examples
    --------
        from voxatlas.features.lexical.properties.features import LexicalPropertiesExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = LexicalPropertiesExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "lexical.properties.features"
    input_units = "token"
    output_units = "token"
    dependencies = []
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
            Structured output aligned to the ``token`` unit level when applicable.
        
        Examples
        --------
            extractor = LexicalPropertiesExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires token, syllable, and phoneme units")

        units = feature_input.units
        tokens = units.get("token").copy()
        properties = compute_lexical_properties(
            tokens=tokens,
            syllables=units.get("syllable"),
            phonemes=units.get("phoneme"),
        )
        merged = pd.concat(
            [
                tokens.reset_index(drop=True),
                properties.drop(columns=["id"], errors="ignore"),
            ],
            axis=1,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=merged,
        )


registry.register(LexicalPropertiesExtractor)
