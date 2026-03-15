from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.lexical.frequency_utils import compute_zipf_frequency
from voxatlas.registry.feature_registry import registry


class ZipfFrequencyExtractor(BaseExtractor):
    r"""
    Extract the ``lexical.frequency.zipf`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``lexical.frequency.zipf`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor converts raw lexical frequencies to the Zipf scale commonly used in psycholinguistic reporting.
    
    1. Dependency retrieval
       The upstream lookup stage provides raw frequency values :math:`f_i` for each token.
    
    2. Logarithmic transformation
       VoxAtlas computes
    
       .. math::
    
          z_i = \log_{10}(f_i) + 3,
    
       which places frequent words on a compact approximately human-interpretable scale.
    
    3. Packaging
       Non-finite inputs remain missing, and the transformed values are returned on the token index.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['lexical.frequency.lookup'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.lexical.frequency.zipf_frequency import ZipfFrequencyExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = ZipfFrequencyExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "lexical.frequency.zipf"
    input_units = "token"
    output_units = "token"
    dependencies = ["lexical.frequency.lookup"]
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
            extractor = ZipfFrequencyExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        table = feature_input.context["feature_store"].get(
            "lexical.frequency.lookup"
        ).values
        values = compute_zipf_frequency(table["frequency"])
        values.index = table["id"]

        return ScalarFeatureOutput(
            feature=self.name,
            unit="token",
            values=values,
        )


registry.register(ZipfFrequencyExtractor)
