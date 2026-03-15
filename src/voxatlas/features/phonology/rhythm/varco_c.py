from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.phonology.rhythm_utils import compute_varco_c
from voxatlas.registry.feature_registry import registry


class RhythmVarcoCExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.rhythm.varco_c`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.rhythm.varco_c`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``ipu`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives rhythm statistics from aligned syllables or phoneme-derived interval tables at the IPU level.
    
    1. Unit preparation
       Phoneme, syllable, and IPU tables are aligned so that each interval or syllable can be assigned to a speaking chunk.
    
    2. Metric computation
       VarcoC is
    
       .. math::
    
          \mathrm{VarcoC} = 100\frac{\mathrm{sd}(d^{(c)})}{\overline{d^{(c)}}}.
    
    3. Packaging
       The result is aligned to ``ipu`` units so it can participate in later aggregation stages or conversation-level summaries.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['phonology.rhythm.intervals'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.phonology.rhythm.varco_c import RhythmVarcoCExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = RhythmVarcoCExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "phonology.rhythm.varco_c"
    input_units = "phoneme"
    output_units = "ipu"
    dependencies = ["phonology.rhythm.intervals"]
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
            Structured output aligned to the ``ipu`` unit level when applicable.
        
        Examples
        --------
            extractor = RhythmVarcoCExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        intervals = feature_input.context["feature_store"].get("phonology.rhythm.intervals").values
        values = compute_varco_c(intervals)
        return ScalarFeatureOutput(feature=self.name, unit="ipu", values=values)


registry.register(RhythmVarcoCExtractor)
