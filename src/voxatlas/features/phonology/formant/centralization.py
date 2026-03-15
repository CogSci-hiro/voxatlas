import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.phonology.formant_utils import compute_centralization
from voxatlas.registry.feature_registry import registry


class FormantCentralizationExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.formant.centralization`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.formant.centralization`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``conversation`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives vowel formant structure from aligned phoneme segments and then aggregates those measurements to the declared unit level.
    
    1. Segment selection
       Vowel-bearing phoneme spans are isolated from the aligned unit table, and the corresponding waveform segments are converted into short analysis frames.
    
    2. Resonance estimation
       Linear-predictive analysis or Parselmouth formant tracking is used to estimate :math:`F_1`, :math:`F_2`, and :math:`F_3` for each analysis frame.
    
    3. Metric computation
       The extractor summarizes movement toward the acoustic center by contrasting peripheral-vowel formant locations against centralized realizations.
    
    4. Packaging
       The resulting statistic is aligned to ``conversation`` units for use in subsequent phonological or conversation-level analyses.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['phonology.formant.tracks'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.phonology.formant.centralization import FormantCentralizationExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = FormantCentralizationExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "phonology.formant.centralization"
    input_units = "phoneme"
    output_units = "conversation"
    dependencies = ["phonology.formant.tracks"]
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
            Structured output aligned to the ``conversation`` unit level when applicable.
        
        Examples
        --------
            extractor = FormantCentralizationExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        tracks = feature_input.context["feature_store"].get("phonology.formant.tracks").values
        speaker = feature_input.units.speaker if feature_input.units is not None else "speaker"
        values = pd.Series([compute_centralization(tracks)], index=[speaker], dtype="float32")
        return ScalarFeatureOutput(feature=self.name, unit="conversation", values=values)


registry.register(FormantCentralizationExtractor)
