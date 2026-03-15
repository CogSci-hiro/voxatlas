from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.phonology.rhythm_utils import compute_syllable_rate
from voxatlas.registry.feature_registry import registry


class RhythmSyllableRateExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.rhythm.syllable_rate`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.rhythm.syllable_rate`` from VoxAtlas structured inputs. It consumes ``syllable`` units and produces values aligned to ``ipu`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives rhythm statistics from aligned syllables or phoneme-derived interval tables at the IPU level.
    
    1. Unit preparation
       Phoneme, syllable, and IPU tables are aligned so that each interval or syllable can be assigned to a speaking chunk.
    
    2. Metric computation
       Syllable rate is computed as
    
       .. math::
    
          r_j = \frac{N_j^{\mathrm{syll}}}{T_j}.
    
    3. Packaging
       The result is aligned to ``ipu`` units so it can participate in later aggregation stages or conversation-level summaries.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['phonology.rhythm.intervals'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.phonology.rhythm.syllable_rate import RhythmSyllableRateExtractor
    >>> from voxatlas.units import Units
    >>> syllables = pd.DataFrame({"id": [1, 2], "start": [0.0, 0.5], "end": [0.25, 0.75]})
    >>> ipus = pd.DataFrame({"id": [5], "start": [0.0], "end": [1.0]})
    >>> units = Units(syllables=syllables, ipus=ipus)
    >>> out = RhythmSyllableRateExtractor().compute(FeatureInput(audio=None, units=units, context={}), {})
    >>> float(out.values.loc[5])
    2.0
    """
    name = "phonology.rhythm.syllable_rate"
    input_units = "syllable"
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
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.phonology.rhythm.syllable_rate import RhythmSyllableRateExtractor
        >>> from voxatlas.units import Units
        >>> syllables = pd.DataFrame({"id": [1], "start": [0.0], "end": [0.25]})
        >>> ipus = pd.DataFrame({"id": [5], "start": [0.0], "end": [1.0]})
        >>> units = Units(syllables=syllables, ipus=ipus)
        >>> result = RhythmSyllableRateExtractor().compute(FeatureInput(audio=None, units=units, context={}), {})
        >>> result.unit
        'ipu'
        """
        units = feature_input.units
        if units is None:
            raise ValueError(f"{self.name} requires syllable and IPU units")

        values = compute_syllable_rate(
            syllables=units.get("syllable"),
            ipus=units.get("ipu"),
        )
        return ScalarFeatureOutput(feature=self.name, unit="ipu", values=values)


registry.register(RhythmSyllableRateExtractor)
