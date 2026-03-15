from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.phonology.rhythm_utils import compute_rhythm_intervals
from voxatlas.registry.feature_registry import registry


class RhythmIntervalsExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.rhythm.intervals`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.rhythm.intervals`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``ipu`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives rhythm statistics from aligned syllables or phoneme-derived interval tables at the IPU level.
    
    1. Unit preparation
       Phoneme, syllable, and IPU tables are aligned so that each interval or syllable can be assigned to a speaking chunk.
    
    2. Metric computation
       Consecutive phonemes with the same vowel/consonant class are merged into intervals with duration :math:`d = t^{end} - t^{start}`.
    
    3. Packaging
       The result is aligned to ``ipu`` units so it can participate in later aggregation stages or conversation-level summaries.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['phonology.articulatory.vowel'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.phonology.rhythm.intervals import RhythmIntervalsExtractor
    >>> from voxatlas.features.feature_output import ScalarFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> from voxatlas.units import Units
    >>> phonemes = pd.DataFrame(
    ...     {"id": [1, 2, 3], "start": [0.0, 0.1, 0.2], "end": [0.1, 0.2, 0.3], "label": ["p", "a", "a"]}
    ... )
    >>> ipus = pd.DataFrame({"id": [5], "start": [0.0], "end": [1.0]})
    >>> units = Units(phonemes=phonemes, ipus=ipus)
    >>> store = FeatureStore()
    >>> store.add(
    ...     "phonology.articulatory.vowel",
    ...     ScalarFeatureOutput(
    ...         feature="phonology.articulatory.vowel",
    ...         unit="phoneme",
    ...         values=pd.Series([0.0, 1.0, 1.0], index=[1, 2, 3], dtype="float32"),
    ...     ),
    ... )
    >>> feature_input = FeatureInput(audio=None, units=units, context={"feature_store": store})
    >>> out = RhythmIntervalsExtractor().compute(feature_input, {})
    >>> out.values["type"].tolist()
    ['c', 'v']
    """
    name = "phonology.rhythm.intervals"
    input_units = "phoneme"
    output_units = "ipu"
    dependencies = ["phonology.articulatory.vowel"]
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
        >>> from voxatlas.features.phonology.rhythm.intervals import RhythmIntervalsExtractor
        >>> from voxatlas.features.feature_output import ScalarFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> from voxatlas.units import Units
        >>> phonemes = pd.DataFrame({"id": [1], "start": [0.0], "end": [0.1], "label": ["a"]})
        >>> ipus = pd.DataFrame({"id": [5], "start": [0.0], "end": [1.0]})
        >>> units = Units(phonemes=phonemes, ipus=ipus)
        >>> store = FeatureStore()
        >>> store.add(
        ...     "phonology.articulatory.vowel",
        ...     ScalarFeatureOutput(
        ...         feature="phonology.articulatory.vowel",
        ...         unit="phoneme",
        ...         values=pd.Series([1.0], index=[1], dtype="float32"),
        ...     ),
        ... )
        >>> feature_input = FeatureInput(audio=None, units=units, context={"feature_store": store})
        >>> result = RhythmIntervalsExtractor().compute(feature_input, {})
        >>> result.unit
        'ipu'
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires phoneme and IPU units")

        units = feature_input.units
        vowel_output = feature_input.context["feature_store"].get("phonology.articulatory.vowel")
        values = compute_rhythm_intervals(
            phonemes=units.get("phoneme"),
            vowel_flags=vowel_output.values,
            ipus=units.get("ipu"),
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="ipu",
            values=values,
        )


registry.register(RhythmIntervalsExtractor)
