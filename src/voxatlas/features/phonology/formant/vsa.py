import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.phonology.formant_utils import compute_vsa
from voxatlas.registry.feature_registry import registry


class FormantVSAExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.formant.vsa`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.formant.vsa`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``conversation`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives vowel formant structure from aligned phoneme segments and then aggregates those measurements to the declared unit level.
    
    1. Segment selection
       Vowel-bearing phoneme spans are isolated from the aligned unit table, and the corresponding waveform segments are converted into short analysis frames.
    
    2. Resonance estimation
       Linear-predictive analysis or Parselmouth formant tracking is used to estimate :math:`F_1`, :math:`F_2`, and :math:`F_3` for each analysis frame.
    
    3. Metric computation
       The extractor computes vowel-space area from representative corner-vowel means, using the polygon area implied by their :math:`(F_1, F_2)` coordinates.
    
    4. Packaging
       The resulting statistic is aligned to ``conversation`` units for use in subsequent phonological or conversation-level analyses.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['phonology.formant.tracks'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.phonology.formant.vsa import FormantVSAExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> from voxatlas.units import Units
    >>> tracks = pd.DataFrame(
    ...     [
    ...         {"frame_id": 1, "start": 0.0, "end": 0.01, "time": 0.005, "phoneme_id": 1, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 300.0, "F2": 2200.0, "F3": 3000.0},
    ...         {"frame_id": 2, "start": 0.1, "end": 0.11, "time": 0.105, "phoneme_id": 2, "label": "a", "ipa": "a", "is_vowel": 1.0, "F1": 700.0, "F2": 1200.0, "F3": 2600.0},
    ...         {"frame_id": 3, "start": 0.2, "end": 0.21, "time": 0.205, "phoneme_id": 3, "label": "u", "ipa": "u", "is_vowel": 1.0, "F1": 350.0, "F2": 900.0, "F3": 2400.0},
    ...     ]
    ... )
    >>> store = FeatureStore()
    >>> store.add("phonology.formant.tracks", TableFeatureOutput(feature="phonology.formant.tracks", unit="frame", values=tracks))
    >>> units = Units(speaker="A")
    >>> out = FormantVSAExtractor().compute(FeatureInput(audio=None, units=units, context={"feature_store": store}), {})
    >>> out.unit
    'conversation'
    """
    name = "phonology.formant.vsa"
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
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import TableFeatureOutput
        >>> from voxatlas.features.phonology.formant.vsa import FormantVSAExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> from voxatlas.units import Units
        >>> tracks = pd.DataFrame(
        ...     [
        ...         {"frame_id": 1, "start": 0.0, "end": 0.01, "time": 0.005, "phoneme_id": 1, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 300.0, "F2": 2200.0, "F3": 3000.0},
        ...         {"frame_id": 2, "start": 0.1, "end": 0.11, "time": 0.105, "phoneme_id": 2, "label": "a", "ipa": "a", "is_vowel": 1.0, "F1": 700.0, "F2": 1200.0, "F3": 2600.0},
        ...         {"frame_id": 3, "start": 0.2, "end": 0.21, "time": 0.205, "phoneme_id": 3, "label": "u", "ipa": "u", "is_vowel": 1.0, "F1": 350.0, "F2": 900.0, "F3": 2400.0},
        ...     ]
        ... )
        >>> store = FeatureStore()
        >>> store.add("phonology.formant.tracks", TableFeatureOutput(feature="phonology.formant.tracks", unit="frame", values=tracks))
        >>> result = FormantVSAExtractor().compute(FeatureInput(audio=None, units=Units(speaker="A"), context={"feature_store": store}), {})
        >>> list(result.values.index)
        ['A']
        """
        tracks = feature_input.context["feature_store"].get("phonology.formant.tracks").values
        speaker = feature_input.units.speaker if feature_input.units is not None else "speaker"
        values = pd.Series([compute_vsa(tracks)], index=[speaker], dtype="float32")
        return ScalarFeatureOutput(feature=self.name, unit="conversation", values=values)


registry.register(FormantVSAExtractor)
