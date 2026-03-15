from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.phonology.formant_utils import compute_dct_coefficients
from voxatlas.registry.feature_registry import registry


class FormantDCTExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.formant.dct`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.formant.dct`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``phoneme`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives vowel formant structure from aligned phoneme segments and then aggregates those measurements to the declared unit level.
    
    1. Segment selection
       Vowel-bearing phoneme spans are isolated from the aligned unit table, and the corresponding waveform segments are converted into short analysis frames.
    
    2. Resonance estimation
       Linear-predictive analysis or Parselmouth formant tracking is used to estimate :math:`F_1`, :math:`F_2`, and :math:`F_3` for each analysis frame.
    
    3. Metric computation
       The extractor projects each formant trajectory onto a discrete cosine basis to obtain low-dimensional trajectory coefficients.
    
    4. Packaging
       The resulting statistic is aligned to ``phoneme`` units for use in subsequent phonological or conversation-level analyses.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['phonology.formant.tracks'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.phonology.formant.dct import FormantDCTExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> tracks = pd.DataFrame(
    ...     [
    ...         {"frame_id": 1, "start": 0.0, "end": 0.01, "time": 0.005, "phoneme_id": 1, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 300.0, "F2": 2200.0, "F3": 3000.0},
    ...         {"frame_id": 2, "start": 0.01, "end": 0.02, "time": 0.015, "phoneme_id": 1, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 320.0, "F2": 2180.0, "F3": 2980.0},
    ...         {"frame_id": 3, "start": 0.02, "end": 0.03, "time": 0.025, "phoneme_id": 1, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 340.0, "F2": 2160.0, "F3": 2960.0},
    ...     ]
    ... )
    >>> store = FeatureStore()
    >>> store.add("phonology.formant.tracks", TableFeatureOutput(feature="phonology.formant.tracks", unit="frame", values=tracks))
    >>> params = FormantDCTExtractor.default_config.copy()
    >>> params["n_coeffs"] = 2
    >>> out = FormantDCTExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), params)
    >>> "F1_dct0" in out.values.columns
    True
    """
    name = "phonology.formant.dct"
    input_units = "phoneme"
    output_units = "phoneme"
    dependencies = ["phonology.formant.tracks"]
    default_config = {
        "n_coeffs": 3,
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
            Structured output aligned to the ``phoneme`` unit level when applicable.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import TableFeatureOutput
        >>> from voxatlas.features.phonology.formant.dct import FormantDCTExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> tracks = pd.DataFrame(
        ...     [{"frame_id": 1, "start": 0.0, "end": 0.01, "time": 0.005, "phoneme_id": 1, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 300.0, "F2": 2200.0, "F3": 3000.0}]
        ... )
        >>> store = FeatureStore()
        >>> store.add("phonology.formant.tracks", TableFeatureOutput(feature="phonology.formant.tracks", unit="frame", values=tracks))
        >>> result = FormantDCTExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
        >>> result.unit
        'phoneme'
        """
        tracks = feature_input.context["feature_store"].get("phonology.formant.tracks").values
        return TableFeatureOutput(
            feature=self.name,
            unit="phoneme",
            values=compute_dct_coefficients(
                tracks,
                n_coeffs=params.get("n_coeffs", 3),
            ),
        )


registry.register(FormantDCTExtractor)
