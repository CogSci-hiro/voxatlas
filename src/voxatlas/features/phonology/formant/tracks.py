from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.phonology.formant_utils import compute_formant_tracks
from voxatlas.registry.feature_registry import registry


class FormantTracksExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.formant.tracks`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.formant.tracks`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives vowel formant structure from aligned phoneme segments and then aggregates those measurements to the declared unit level.
    
    1. Segment selection
       Vowel-bearing phoneme spans are isolated from the aligned unit table, and the corresponding waveform segments are converted into short analysis frames.
    
    2. Resonance estimation
       Linear-predictive analysis or Parselmouth formant tracking is used to estimate :math:`F_1`, :math:`F_2`, and :math:`F_3` for each analysis frame.
    
    3. Metric computation
       Linear-predictive analysis or Parselmouth formant tracking is used to recover resonant frequencies :math:`F_1`, :math:`F_2`, and :math:`F_3` over vowel-bearing segments.
    
    4. Packaging
       The resulting statistic is aligned to ``frame`` units for use in subsequent phonological or conversation-level analyses.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from voxatlas.audio.audio import Audio
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.phonology.formant.tracks import FormantTracksExtractor
    >>> from voxatlas.units import Units
    >>> audio = Audio(waveform=np.zeros(800, dtype=np.float32), sample_rate=8000)
    >>> phonemes = pd.DataFrame(columns=["id", "start", "end", "label"])
    >>> units = Units(phonemes=phonemes)
    >>> params = FormantTracksExtractor.default_config.copy()
    >>> params["use_parselmouth"] = False
    >>> out = FormantTracksExtractor().compute(FeatureInput(audio=audio, units=units, context={}), params)
    >>> ("F1" in out.values.columns, out.unit)
    (True, 'frame')
    """
    name = "phonology.formant.tracks"
    input_units = "phoneme"
    output_units = "frame"
    dependencies = []
    default_config = {
        "language": None,
        "resource_root": None,
        "frame_length": 0.025,
        "frame_step": 0.010,
        "lpc_order": 10,
        "max_formant": 5500.0,
        "use_parselmouth": True,
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
            Structured output aligned to the ``frame`` unit level when applicable.
        
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from voxatlas.audio.audio import Audio
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.phonology.formant.tracks import FormantTracksExtractor
        >>> from voxatlas.units import Units
        >>> audio = Audio(waveform=np.zeros(800, dtype=np.float32), sample_rate=8000)
        >>> phonemes = pd.DataFrame(columns=["id", "start", "end", "label"])
        >>> units = Units(phonemes=phonemes)
        >>> params = FormantTracksExtractor.default_config.copy()
        >>> params["use_parselmouth"] = False
        >>> result = FormantTracksExtractor().compute(FeatureInput(audio=audio, units=units, context={}), params)
        >>> result.values.empty
        True
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires phoneme units")

        values = compute_formant_tracks(
            signal=feature_input.audio.waveform,
            sr=feature_input.audio.sample_rate,
            phonemes=feature_input.units.get("phoneme"),
            language=params.get("language"),
            resource_root=params.get("resource_root"),
            frame_length=params.get("frame_length", 0.025),
            frame_step=params.get("frame_step", 0.010),
            lpc_order=params.get("lpc_order", 10),
            max_formant=params.get("max_formant", 5500.0),
            use_parselmouth=params.get("use_parselmouth", True),
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="frame",
            values=values,
        )


registry.register(FormantTracksExtractor)
