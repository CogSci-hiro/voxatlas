import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.phonology.prosody_utils import detect_stress
from voxatlas.registry.feature_registry import registry


class ProsodicStressExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.prosody.stressed`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.prosody.stressed`` from VoxAtlas structured inputs. It consumes ``syllable`` units and produces values aligned to ``syllable`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives prosodic position or stress information from the aligned unit hierarchy.
    
    1. Hierarchical alignment
       Word, syllable, and IPU tables are linked so each target unit has access to its local structural context.
    
    2. Prosodic computation
       Depending on the feature, VoxAtlas computes a positional index, a normalized relative position, or a stress indicator :math:`\mathbf{1}[\mathrm{stressed}]`.
    
    3. Packaging
       The result is returned on the requested unit level for later modeling of prominence and discourse structure.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.phonology.prosody.stress import ProsodicStressExtractor
    >>> from voxatlas.units import Units
    >>> syllables = pd.DataFrame({"id": [10, 11], "start": [0.0, 0.5], "end": [0.5, 1.0]})
    >>> words = pd.DataFrame({"id": [1], "start": [0.0], "end": [1.0]})
    >>> ipus = pd.DataFrame({"id": [5], "start": [0.0], "end": [1.0]})
    >>> units = Units(syllables=syllables, words=words, ipus=ipus)
    >>> params = ProsodicStressExtractor.default_config.copy()
    >>> out = ProsodicStressExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
    >>> float(out.values.loc[10]), float(out.values.loc[11])
    (0.0, 1.0)
    """
    name = "phonology.prosody.stressed"
    input_units = "syllable"
    output_units = "syllable"
    dependencies = []
    default_config = {
        "language": None,
        "resource_root": None,
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
            Structured output aligned to the ``syllable`` unit level when applicable.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.phonology.prosody.stress import ProsodicStressExtractor
        >>> from voxatlas.units import Units
        >>> syllables = pd.DataFrame({"id": [10], "start": [0.0], "end": [0.5]})
        >>> words = pd.DataFrame({"id": [1], "start": [0.0], "end": [1.0]})
        >>> ipus = pd.DataFrame({"id": [5], "start": [0.0], "end": [1.0]})
        >>> units = Units(syllables=syllables, words=words, ipus=ipus)
        >>> params = ProsodicStressExtractor.default_config.copy()
        >>> result = ProsodicStressExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
        >>> result.unit
        'syllable'
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires syllable, word, and IPU units")

        units = feature_input.units
        stressed = detect_stress(
            syllables=units.get("syllable"),
            words=units.get("word"),
            ipus=units.get("ipu"),
            language=params.get("language"),
            resource_root=params.get("resource_root"),
        )
        values = pd.Series(
            stressed.to_numpy(dtype="float32"),
            index=units.get("syllable")["id"],
            dtype="float32",
        )

        return ScalarFeatureOutput(
            feature=self.name,
            unit="syllable",
            values=values,
        )


registry.register(ProsodicStressExtractor)
