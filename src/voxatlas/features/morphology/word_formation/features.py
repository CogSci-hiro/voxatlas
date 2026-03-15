import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.morphology.word_formation_utils import extract_word_formation_features
from voxatlas.registry.feature_registry import registry


class WordFormationFeaturesExtractor(BaseExtractor):
    r"""
    Extract the ``morphology.word_formation.features`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``morphology.word_formation.features`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor projects morphological annotations or derived segmentation features onto the token index.
    
    1. Morphological preparation
       Token-level annotations or derived morphological resources are loaded from the dependency graph.
    
    2. Feature computation
       Depending on the extractor, the output is a categorical label, a binary indicator :math:`\mathbf{1}[\cdot]`, or a count such as :math:`N_i^{morpheme}`.
    
    3. Packaging
       The result is returned as a token-aligned scalar series so later discourse-level aggregation can preserve speaker and timing metadata.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.morphology.word_formation.features import WordFormationFeaturesExtractor
    >>> from voxatlas.units import Units
    >>> tokens = pd.DataFrame([{"id": 1, "text": "ice-cream"}])
    >>> units = Units(tokens=tokens)
    >>> params = WordFormationFeaturesExtractor.default_config.copy()
    >>> params["clitic_list"] = ["l"]
    >>> params["lexicon_lookup"] = ["ice", "cream"]
    >>> out = WordFormationFeaturesExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
    >>> out.values.loc[0, ["Compound", "Clitic"]].to_dict()
    {'Compound': 1.0, 'Clitic': 0.0}
    """
    name = "morphology.word_formation.features"
    input_units = "token"
    output_units = "token"
    dependencies = []
    default_config = {
        "language": None,
        "resource_root": None,
        "clitic_list": None,
        "lexicon_lookup": None,
        "segmentation_lookup": None,
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
            Structured output aligned to the ``token`` unit level when applicable.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.morphology.word_formation.features import WordFormationFeaturesExtractor
        >>> from voxatlas.units import Units
        >>> tokens = pd.DataFrame([{"id": 1, "text": "ice-cream"}])
        >>> units = Units(tokens=tokens)
        >>> params = WordFormationFeaturesExtractor.default_config.copy()
        >>> params["clitic_list"] = ["l"]
        >>> params["lexicon_lookup"] = ["ice", "cream"]
        >>> result = WordFormationFeaturesExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
        >>> result.unit
        'token'
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires token units")

        try:
            tokens = feature_input.units.get("token").copy()
        except Exception as exc:
            raise ValueError(f"{self.name} requires token units") from exc

        feature_table = extract_word_formation_features(
            tokens=tokens,
            language=params.get("language"),
            resource_root=params.get("resource_root"),
            clitic_list=params.get("clitic_list"),
            lexicon_lookup=params.get("lexicon_lookup"),
            segmentation_lookup=params.get("segmentation_lookup"),
        )
        merged = pd.concat(
            [
                tokens.reset_index(drop=True),
                feature_table.drop(columns=["id"], errors="ignore"),
            ],
            axis=1,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=merged,
        )


registry.register(WordFormationFeaturesExtractor)
