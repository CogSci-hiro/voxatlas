import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry


class MorphemeCountExtractor(BaseExtractor):
    r"""
    Extract the ``morphology.derivation.morpheme_count`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``morphology.derivation.morpheme_count`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor projects morphological annotations or derived segmentation features onto the token index.
    
    1. Morphological preparation
       Token-level annotations or derived morphological resources are loaded from the dependency graph.
    
    2. Feature computation
       Depending on the extractor, the output is a categorical label, a binary indicator :math:`\mathbf{1}[\cdot]`, or a count such as :math:`N_i^{morpheme}`.
    
    3. Packaging
       The result is returned as a token-aligned scalar series so later discourse-level aggregation can preserve speaker and timing metadata.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['morphology.derivation.segmentation'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.morphology.derivation.morpheme_count import MorphemeCountExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> table = pd.DataFrame({"id": [1], "morpheme_count": [2.0]})
    >>> store = FeatureStore()
    >>> store.add("morphology.derivation.segmentation", TableFeatureOutput(feature="morphology.derivation.segmentation", unit="token", values=table))
    >>> out = MorphemeCountExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
    >>> float(out.values.loc[1])
    2.0
    """
    name = "morphology.derivation.morpheme_count"
    input_units = "token"
    output_units = "token"
    dependencies = ["morphology.derivation.segmentation"]
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
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import TableFeatureOutput
        >>> from voxatlas.features.morphology.derivation.morpheme_count import MorphemeCountExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> table = pd.DataFrame({"id": [1], "morpheme_count": [2.0]})
        >>> store = FeatureStore()
        >>> store.add("morphology.derivation.segmentation", TableFeatureOutput(feature="morphology.derivation.segmentation", unit="token", values=table))
        >>> result = MorphemeCountExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
        >>> result.unit
        'token'
        """
        table = feature_input.context["feature_store"].get(
            "morphology.derivation.segmentation"
        ).values
        values = pd.Series(
            table["morpheme_count"].astype("float32").values,
            index=table["id"],
            dtype="float32",
        )
        return ScalarFeatureOutput(feature=self.name, unit="token", values=values)


registry.register(MorphemeCountExtractor)
