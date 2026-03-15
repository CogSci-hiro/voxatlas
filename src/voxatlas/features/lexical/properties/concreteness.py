import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry


class ConcretenessExtractor(BaseExtractor):
    r"""
    Extract the ``lexical.properties.concreteness`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``lexical.properties.concreteness`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor computes a lookup-based lexical property from a resource table or token annotation.
    
    1. Token preparation
       Token rows are normalized so that text, lemma, or aligned subunit identifiers can be queried consistently.
    
    2. Property computation
       The feature value follows
    
       .. math::
    
          x_i = L(w_i).
    
    3. Packaging
       The resulting token-level series is returned without altering the original unit index.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['lexical.properties.lookup'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.lexical.properties.concreteness import ConcretenessExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> table = pd.DataFrame({"id": [1], "concreteness": [4.5]})
    >>> store = FeatureStore()
    >>> store.add("lexical.properties.lookup", TableFeatureOutput(feature="lexical.properties.lookup", unit="token", values=table))
    >>> out = ConcretenessExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
    >>> float(out.values.loc[1])
    4.5
    """
    name = "lexical.properties.concreteness"
    input_units = "token"
    output_units = "token"
    dependencies = ["lexical.properties.lookup"]
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
        >>> from voxatlas.features.lexical.properties.concreteness import ConcretenessExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> table = pd.DataFrame({"id": [1], "concreteness": [4.5]})
        >>> store = FeatureStore()
        >>> store.add("lexical.properties.lookup", TableFeatureOutput(feature="lexical.properties.lookup", unit="token", values=table))
        >>> result = ConcretenessExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
        >>> result.unit
        'token'
        """
        table = feature_input.context["feature_store"].get(
            "lexical.properties.lookup"
        ).values
        values = pd.Series(
            table["concreteness"].values,
            index=table["id"],
            dtype="object",
        )

        return ScalarFeatureOutput(
            feature=self.name,
            unit="token",
            values=values,
        )


registry.register(ConcretenessExtractor)
