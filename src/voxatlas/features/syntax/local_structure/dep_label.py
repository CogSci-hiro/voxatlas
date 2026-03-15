import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry


class SyntaxLocalStructureDependencyLabelExtractor(BaseExtractor):
    r"""
    Extract the ``syntax.local_structure.dep_label`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``syntax.local_structure.dep_label`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor works directly from the dependency-annotated token table produced earlier in the pipeline.
    
    1. Parsed token retrieval
       Token rows, head identifiers, and dependency metadata are loaded from the upstream syntax table.
    
    2. Local-structure computation
       The feature exposes the dependency label assigned to each token from the parsed dependency table.
    
    3. Packaging
       Values are returned at token resolution so they can be aggregated into sentence-level complexity measures later in the pipeline.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['syntax.dependencies'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.syntax.local_structure.dep_label import SyntaxLocalStructureDependencyLabelExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> deps = pd.DataFrame({"token_id": [1, 2], "dep_label": ["nsubj", "root"]})
    >>> store = FeatureStore()
    >>> store.add("syntax.dependencies", TableFeatureOutput(feature="syntax.dependencies", unit="token", values=deps))
    >>> out = SyntaxLocalStructureDependencyLabelExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
    >>> out.values.loc[1]
    'nsubj'
    """
    name = "syntax.local_structure.dep_label"
    input_units = "token"
    output_units = "token"
    dependencies = ["syntax.dependencies"]
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
        >>> from voxatlas.features.syntax.local_structure.dep_label import SyntaxLocalStructureDependencyLabelExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> deps = pd.DataFrame({"token_id": [1], "dep_label": ["root"]})
        >>> store = FeatureStore()
        >>> store.add("syntax.dependencies", TableFeatureOutput(feature="syntax.dependencies", unit="token", values=deps))
        >>> result = SyntaxLocalStructureDependencyLabelExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
        >>> result.unit
        'token'
        """
        table = feature_input.context["feature_store"].get("syntax.dependencies").values
        values = pd.Series(
            table["dep_label"].values,
            index=table["token_id"],
            dtype="object",
        )

        return ScalarFeatureOutput(feature=self.name, unit="token", values=values)


registry.register(SyntaxLocalStructureDependencyLabelExtractor)
