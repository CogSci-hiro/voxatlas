from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry
from voxatlas.syntax.clause_utils import compute_clause_membership


RELATIVE_CLAUSE_LABELS = {"acl"}


class SyntaxRelativeClauseExtractor(BaseExtractor):
    r"""
    Extract the ``syntax.clause_structure.relative_clause`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``syntax.clause_structure.relative_clause`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives syntactic descriptors from dependency annotations aligned to tokens or sentences.
    
    1. Dependency retrieval
       The required dependency table is loaded from the feature store.
    
    2. Structural computation
       The implementation applies relation labeling, clause grouping, or sentence-level aggregation depending on the extractor.
    
    3. Packaging
       Results are aligned to ``token`` units and returned for later discourse-level summaries.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['syntax.dependencies'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.syntax.clause_structure.relative_clause import SyntaxRelativeClauseExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> deps = pd.DataFrame({"id": [1, 2], "deprel": ["acl", "root"]})
    >>> store = FeatureStore()
    >>> store.add("syntax.dependencies", TableFeatureOutput(feature="syntax.dependencies", unit="token", values=deps))
    >>> out = SyntaxRelativeClauseExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
    >>> out.values.to_dict()
    {1: 1, 2: 0}
    """
    name = "syntax.clause_structure.relative_clause"
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
        >>> from voxatlas.features.syntax.clause_structure.relative_clause import SyntaxRelativeClauseExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> deps = pd.DataFrame({"id": [1], "deprel": ["root"]})
        >>> store = FeatureStore()
        >>> store.add("syntax.dependencies", TableFeatureOutput(feature="syntax.dependencies", unit="token", values=deps))
        >>> result = SyntaxRelativeClauseExtractor().compute(FeatureInput(audio=None, units=None, context={"feature_store": store}), {})
        >>> result.unit
        'token'
        """
        table = feature_input.context["feature_store"].get("syntax.dependencies").values
        values = compute_clause_membership(table, RELATIVE_CLAUSE_LABELS)

        return ScalarFeatureOutput(
            feature=self.name,
            unit="token",
            values=values,
        )


registry.register(SyntaxRelativeClauseExtractor)
