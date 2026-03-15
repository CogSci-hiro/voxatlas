from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.registry.feature_registry import registry
from voxatlas.syntax.dependency_utils import extract_dependency_features


class SyntaxDependenciesExtractor(BaseExtractor):
    r"""
    Extract the ``syntax.dependencies`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``syntax.dependencies`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor derives syntactic descriptors from dependency annotations aligned to tokens or sentences.
    
    1. Dependency retrieval
       The required dependency table is loaded from the feature store.
    
    2. Structural computation
       The implementation applies relation labeling, clause grouping, or sentence-level aggregation depending on the extractor.
    
    3. Packaging
       Results are aligned to ``token`` units and returned for later discourse-level summaries.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.syntax.dependencies import SyntaxDependenciesExtractor
    >>> from voxatlas.units import Units
    >>> tokens = pd.DataFrame(
    ...     {"id": [1, 2], "token": ["hello", "world"], "head": [2, 0], "dep_rel": ["nsubj", "root"], "pos": ["INTJ", "NOUN"]}
    ... )
    >>> units = Units(tokens=tokens)
    >>> out = SyntaxDependenciesExtractor().compute(FeatureInput(audio=None, units=units, context={}), {"backend": "spacy"})
    >>> out.values.loc[:, ["token_id", "head_id"]].to_dict(orient="list")
    {'token_id': [1, 2], 'head_id': [2, 0]}
    """

    name = "syntax.dependencies"
    input_units = "token"
    output_units = "token"
    dependencies = []
    default_config = {"backend": "spacy"}

    def compute(self, feature_input, params):
        """
        Compute dependency annotations for one stream.

        Parameters
        ----------
        feature_input : FeatureInput
            Prepared stream input containing token annotations and context.
        params : dict
            Resolved extractor configuration.

        Returns
        -------
        TableFeatureOutput
            Token-aligned dependency table.

        Raises
        ------
        ValueError
            Raised when parsing fails or token annotations are incompatible.

        Notes
        -----
        The returned table is designed to be consumed by other syntax features.

        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.syntax.dependencies import SyntaxDependenciesExtractor
        >>> from voxatlas.units import Units
        >>> tokens = pd.DataFrame(
        ...     {"id": [1, 2], "token": ["hello", "world"], "head": [2, 0], "dep_rel": ["nsubj", "root"], "pos": ["INTJ", "NOUN"]}
        ... )
        >>> units = Units(tokens=tokens)
        >>> result = SyntaxDependenciesExtractor().compute(FeatureInput(audio=None, units=units, context={}), {"backend": "spacy"})
        >>> result.unit
        'token'
        """
        tokens = feature_input.units.get("token")
        values = extract_dependency_features(
            tokens,
            params=params,
            context=feature_input.context,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=values,
        )


registry.register(SyntaxDependenciesExtractor)
