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
        from voxatlas.features.syntax.dependencies import SyntaxDependenciesExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = SyntaxDependenciesExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
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
        Usage example::

            output = SyntaxDependenciesExtractor().compute(feature_input, params)
            print(output.values.columns)
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
