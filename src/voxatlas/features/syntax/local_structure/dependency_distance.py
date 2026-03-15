from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry
from voxatlas.syntax.dependency_utils import absolute_dependency_distance


class SyntaxLocalStructureDependencyDistanceExtractor(BaseExtractor):
    r"""
    Extract the ``syntax.local_structure.dependency_distance`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``syntax.local_structure.dependency_distance`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor works directly from the dependency-annotated token table produced earlier in the pipeline.
    
    1. Parsed token retrieval
       Token rows, head identifiers, and dependency metadata are loaded from the upstream syntax table.
    
    2. Local-structure computation
       The feature uses absolute linear distance :math:`|h_i - i|` between token :math:`i` and its head position :math:`h_i`.
    
    3. Packaging
       Values are returned at token resolution so they can be aggregated into sentence-level complexity measures later in the pipeline.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['syntax.dependencies'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.syntax.local_structure.dependency_distance import SyntaxLocalStructureDependencyDistanceExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = SyntaxLocalStructureDependencyDistanceExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "syntax.local_structure.dependency_distance"
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
            extractor = SyntaxLocalStructureDependencyDistanceExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        table = feature_input.context["feature_store"].get("syntax.dependencies").values
        values = absolute_dependency_distance(table)

        return ScalarFeatureOutput(feature=self.name, unit="token", values=values)


registry.register(SyntaxLocalStructureDependencyDistanceExtractor)
