from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry
from voxatlas.syntax.complexity_utils import compute_parse_tree_depth_by_sentence


class SyntaxComplexityParseTreeDepthExtractor(BaseExtractor):
    r"""
    Extract the ``syntax.complexity.parse_tree_depth`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``syntax.complexity.parse_tree_depth`` from VoxAtlas structured inputs. It consumes ``sentence`` units and produces values aligned to ``sentence`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor converts token-level dependency annotations into sentence-level structural complexity statistics.
    
    1. Sentence partitioning
       The dependency table is split by sentence so each statistic is computed on a well-defined local parse.
    
    2. Tree construction
       VoxAtlas reconstructs a dependency tree rooted at the sentence head.
    
    3. Complexity computation
       Parse-tree depth is the maximum root-to-leaf path length in the dependency tree.
    
    4. Packaging
       One scalar is returned per sentence, aligned to the sentence identifiers used elsewhere in the pipeline.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['syntax.dependencies'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.syntax.complexity.parse_tree_depth import SyntaxComplexityParseTreeDepthExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = SyntaxComplexityParseTreeDepthExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "syntax.complexity.parse_tree_depth"
    input_units = "sentence"
    output_units = "sentence"
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
            Structured output aligned to the ``sentence`` unit level when applicable.
        
        Examples
        --------
            extractor = SyntaxComplexityParseTreeDepthExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        table = feature_input.context["feature_store"].get("syntax.dependencies").values
        values = compute_parse_tree_depth_by_sentence(table)

        return ScalarFeatureOutput(
            feature=self.name,
            unit="sentence",
            values=values,
        )


registry.register(SyntaxComplexityParseTreeDepthExtractor)
