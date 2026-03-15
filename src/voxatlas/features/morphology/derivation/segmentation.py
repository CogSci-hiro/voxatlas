import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.morphology.derivation_utils import (
    load_derivational_resources,
    segment_tokens,
)
from voxatlas.registry.feature_registry import registry


class DerivationalSegmentationExtractor(BaseExtractor):
    r"""
    Extract the ``morphology.derivation.segmentation`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``morphology.derivation.segmentation`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
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
    >>> from voxatlas.features.morphology.derivation.segmentation import DerivationalSegmentationExtractor
    >>> from voxatlas.units import Units
    >>> tokens = pd.DataFrame([{"id": 1, "token": "hello"}])
    >>> units = Units(tokens=tokens)
    >>> params = DerivationalSegmentationExtractor.default_config.copy()
    >>> params["language"] = ""  # no resource prefixes/suffixes
    >>> out = DerivationalSegmentationExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
    >>> float(out.values.loc[0, "morpheme_count"])
    1.0
    """
    name = "morphology.derivation.segmentation"
    input_units = "token"
    output_units = "token"
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
            Structured output aligned to the ``token`` unit level when applicable.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.morphology.derivation.segmentation import DerivationalSegmentationExtractor
        >>> from voxatlas.units import Units
        >>> tokens = pd.DataFrame([{"id": 1, "token": "hello"}])
        >>> units = Units(tokens=tokens)
        >>> params = DerivationalSegmentationExtractor.default_config.copy()
        >>> params["language"] = ""
        >>> result = DerivationalSegmentationExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
        >>> "prefix_presence" in result.values.columns
        True
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires token units")

        try:
            tokens = feature_input.units.get("token").copy()
        except Exception as exc:
            raise ValueError(f"{self.name} requires token units") from exc

        resources = load_derivational_resources(
            language=params.get("language"),
            resource_root=params.get("resource_root"),
        )
        segmented = segment_tokens(tokens, resources=resources)
        merged = pd.concat(
            [
                tokens.reset_index(drop=True),
                segmented.drop(columns=["id"], errors="ignore"),
            ],
            axis=1,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=merged,
        )


registry.register(DerivationalSegmentationExtractor)
