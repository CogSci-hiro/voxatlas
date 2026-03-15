import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.lexical.property_utils import (
    load_lexical_property_resources,
    lookup_lexical_properties,
)
from voxatlas.registry.feature_registry import registry


class LexicalPropertyLookupExtractor(BaseExtractor):
    r"""
    Extract the ``lexical.properties.lookup`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``lexical.properties.lookup`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
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
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.lexical.properties.lookup import LexicalPropertyLookupExtractor
    >>> from voxatlas.units import Units
    >>> tokens = pd.DataFrame([{"id": 1, "lemma": "dog", "upos": "NOUN"}])
    >>> units = Units(tokens=tokens)
    >>> params = LexicalPropertyLookupExtractor.default_config.copy()
    >>> params["language"] = ""  # empty resources, so values remain NaN
    >>> out = LexicalPropertyLookupExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
    >>> out.values.loc[0, "pos"]
    'NOUN'
    """
    name = "lexical.properties.lookup"
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
        >>> from voxatlas.features.lexical.properties.lookup import LexicalPropertyLookupExtractor
        >>> from voxatlas.units import Units
        >>> tokens = pd.DataFrame([{"id": 1, "lemma": "dog", "upos": "NOUN"}])
        >>> units = Units(tokens=tokens)
        >>> params = LexicalPropertyLookupExtractor.default_config.copy()
        >>> params["language"] = ""
        >>> result = LexicalPropertyLookupExtractor().compute(FeatureInput(audio=None, units=units, context={}), params)
        >>> result.unit
        'token'
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires token units")

        try:
            tokens = feature_input.units.get("token").copy()
        except Exception as exc:
            raise ValueError(f"{self.name} requires token units") from exc

        resources = load_lexical_property_resources(
            language=params.get("language"),
            resource_root=params.get("resource_root"),
        )
        property_table = lookup_lexical_properties(tokens, resources).reset_index(drop=True)
        merged = tokens.reset_index(drop=True).copy()

        for column in ("pos", "function_word", "animacy", "concreteness"):
            merged[column] = property_table[column]

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=merged,
        )


registry.register(LexicalPropertyLookupExtractor)
