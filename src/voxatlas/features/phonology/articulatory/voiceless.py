import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry


class ArticulatoryVoicelessExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.articulatory.voiceless`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.articulatory.voiceless`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``phoneme`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor maps phoneme labels to articulatory classes using the phonology resource tables bundled with VoxAtlas.
    
    1. Resource lookup
       Each aligned phoneme label is normalized to IPA-like form and matched against the articulatory feature inventory.
    
    2. Class projection
       The output is a binary or categorical indicator, typically representable as :math:`x_i = \mathbf{1}[\mathrm{phoneme}_i \in C]` for a class :math:`C` such as vowels, nasals, or plosives.
    
    3. Packaging
       The resulting phoneme-aligned values can then be aggregated into rhythm or segmental summaries.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['phonology.articulatory.features'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.phonology.articulatory.voiceless import ArticulatoryVoicelessExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> table = pd.DataFrame({"id": [1], "voiceless": [1]})
    >>> store = FeatureStore()
    >>> store.add(
    ...     "phonology.articulatory.features",
    ...     TableFeatureOutput(feature="phonology.articulatory.features", unit="phoneme", values=table),
    ... )
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = ArticulatoryVoicelessExtractor().compute(feature_input, {})
    >>> float(out.values.loc[1])
    1.0
    """
    name = "phonology.articulatory.voiceless"
    input_units = "phoneme"
    output_units = "phoneme"
    dependencies = ["phonology.articulatory.features"]
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
            Structured output aligned to the ``phoneme`` unit level when applicable.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import TableFeatureOutput
        >>> from voxatlas.features.phonology.articulatory.voiceless import ArticulatoryVoicelessExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> table = pd.DataFrame({"id": [1], "voiceless": [1]})
        >>> store = FeatureStore()
        >>> store.add(
        ...     "phonology.articulatory.features",
        ...     TableFeatureOutput(feature="phonology.articulatory.features", unit="phoneme", values=table),
        ... )
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = ArticulatoryVoicelessExtractor().compute(feature_input, {})
        >>> result.unit
        'phoneme'
        """
        table = feature_input.context["feature_store"].get(
            "phonology.articulatory.features"
        ).values
        values = pd.Series(table["voiceless"].astype("float32").values, index=table["id"])
        return ScalarFeatureOutput(feature=self.name, unit="phoneme", values=values)


registry.register(ArticulatoryVoicelessExtractor)
