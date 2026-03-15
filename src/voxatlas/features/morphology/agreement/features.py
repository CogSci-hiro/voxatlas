import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.registry.feature_registry import registry
from voxatlas.syntax.agreement_utils import extract_agreement_features


class AgreementFeaturesExtractor(BaseExtractor):
    r"""
    Extract the ``morphology.agreement.features`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``morphology.agreement.features`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor projects morphological annotations or derived segmentation features onto the token index.
    
    1. Morphological preparation
       Token-level annotations or derived morphological resources are loaded from the dependency graph.
    
    2. Feature computation
       Depending on the extractor, the output is a categorical label, a binary indicator :math:`\mathbf{1}[\cdot]`, or a count such as :math:`N_i^{morpheme}`.
    
    3. Packaging
       The result is returned as a token-aligned scalar series so later discourse-level aggregation can preserve speaker and timing metadata.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['morphology.inflection.features'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import TableFeatureOutput
    >>> from voxatlas.features.morphology.agreement.features import AgreementFeaturesExtractor
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> inflection = pd.DataFrame(
    ...     [
    ...         {"id": 1, "head": 2, "dep_rel": "nsubj", "Person": 3, "Number": "Sing"},
    ...         {"id": 2, "head": 0, "dep_rel": "root", "Person": 3, "Number": "Sing"},
    ...     ]
    ... )
    >>> store = FeatureStore()
    >>> store.add("morphology.inflection.features", TableFeatureOutput(feature="morphology.inflection.features", unit="token", values=inflection))
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = AgreementFeaturesExtractor().compute(feature_input, {})
    >>> list(map(float, out.values["SubjectVerbAgreement"].tolist()))
    [1.0, 1.0]
    """
    name = "morphology.agreement.features"
    input_units = "token"
    output_units = "token"
    dependencies = ["morphology.inflection.features"]
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
        >>> from voxatlas.features.morphology.agreement.features import AgreementFeaturesExtractor
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> inflection = pd.DataFrame(
        ...     [
        ...         {"id": 1, "head": 2, "dep_rel": "nsubj", "Person": 3, "Number": "Sing"},
        ...         {"id": 2, "head": 0, "dep_rel": "root", "Person": 3, "Number": "Sing"},
        ...     ]
        ... )
        >>> store = FeatureStore()
        >>> store.add("morphology.inflection.features", TableFeatureOutput(feature="morphology.inflection.features", unit="token", values=inflection))
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = AgreementFeaturesExtractor().compute(feature_input, {})
        >>> result.unit
        'token'
        """
        inflection_table = feature_input.context["feature_store"].get(
            "morphology.inflection.features"
        ).values
        agreement_table = extract_agreement_features(inflection_table)
        merged = pd.concat(
            [
                inflection_table.reset_index(drop=True),
                agreement_table.drop(columns=["id"], errors="ignore"),
            ],
            axis=1,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=merged,
        )


registry.register(AgreementFeaturesExtractor)
