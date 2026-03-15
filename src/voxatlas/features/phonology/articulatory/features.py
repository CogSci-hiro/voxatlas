from __future__ import annotations

import numpy as np
import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.phonology.articulatory_utils import (
    FEATURE_COLUMNS,
    load_phonology_resources,
    log_unknown_phoneme,
    lookup_articulatory_features,
)
from voxatlas.registry.feature_registry import registry


class ArticulatoryFeaturesExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.articulatory.features`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.articulatory.features`` from VoxAtlas structured inputs. It consumes ``phoneme`` units and produces values aligned to ``phoneme`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor maps phoneme labels to articulatory classes using the phonology resource tables bundled with VoxAtlas.
    
    1. Resource lookup
       Each aligned phoneme label is normalized to IPA-like form and matched against the articulatory feature inventory.
    
    2. Class projection
       The output is a binary or categorical indicator, typically representable as :math:`x_i = \mathbf{1}[\mathrm{phoneme}_i \in C]` for a class :math:`C` such as vowels, nasals, or plosives.
    
    3. Packaging
       The resulting phoneme-aligned values can then be aggregated into rhythm or segmental summaries.
    
    Examples
    --------
        from voxatlas.features.phonology.articulatory.features import ArticulatoryFeaturesExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = ArticulatoryFeaturesExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "phonology.articulatory.features"
    input_units = "phoneme"
    output_units = "phoneme"
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
            Structured output aligned to the ``phoneme`` unit level when applicable.
        
        Examples
        --------
            extractor = ArticulatoryFeaturesExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires phoneme units")

        phonemes = feature_input.units.get("phoneme").copy()
        resources = load_phonology_resources(
            language=params.get("language"),
            resource_root=params.get("resource_root"),
        )

        rows = []
        for _, row in phonemes.iterrows():
            ipa, features = lookup_articulatory_features(row.get("label"), resources)
            if features is None:
                log_unknown_phoneme(
                    row.get("label"),
                    ipa,
                    params.get("language"),
                    feature_input.context,
                )
                features = {column: np.nan for column in FEATURE_COLUMNS}
                features["ipa"] = ipa

            rows.append(features)

        feature_table = pd.concat(
            [
                phonemes.reset_index(drop=True),
                pd.DataFrame(rows),
            ],
            axis=1,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="phoneme",
            values=feature_table,
        )


registry.register(ArticulatoryFeaturesExtractor)
