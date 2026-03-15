import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.registry.feature_registry import registry


class WordFrequencyExtractor(BaseExtractor):
    r"""
    Extract the ``lexical.frequency.word`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``lexical.frequency.word`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor follows the standard VoxAtlas feature-computation pattern.
    
    1. Input preparation
       Structured audio, unit tables, and dependency outputs are gathered from ``feature_input``.
    
    2. Feature-specific computation
       The implementation applies the domain-specific transformation required by this extractor.
    
    3. Packaging
       Results are aligned to ``token`` units and returned as a ``FeatureOutput`` object.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['lexical.frequency.lookup'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
        from voxatlas.features.lexical.frequency.word_frequency import WordFrequencyExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = WordFrequencyExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """

    name = "lexical.frequency.word"
    input_units = "token"
    output_units = "token"
    dependencies = ["lexical.frequency.lookup"]
    default_config = {}

    def compute(self, feature_input, params):
        """
        Compute raw token-level frequency values from the lookup table.

        Parameters
        ----------
        feature_input : FeatureInput
            Prepared stream input containing the feature store.
        params : dict
            Resolved extractor configuration. Present for API consistency.

        Returns
        -------
        ScalarFeatureOutput
            Token-aligned raw frequency values.

        Raises
        ------
        KeyError
            Raised when the lexical lookup dependency is unavailable.

        Notes
        -----
        The output index matches the token ids from the lookup dependency.

        Examples
        --------
        Usage example::

            output = WordFrequencyExtractor().compute(feature_input, params)
            print(output.values.head())
        """
        table = feature_input.context["feature_store"].get(
            "lexical.frequency.lookup"
        ).values
        values = pd.Series(
            table["frequency"].astype("float32").values,
            index=table["id"],
        )

        return ScalarFeatureOutput(
            feature=self.name,
            unit="token",
            values=values,
        )


registry.register(WordFrequencyExtractor)
