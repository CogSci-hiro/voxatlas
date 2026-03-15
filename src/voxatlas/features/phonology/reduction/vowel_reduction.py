import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import ScalarFeatureOutput
from voxatlas.phonology.reduction_utils import (
    align_phoneme_sequences,
    detect_vowel_reduction,
    get_expected_pronunciation,
    get_observed_pronunciation,
)
from voxatlas.registry.feature_registry import registry


class VowelReductionExtractor(BaseExtractor):
    r"""
    Extract the ``phonology.reduction.vowel_reduction`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``phonology.reduction.vowel_reduction`` from VoxAtlas structured inputs. It consumes ``word`` units and produces values aligned to ``word`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor compares expected and observed segmental realizations to quantify reduction phenomena.
    
    1. Canonical-observed alignment
       Expected phoneme sequences are aligned with observed phoneme labels from the annotated speech signal.
    
    2. Reduction scoring
       The output is a binary or scalar mismatch measure, typically of the form :math:`x_i = \mathbf{1}[\mathrm{expected}_i \ne \mathrm{observed}_i]` or a closely related normalized score.
    
    3. Packaging
       Values are returned on the requested phonological unit level for downstream aggregation.
    
    Examples
    --------
        from voxatlas.features.phonology.reduction.vowel_reduction import VowelReductionExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = VowelReductionExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "phonology.reduction.vowel_reduction"
    input_units = "word"
    output_units = "word"
    dependencies = []
    default_config = {
        "pronunciation_dictionary": None,
        "central_vowels": ["ə", "ɘ", "ɜ", "ɐ"],
        "schwa_symbols": ["ə", "@", "schwa"],
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
            Structured output aligned to the ``word`` unit level when applicable.
        
        Examples
        --------
            extractor = VowelReductionExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires word and phoneme units")

        words = feature_input.units.get("word")
        phonemes = feature_input.units.get("phoneme")
        pronunciation_dictionary = params.get("pronunciation_dictionary")

        values = []
        index = []

        for _, word_row in words.iterrows():
            expected = get_expected_pronunciation(word_row, pronunciation_dictionary)
            observed = get_observed_pronunciation(phonemes, word_row)
            alignment = align_phoneme_sequences(expected, observed)
            values.append(
                detect_vowel_reduction(
                    alignment,
                    central_vowels=params.get("central_vowels"),
                    schwa_symbols=params.get("schwa_symbols"),
                )
            )
            index.append(word_row.get("id", word_row.name))

        return ScalarFeatureOutput(
            feature=self.name,
            unit="word",
            values=pd.Series(values, index=index, dtype="float32"),
        )


registry.register(VowelReductionExtractor)
