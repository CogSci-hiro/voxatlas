import pandas as pd

from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.lexical.frequency_utils import load_frequency_lexicon, lookup_word_frequency
from voxatlas.registry.feature_registry import registry


class LexicalFrequencyLookupExtractor(BaseExtractor):
    r"""
    Extract the ``lexical.frequency.lookup`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``lexical.frequency.lookup`` from VoxAtlas structured inputs. It consumes ``token`` units and produces values aligned to ``token`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor performs lemma-based frequency lookup for token units.
    
    1. Lexicon retrieval
       A language-specific frequency lexicon is loaded into memory and converted to a lookup dictionary keyed by lemma-like forms.
    
    2. Token normalization
       Each token row contributes the first available lemma/form/text field used for lexicon matching.
    
    3. Lookup
       The output value is
    
       .. math::
    
          x_i = L(w_i),
    
       where :math:`L` maps a token lemma :math:`w_i` to its raw corpus frequency.
    
    4. Packaging
       Values are returned as a token-aligned scalar series so later lexical statistics can operate on a common index.
    
    Examples
    --------
        from voxatlas.features.lexical.frequency.lookup import LexicalFrequencyLookupExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = LexicalFrequencyLookupExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "lexical.frequency.lookup"
    input_units = "token"
    output_units = "token"
    dependencies = []
    default_config = {
        "language": None,
        "resource_root": None,
        "lexicon": None,
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
            extractor = LexicalFrequencyLookupExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.units is None:
            raise ValueError(f"{self.name} requires token units")

        try:
            tokens = feature_input.units.get("token").copy()
        except Exception as exc:
            raise ValueError(f"{self.name} requires token units") from exc

        lexicon = load_frequency_lexicon(
            language=params.get("language"),
            resource_root=params.get("resource_root"),
            lexicon=params.get("lexicon"),
        )
        frequencies = lookup_word_frequency(tokens, lexicon)

        feature_table = pd.DataFrame(
            {
                "id": tokens.get("id", pd.Series(tokens.index)),
                "frequency": frequencies.values,
            }
        )
        merged = pd.concat(
            [
                tokens.reset_index(drop=True),
                feature_table.drop(columns=["id"], errors="ignore"),
            ],
            axis=1,
        )

        return TableFeatureOutput(
            feature=self.name,
            unit="token",
            values=merged,
        )


registry.register(LexicalFrequencyLookupExtractor)
