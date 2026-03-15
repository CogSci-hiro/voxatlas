import numpy as np
import pandas as pd

from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.lexical.properties.features import LexicalPropertiesExtractor
from voxatlas.features.lexical.properties.phoneme_count import PhonemeCountExtractor
from voxatlas.features.lexical.properties.syllable_count import SyllableCountExtractor
from voxatlas.features.lexical.properties.word_length import WordLengthExtractor
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.units.units import Units


def _units():
    tokens = pd.DataFrame(
        [
            {"id": 1, "start": 0.0, "end": 0.3, "token": "hello"},
            {"id": 2, "start": 0.3, "end": 0.6, "token": "cat"},
        ]
    )
    syllables = pd.DataFrame(
        [
            {"id": 10, "start": 0.0, "end": 0.15, "label": "he"},
            {"id": 11, "start": 0.15, "end": 0.3, "label": "llo"},
            {"id": 12, "start": 0.3, "end": 0.6, "label": "cat"},
        ]
    )
    phonemes = pd.DataFrame(
        [
            {"id": 20, "start": 0.00, "end": 0.08, "label": "h"},
            {"id": 21, "start": 0.08, "end": 0.15, "label": "ɛ"},
            {"id": 22, "start": 0.15, "end": 0.22, "label": "l"},
            {"id": 23, "start": 0.22, "end": 0.30, "label": "o"},
            {"id": 24, "start": 0.30, "end": 0.40, "label": "k"},
            {"id": 25, "start": 0.40, "end": 0.50, "label": "æ"},
            {"id": 26, "start": 0.50, "end": 0.60, "label": "t"},
        ]
    )
    return Units(tokens=tokens, syllables=syllables, phonemes=phonemes)


def test_lexical_properties_base_extractor_counts_from_alignments():
    feature_input = FeatureInput(audio=None, units=_units(), context={})
    output = LexicalPropertiesExtractor().compute(feature_input, {})

    assert list(output.values["word_length"]) == [5, 3]
    assert list(output.values["syllable_count"]) == [2, 1]
    assert list(output.values["phoneme_count"]) == [4, 3]


def test_lexical_properties_derived_extractors_read_base_table():
    feature_input = FeatureInput(
        audio=None,
        units=_units(),
        context={"feature_store": FeatureStore()},
    )
    base = LexicalPropertiesExtractor().compute(feature_input, {})
    feature_input.context["feature_store"].add("lexical.properties.features", base)

    word_length = WordLengthExtractor().compute(feature_input, {})
    syllable_count = SyllableCountExtractor().compute(feature_input, {})
    phoneme_count = PhonemeCountExtractor().compute(feature_input, {})

    assert list(word_length.values.index) == [1, 2]
    assert np.allclose(word_length.values.to_numpy(dtype=np.float32), [5.0, 3.0])
    assert np.allclose(syllable_count.values.to_numpy(dtype=np.float32), [2.0, 1.0])
    assert np.allclose(phoneme_count.values.to_numpy(dtype=np.float32), [4.0, 3.0])
