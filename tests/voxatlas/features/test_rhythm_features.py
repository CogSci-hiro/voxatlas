import numpy as np
import pandas as pd

from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.feature_output import ScalarFeatureOutput, TableFeatureOutput
from voxatlas.features.phonology.rhythm.delta_c import RhythmDeltaCExtractor
from voxatlas.features.phonology.rhythm.delta_v import RhythmDeltaVExtractor
from voxatlas.features.phonology.rhythm.intervals import RhythmIntervalsExtractor
from voxatlas.features.phonology.rhythm.npvi import RhythmNPVIExtractor
from voxatlas.features.phonology.rhythm.pause_rate import RhythmPauseRateExtractor
from voxatlas.features.phonology.rhythm.percent_c import RhythmPercentCExtractor
from voxatlas.features.phonology.rhythm.percent_v import RhythmPercentVExtractor
from voxatlas.features.phonology.rhythm.syllable_duration import (
    RhythmSyllableDurationExtractor,
)
from voxatlas.features.phonology.rhythm.syllable_rate import RhythmSyllableRateExtractor
from voxatlas.features.phonology.rhythm.varco_c import RhythmVarcoCExtractor
from voxatlas.features.phonology.rhythm.varco_v import RhythmVarcoVExtractor
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.units.units import Units


def _units():
    phonemes = pd.DataFrame(
        [
            {"id": 1, "start": 0.00, "end": 0.10, "label": "k"},
            {"id": 2, "start": 0.10, "end": 0.20, "label": "a"},
            {"id": 3, "start": 0.20, "end": 0.28, "label": "t"},
            {"id": 4, "start": 0.28, "end": 0.36, "label": "i"},
            {"id": 5, "start": 0.50, "end": 0.60, "label": "m"},
            {"id": 6, "start": 0.60, "end": 0.72, "label": "u"},
        ]
    )
    syllables = pd.DataFrame(
        [
            {"id": 10, "start": 0.00, "end": 0.20, "label": "ka"},
            {"id": 11, "start": 0.20, "end": 0.36, "label": "ti"},
            {"id": 12, "start": 0.50, "end": 0.72, "label": "mu"},
        ]
    )
    ipus = pd.DataFrame(
        [
            {"id": 100, "start": 0.00, "end": 0.40, "label": "ipu1"},
            {"id": 101, "start": 0.50, "end": 0.80, "label": "ipu2"},
        ]
    )
    return Units(phonemes=phonemes, syllables=syllables, ipus=ipus, speaker="A")


def test_rhythm_intervals_and_syllable_metrics():
    units = _units()
    store = FeatureStore()
    store.add(
        "phonology.articulatory.vowel",
        ScalarFeatureOutput(
            feature="phonology.articulatory.vowel",
            unit="phoneme",
            values=pd.Series(
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                index=[1, 2, 3, 4, 5, 6],
                dtype="float32",
            ),
        ),
    )
    feature_input = FeatureInput(audio=None, units=units, context={"feature_store": store})

    intervals = RhythmIntervalsExtractor().compute(feature_input, {})
    store.add("phonology.rhythm.intervals", intervals)

    syllable_duration = RhythmSyllableDurationExtractor().compute(feature_input, {})
    syllable_rate = RhythmSyllableRateExtractor().compute(feature_input, {})
    pause_rate = RhythmPauseRateExtractor().compute(feature_input, {"pause_threshold": 0.05})

    assert intervals.unit == "ipu"
    assert list(intervals.values["type"]) == ["c", "v", "c", "v", "c", "v"]
    assert list(syllable_duration.values.index) == [10, 11, 12]
    assert np.allclose(syllable_duration.values.to_numpy(dtype=np.float32), [0.2, 0.16, 0.22])
    assert np.allclose(syllable_rate.values.loc[[100, 101]].to_numpy(dtype=np.float32), [5.0, 3.3333333], atol=1e-5)
    assert np.allclose(pause_rate.values.loc[[100, 101]].to_numpy(dtype=np.float32), [0.0, 0.0], atol=1e-6)


def test_rhythm_ipu_metrics_from_intervals():
    intervals = pd.DataFrame(
        [
            {"interval_id": 1, "ipu_id": 100, "type": "c", "start": 0.00, "end": 0.10, "duration": 0.10, "n_phonemes": 1},
            {"interval_id": 2, "ipu_id": 100, "type": "v", "start": 0.10, "end": 0.20, "duration": 0.10, "n_phonemes": 1},
            {"interval_id": 3, "ipu_id": 100, "type": "c", "start": 0.20, "end": 0.28, "duration": 0.08, "n_phonemes": 1},
            {"interval_id": 4, "ipu_id": 100, "type": "v", "start": 0.28, "end": 0.36, "duration": 0.08, "n_phonemes": 1},
            {"interval_id": 5, "ipu_id": 101, "type": "c", "start": 0.50, "end": 0.60, "duration": 0.10, "n_phonemes": 1},
            {"interval_id": 6, "ipu_id": 101, "type": "v", "start": 0.60, "end": 0.72, "duration": 0.12, "n_phonemes": 1},
        ]
    )
    store = FeatureStore()
    store.add(
        "phonology.rhythm.intervals",
        TableFeatureOutput(
            feature="phonology.rhythm.intervals",
            unit="ipu",
            values=intervals,
        ),
    )
    feature_input = FeatureInput(audio=None, units=_units(), context={"feature_store": store})

    extractors = [
        RhythmNPVIExtractor(),
        RhythmPercentVExtractor(),
        RhythmPercentCExtractor(),
        RhythmDeltaVExtractor(),
        RhythmDeltaCExtractor(),
        RhythmVarcoVExtractor(),
        RhythmVarcoCExtractor(),
    ]

    outputs = {extractor.name: extractor.compute(feature_input, {}) for extractor in extractors}

    assert np.allclose(outputs["phonology.rhythm.percent_v"].values.loc[[100, 101]], [50.0, 54.545454], atol=1e-5)
    assert np.allclose(outputs["phonology.rhythm.percent_c"].values.loc[[100, 101]], [50.0, 45.454545], atol=1e-5)
    assert np.isfinite(float(outputs["phonology.rhythm.npvi"].values.loc[100]))
    assert np.allclose(outputs["phonology.rhythm.delta_v"].values.loc[[100, 101]], [0.01, 0.0], atol=1e-5)
    assert np.allclose(outputs["phonology.rhythm.delta_c"].values.loc[[100, 101]], [0.01, 0.0], atol=1e-5)
    assert np.isfinite(float(outputs["phonology.rhythm.varco_v"].values.loc[100]))
    assert np.isfinite(float(outputs["phonology.rhythm.varco_c"].values.loc[100]))
