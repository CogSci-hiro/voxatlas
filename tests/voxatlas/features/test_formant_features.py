import numpy as np
import pandas as pd

from voxatlas.audio.audio import Audio
from voxatlas.features.feature_input import FeatureInput
from voxatlas.features.feature_output import TableFeatureOutput
from voxatlas.features.phonology.formant.centralization import (
    FormantCentralizationExtractor,
)
from voxatlas.features.phonology.formant.dct import FormantDCTExtractor
from voxatlas.features.phonology.formant.mean import FormantMeanExtractor
from voxatlas.features.phonology.formant.median import FormantMedianExtractor
from voxatlas.features.phonology.formant.midpoint import FormantMidpointExtractor
from voxatlas.features.phonology.formant.onset_mid_offset import (
    FormantOnsetMidOffsetExtractor,
)
from voxatlas.features.phonology.formant.polynomial import (
    FormantPolynomialExtractor,
)
from voxatlas.features.phonology.formant.slope import FormantSlopeExtractor
from voxatlas.features.phonology.formant.tracks import FormantTracksExtractor
from voxatlas.features.phonology.formant.trajectory import (
    FormantTrajectoryExtractor,
)
from voxatlas.features.phonology.formant.tvsa import FormantTVSAExtractor
from voxatlas.features.phonology.formant.vai import FormantVAIExtractor
from voxatlas.features.phonology.formant.variance import FormantVarianceExtractor
from voxatlas.features.phonology.formant.vsa import FormantVSAExtractor
from voxatlas.pipeline.feature_store import FeatureStore
from voxatlas.units.units import Units


def test_formant_tracks_extractor_returns_frame_table():
    sr = 16000
    time = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False, dtype=np.float32)
    signal = 0.2 * np.sin(2 * np.pi * 220 * time).astype(np.float32)
    phonemes = pd.DataFrame(
        [
            {"id": 1, "start": 0.0, "end": 0.1, "label": "a"},
            {"id": 2, "start": 0.1, "end": 0.2, "label": "s"},
        ]
    )
    feature_input = FeatureInput(
        audio=Audio(waveform=signal, sample_rate=sr),
        units=Units(phonemes=phonemes, speaker="A"),
        context={},
    )

    output = FormantTracksExtractor().compute(
        feature_input,
        {
            "language": None,
            "resource_root": None,
            "frame_length": 0.025,
            "frame_step": 0.010,
            "lpc_order": 10,
            "max_formant": 5500.0,
            "use_parselmouth": False,
        },
    )

    assert output.unit == "frame"
    assert {"phoneme_id", "time", "is_vowel", "F1", "F2", "F3"}.issubset(output.values.columns)
    assert set(output.values["phoneme_id"]) == {1, 2}
    assert output.values.loc[output.values["phoneme_id"] == 1, "is_vowel"].eq(1.0).all()
    assert output.values.loc[output.values["phoneme_id"] == 2, "F1"].isna().all()


def test_formant_derived_extractors_run_from_tracks_table():
    tracks = pd.DataFrame(
        [
            {"frame_id": 1, "start": 0.00, "end": 0.03, "time": 0.015, "phoneme_id": 10, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 300.0, "F2": 2300.0, "F3": 3000.0},
            {"frame_id": 2, "start": 0.03, "end": 0.06, "time": 0.045, "phoneme_id": 10, "label": "i", "ipa": "i", "is_vowel": 1.0, "F1": 320.0, "F2": 2250.0, "F3": 2980.0},
            {"frame_id": 1, "start": 0.10, "end": 0.13, "time": 0.115, "phoneme_id": 11, "label": "a", "ipa": "a", "is_vowel": 1.0, "F1": 800.0, "F2": 1400.0, "F3": 2600.0},
            {"frame_id": 2, "start": 0.13, "end": 0.16, "time": 0.145, "phoneme_id": 11, "label": "a", "ipa": "a", "is_vowel": 1.0, "F1": 820.0, "F2": 1350.0, "F3": 2580.0},
            {"frame_id": 1, "start": 0.20, "end": 0.23, "time": 0.215, "phoneme_id": 12, "label": "u", "ipa": "u", "is_vowel": 1.0, "F1": 350.0, "F2": 900.0, "F3": 2400.0},
            {"frame_id": 2, "start": 0.23, "end": 0.26, "time": 0.245, "phoneme_id": 12, "label": "u", "ipa": "u", "is_vowel": 1.0, "F1": 360.0, "F2": 880.0, "F3": 2380.0},
        ]
    )
    store = FeatureStore()
    store.add(
        "phonology.formant.tracks",
        TableFeatureOutput(
            feature="phonology.formant.tracks",
            unit="frame",
            values=tracks,
        ),
    )
    feature_input = FeatureInput(
        audio=None,
        units=Units(
            phonemes=pd.DataFrame(
                [
                    {"id": 10, "start": 0.0, "end": 0.06, "label": "i"},
                    {"id": 11, "start": 0.1, "end": 0.16, "label": "a"},
                    {"id": 12, "start": 0.2, "end": 0.26, "label": "u"},
                ]
            ),
            speaker="A",
        ),
        context={"feature_store": store},
    )

    table_extractors = [
        FormantMidpointExtractor(),
        FormantTrajectoryExtractor(),
        FormantMeanExtractor(),
        FormantMedianExtractor(),
        FormantVarianceExtractor(),
        FormantOnsetMidOffsetExtractor(),
        FormantPolynomialExtractor(),
        FormantDCTExtractor(),
        FormantSlopeExtractor(),
    ]
    scalar_extractors = [
        FormantVSAExtractor(),
        FormantTVSAExtractor(),
        FormantVAIExtractor(),
        FormantCentralizationExtractor(),
    ]

    for extractor in table_extractors:
        output = extractor.compute(feature_input, extractor.default_config)
        assert output.unit == "phoneme"
        assert len(output.values) == 3

    for extractor in scalar_extractors:
        output = extractor.compute(feature_input, extractor.default_config)
        assert output.unit == "conversation"
        assert len(output.values) == 1
        assert np.isfinite(float(output.values.iloc[0]))
