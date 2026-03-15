import numpy as np
import pandas as pd
import pytest

from voxatlas.audio.audio import Audio
from voxatlas.units.units import Units


@pytest.fixture
def dummy_audio():

    waveform = np.zeros(16000)

    return Audio(
        waveform=waveform,
        sample_rate=16000,
    )


@pytest.fixture
def dummy_units():

    frames = pd.DataFrame({
        "unit_id": range(10),
        "start": [i * 0.01 for i in range(10)],
        "end": [(i + 1) * 0.01 for i in range(10)],
    })

    return Units(frames=frames)