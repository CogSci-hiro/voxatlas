import numpy as np

from voxatlas.acoustic.spectrogram_utils import compute_stft
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import MatrixFeatureOutput
from voxatlas.registry.feature_registry import registry


class STFTSpectrogramExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectrogram.stft`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectrogram.stft`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor exposes the short-time Fourier transform magnitude as a matrix-valued public API.
    
    1. Windowed analysis
       Frames are extracted from the waveform according to the configured frame size and hop.
    
    2. Fourier transform
       For each frame, the discrete Fourier transform is computed and stored as
    
       .. math::
    
          X_t(k) = \sum_{n=0}^{N-1} x_t[n]e^{-j2\pi kn/N}.
    
    3. Magnitude packaging
       VoxAtlas stores :math:`|X_t(k)|` together with explicit time and frequency axes so later features can be defined transparently.
    
    Examples
    --------
        from voxatlas.features.acoustic.spectrogram.stft import STFTSpectrogramExtractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = STFTSpectrogramExtractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """
    name = "acoustic.spectrogram.stft"
    input_units = None
    output_units = "frame"
    dependencies = []
    default_config = {
        "frame_length": 0.025,
        "frame_step": 0.010,
        "n_mels": 40,
        "fmin": 0.0,
        "fmax": None,
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
            Structured output aligned to the ``frame`` unit level when applicable.
        
        Examples
        --------
            extractor = STFTSpectrogramExtractor()
            feature_input = FeatureInput(audio=audio, units=units, context={})
            result = extractor.compute(feature_input, {})
            print(result)
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        time, frequency, values = compute_stft(
            feature_input.audio.waveform,
            feature_input.audio.sample_rate,
            frame_length=params["frame_length"],
            frame_step=params["frame_step"],
        )

        return MatrixFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            frequency=np.asarray(frequency, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(STFTSpectrogramExtractor)
