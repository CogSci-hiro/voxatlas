import numpy as np

from voxatlas.acoustic.spectral_utils import compute_spectrum
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import MatrixFeatureOutput
from voxatlas.registry.feature_registry import registry


class SpectrumExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectral.spectrum`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectral.spectrum`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor returns the frame-by-frequency magnitude spectrum computed from a short-time Fourier transform.
    
    1. STFT analysis
       The waveform is windowed and transformed framewise.
    
    2. Magnitude projection
       For frame :math:`t` and frequency bin :math:`k`, the stored value is
    
       .. math::
    
          S_{t,k} = |X_t(k)|.
    
    3. Matrix packaging
       The resulting matrix, together with time and frequency axes, is emitted as a ``MatrixFeatureOutput`` for downstream spectral descriptors.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.audio.audio import Audio
    >>> from voxatlas.features.acoustic.spectral.spectrum import SpectrumExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
    >>> feature_input = FeatureInput(audio=audio, units=None, context={})
    >>> params = SpectrumExtractor.default_config.copy()
    >>> out = SpectrumExtractor().compute(feature_input, params)
    >>> out.unit
    'frame'
    """
    name = "acoustic.spectral.spectrum"
    input_units = None
    output_units = "frame"
    dependencies = []
    default_config = {
        "frame_length": 0.025,
        "frame_step": 0.010,
        "window": "hann",
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
        >>> import numpy as np
        >>> from voxatlas.audio.audio import Audio
        >>> from voxatlas.features.acoustic.spectral.spectrum import SpectrumExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
        >>> feature_input = FeatureInput(audio=audio, units=None, context={})
        >>> params = SpectrumExtractor.default_config.copy()
        >>> result = SpectrumExtractor().compute(feature_input, params)
        >>> result.values.ndim
        2
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        time, frequency, values = compute_spectrum(
            feature_input.audio.waveform,
            feature_input.audio.sample_rate,
            frame_length=params["frame_length"],
            frame_step=params["frame_step"],
            window=params.get("window", "hann"),
        )

        return MatrixFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            frequency=np.asarray(frequency, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(SpectrumExtractor)
