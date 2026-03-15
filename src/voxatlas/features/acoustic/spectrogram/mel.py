import numpy as np

from voxatlas.acoustic.spectrogram_utils import compute_mel_filterbank, compute_mel_spectrogram
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import MatrixFeatureOutput
from voxatlas.registry.feature_registry import registry


class MelSpectrogramExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectrogram.mel`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectrogram.mel`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor projects an STFT magnitude spectrum onto a mel-spaced filterbank.
    
    1. Spectrum dependency
       The upstream STFT provides :math:`S_{t,k}` values and frequency bins.
    
    2. Filterbank projection
       Each mel channel is computed as
    
       .. math::
    
          M_{t,m} = \sum_k H_{m,k} S_{t,k},
    
       where :math:`H` is the triangular mel filterbank.
    
    3. Matrix packaging
       The mel spectrogram is returned as a matrix-valued feature aligned to the original frame grid.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.spectrogram.stft'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.audio.audio import Audio
    >>> from voxatlas.features.acoustic.spectrogram.mel import MelSpectrogramExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import MatrixFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> audio = Audio(waveform=np.zeros(800, dtype=np.float32), sample_rate=8000)
    >>> store = FeatureStore()
    >>> stft_out = MatrixFeatureOutput(
    ...     feature="acoustic.spectrogram.stft",
    ...     unit="frame",
    ...     time=np.array([0.0, 0.01], dtype=np.float32),
    ...     frequency=np.array([0.0, 2000.0, 4000.0], dtype=np.float32),
    ...     values=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    ... )
    >>> store.add("acoustic.spectrogram.stft", stft_out)
    >>> feature_input = FeatureInput(audio=audio, units=None, context={"feature_store": store})
    >>> params = MelSpectrogramExtractor.default_config.copy()
    >>> params["n_mels"] = 2
    >>> out = MelSpectrogramExtractor().compute(feature_input, params)
    >>> out.values.shape
    (2, 2)
    """
    name = "acoustic.spectrogram.mel"
    input_units = None
    output_units = "frame"
    dependencies = ["acoustic.spectrogram.stft"]
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
        >>> import numpy as np
        >>> from voxatlas.audio.audio import Audio
        >>> from voxatlas.features.acoustic.spectrogram.mel import MelSpectrogramExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import MatrixFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> audio = Audio(waveform=np.zeros(800, dtype=np.float32), sample_rate=8000)
        >>> store = FeatureStore()
        >>> stft_out = MatrixFeatureOutput(
        ...     feature="acoustic.spectrogram.stft",
        ...     unit="frame",
        ...     time=np.array([0.0], dtype=np.float32),
        ...     frequency=np.array([0.0, 2000.0, 4000.0], dtype=np.float32),
        ...     values=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        ... )
        >>> store.add("acoustic.spectrogram.stft", stft_out)
        >>> feature_input = FeatureInput(audio=audio, units=None, context={"feature_store": store})
        >>> params = MelSpectrogramExtractor.default_config.copy()
        >>> params["n_mels"] = 2
        >>> result = MelSpectrogramExtractor().compute(feature_input, params)
        >>> result.unit
        'frame'
        """
        stft_output = feature_input.context["feature_store"].get(
            "acoustic.spectrogram.stft"
        )

        n_fft = max(2, 2 * (len(stft_output.frequency) - 1))
        mel_filterbank, mel_frequencies = compute_mel_filterbank(
            sr=feature_input.audio.sample_rate,
            n_fft=n_fft,
            n_mels=params["n_mels"],
            fmin=params.get("fmin", 0.0),
            fmax=params.get("fmax"),
        )
        values = compute_mel_spectrogram(
            stft_output.values,
            mel_filterbank,
        )

        return MatrixFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(stft_output.time, dtype=np.float32),
            frequency=np.asarray(mel_frequencies, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(MelSpectrogramExtractor)
