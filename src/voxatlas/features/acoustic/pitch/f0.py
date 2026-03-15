import numpy as np

from voxatlas.acoustic.pitch_utils import compute_f0
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class F0Extractor(BaseExtractor):
    r"""
    Extract the ``acoustic.pitch.f0`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.pitch.f0`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The implementation estimates a frame-level fundamental-frequency contour using autocorrelation over short analysis windows.
    
    1. Framing and centering
       The waveform is segmented into overlapping frames of length ``frame_length`` and hop ``frame_step``. Each frame is mean-centered before periodicity analysis.
    
    2. Period search
       For each frame, the code evaluates the one-sided autocorrelation function
    
       .. math::
    
          R(\tau) = \sum_{n=0}^{N-\tau-1} x[n]x[n+\tau],
    
       and restricts candidate lags to the interval implied by ``fmin`` and ``fmax``.
    
    3. Voicing decision
       The winning lag :math:`\tau^*` is accepted only when the normalized autocorrelation peak exceeds the implementation threshold. The output frequency is then
    
       .. math::
    
          \hat f_0 = \frac{f_s}{\tau^*}.
    
    4. Packaging
       Unvoiced or low-energy frames are set to ``NaN``, and the resulting contour is returned as a frame-aligned ``VectorFeatureOutput`` for downstream voice-quality and prosodic features.
    
    Examples
    --------
        from voxatlas.features.acoustic.pitch.f0 import F0Extractor
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = F0Extractor()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """

    name = "acoustic.pitch.f0"
    input_units = None
    output_units = "frame"
    dependencies = []
    default_config = {
        "fmin": 75.0,
        "fmax": 500.0,
        "frame_length": 0.040,
        "frame_step": 0.010,
    }

    def compute(self, feature_input, params):
        """
        Compute the frame-aligned F0 contour for one stream.

        Parameters
        ----------
        feature_input : FeatureInput
            Prepared stream input containing audio and execution context.
        params : dict
            Resolved extractor configuration.

        Returns
        -------
        VectorFeatureOutput
            Frame-aligned F0 contour.

        Raises
        ------
        ValueError
            Raised when audio input is unavailable.

        Notes
        -----
        The output time axis matches the analysis frames used during F0
        estimation.

        Examples
        --------
        Usage example::

            output = F0Extractor().compute(feature_input, params)
            print(output.time.shape, output.values.shape)
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        time, values = compute_f0(
            feature_input.audio.waveform,
            feature_input.audio.sample_rate,
            params["fmin"],
            params["fmax"],
            frame_length=params.get("frame_length", 0.040),
            frame_step=params.get("frame_step", 0.010),
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(F0Extractor)
