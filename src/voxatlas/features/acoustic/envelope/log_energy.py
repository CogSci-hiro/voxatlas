import numpy as np

from voxatlas.acoustic.envelope_utils import compute_log_energy, compute_rms, smooth_signal
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class LogEnergyEnvelope(BaseExtractor):
    r"""
    Extract the ``acoustic.envelope.log_energy`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.envelope.log_energy`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor transforms the RMS envelope into a log-amplitude representation that compresses large dynamic-range differences.
    
    1. RMS dependency
       The upstream RMS contour supplies a non-negative amplitude :math:`r_t` at each frame.
    
    2. Log transform
       The implementation computes
    
       .. math::
    
          e_t = \log(\max(r_t, \varepsilon)),
    
       where :math:`\varepsilon` is a small numerical floor to avoid taking the logarithm of zero.
    
    3. Packaging
       The transformed contour remains frame-aligned and can be quoted directly as a log-energy envelope in downstream analyses.
    
    Examples
    --------
        from voxatlas.features.acoustic.envelope.log_energy import LogEnergyEnvelope
        from voxatlas.features.feature_input import FeatureInput
    
        extractor = LogEnergyEnvelope()
        feature_input = FeatureInput(audio=audio, units=units, context={})
        output = extractor.compute(feature_input, {})
        print(output)
    """

    name = "acoustic.envelope.log_energy"
    input_units = None
    output_units = "frame"
    dependencies = []
    default_config = {
        "frame_length": 0.025,
        "frame_step": 0.010,
        "smoothing": 1,
        "peak_threshold": 0.1,
    }

    def compute(self, feature_input, params):
        """
        Compute the log-energy contour for one stream.

        Parameters
        ----------
        feature_input : FeatureInput
            Prepared stream input containing audio and execution context.
        params : dict
            Resolved extractor configuration.

        Returns
        -------
        VectorFeatureOutput
            Frame-aligned log-energy contour.

        Raises
        ------
        ValueError
            Raised when audio input is unavailable.

        Notes
        -----
        Smoothing is applied after the log transform.

        Examples
        --------
        Usage example::

            output = LogEnergyEnvelope().compute(feature_input, params)
            print(output.values.shape)
        """
        if feature_input.audio is None:
            raise ValueError(f"{self.name} requires audio input")

        time, rms_values = compute_rms(
            feature_input.audio.waveform,
            feature_input.audio.sample_rate,
            params["frame_length"],
            params["frame_step"],
        )
        values = compute_log_energy(rms_values)
        values = smooth_signal(values, params.get("smoothing", 1))

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(LogEnergyEnvelope)
