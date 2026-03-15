import numpy as np

from voxatlas.acoustic.envelope_utils import compute_log_energy, compute_rms, smooth_signal
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class LogEnergyEnvelope(BaseExtractor):
    r"""
    Extract the ``acoustic.envelope.log_energy`` feature within the VoxAtlas pipeline.

    Computes a smoothed, frame-aligned log-energy contour from the waveform by
    applying a logarithmic transform to a non-negative RMS amplitude envelope.

    Algorithm
    ---------
    The implementation mirrors the code path.

    1. RMS amplitude
       The waveform is framed and converted to RMS values :math:`r_t \ge 0`.

    2. Log transform
       VoxAtlas computes

       .. math::

          e_t = \log(\max(r_t, \varepsilon)),

       where :math:`\varepsilon` is a small numerical floor.

    3. Smoothing
       The resulting contour is optionally smoothed with a moving-average window
       of length ``smoothing`` frames.

    Attributes
    ----------
    name : str
        Registry key for this extractor (``"acoustic.envelope.log_energy"``).
    input_units : str | None
        Required input unit level. ``None`` means this extractor operates
        directly on waveform audio.
    output_units : str | None
        Output alignment unit (``"frame"``).
    dependencies : list[str]
        Upstream features required before execution. Empty for this extractor.
    default_config : dict
        Default runtime parameters:
        ``frame_length=0.025``, ``frame_step=0.01``,
        ``peak_threshold=0.1``, ``smoothing=1``.

    References
    ----------
    Davis, S. B., & Mermelstein, P. (1980). Comparison of parametric
    representations for monosyllabic word recognition in continuously spoken
    sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing,
    28*(4), 357–366. https://doi.org/10.1109/TASSP.1980.1163420

    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.audio.audio import Audio
    >>> from voxatlas.features.acoustic.envelope.log_energy import LogEnergyEnvelope
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
    >>> feature_input = FeatureInput(audio=audio, units=None, context={})
    >>> params = LogEnergyEnvelope.default_config.copy()
    >>> out = LogEnergyEnvelope().compute(feature_input, params)
    >>> out.unit
    'frame'
    """

    name: str = "acoustic.envelope.log_energy"
    input_units: str | None = None
    output_units: str | None = "frame"
    dependencies: list[str] = []
    default_config: dict = {
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
        >>> import numpy as np
        >>> from voxatlas.audio.audio import Audio
        >>> from voxatlas.features.acoustic.envelope.log_energy import LogEnergyEnvelope
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> audio = Audio(waveform=np.zeros(1600, dtype=np.float32), sample_rate=16000)
        >>> feature_input = FeatureInput(audio=audio, units=None, context={})
        >>> params = LogEnergyEnvelope.default_config.copy()
        >>> out = LogEnergyEnvelope().compute(feature_input, params)
        >>> out.values.shape[0] > 0
        True
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
