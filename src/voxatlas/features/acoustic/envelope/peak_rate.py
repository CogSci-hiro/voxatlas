import numpy as np

from voxatlas.acoustic.envelope_utils import compute_derivative, compute_peak_rate, detect_peaks
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


TRANSFORM_BASES = {
    "rms": "acoustic.envelope.rms",
    "log_energy": "acoustic.envelope.log_energy",
    "praat_intensity": "acoustic.envelope.praat_intensity",
    "hilbert": "acoustic.envelope.hilbert",
    "oganian": "acoustic.envelope.oganian",
    "varnet": "acoustic.envelope.varnet",
}


def _make_peak_rate_extractor(base_key, dependency_name):
    class PeakRateExtractor(BaseExtractor):
        r"""
        Extract a frame-level peak-rate contour from an upstream envelope representation.

        This public extractor measures how densely sharp positive envelope changes occur over time. Within the VoxAtlas feature graph it is a reusable event-density representation derived from a dependency contour such as RMS, log energy, Hilbert envelope, or Praat intensity.

        Algorithm
        ---------
        The peak-rate computation proceeds in three explicit stages.

        1. Differencing
           The upstream contour :math:`x_t` is converted to a derivative

           .. math::

              d_t = x_t - x_{t-1}.

        2. Peak detection
           Local maxima of :math:`d_t` are retained only when they exceed the configured threshold :math:`\theta`. These accepted peaks form the event set :math:`P`.

        3. Rate encoding
           The returned contour is an impulse-like rate series

           .. math::

              y_t = f_{\mathrm{frame}}\mathbf{1}[t \in P],

           where :math:`f_{\mathrm{frame}}` is the effective frame sampling rate inferred from the dependency time axis.

        Notes
        -----
        This extractor depends on a single upstream envelope contour and returns a ``VectorFeatureOutput`` aligned to ``frame`` units.

        Attributes
        ----------
        name : str
            Registry key for this extractor. This is derived from the chosen
            dependency and has the form ``"{dependency}.peak_rate"`` (for
            example, ``"acoustic.envelope.oganian.peak_rate"``).
        input_units : str | None
            Required input unit level. ``None`` means this extractor does not
            require linguistic units and instead consumes dependency outputs
            from the feature store.
        output_units : str | None
            Output alignment unit (``"frame"``).
        dependencies : list[str]
            Exactly one upstream contour (``[dependency_name]``), such as
            ``"acoustic.envelope.oganian"`` for ``OganianPeakRate``.
        default_config : dict
            Default runtime parameters:
            ``frame_length=0.025``, ``frame_step=0.01``,
            ``peak_threshold=0.1``, ``smoothing=1``.

        Examples
        --------
            from voxatlas.features.acoustic.envelope.peak_rate import OganianPeakRate
            from voxatlas.features.feature_input import FeatureInput

            # Assumes the upstream dependency (``acoustic.envelope.oganian``)
            # has already been computed and is available in the feature store.
            extractor = OganianPeakRate()
            feature_input = FeatureInput(audio=audio, units=units, context={"feature_store": feature_store})
            output = extractor.compute(feature_input, {})
            print(output)
        """
        name = f"{dependency_name}.peak_rate"
        input_units = None
        output_units = "frame"
        dependencies = [dependency_name]
        default_config = {
            "frame_length": 0.025,
            "frame_step": 0.010,
            "smoothing": 1,
            "peak_threshold": 0.1,
        }

        def compute(self, feature_input, params):
            """
            Compute the `feature` output for a single stream.
            
            This method is called by the pipeline after dependency resolution has completed. It receives the prepared feature input object together with the resolved feature-specific configuration.
            
            Parameters
            ----------
            feature_input : FeatureInput
                Container with audio, unit tables, and pipeline context.
            params : dict
                Resolved configuration dictionary for the extractor.
            
            Returns
            -------
            object
                Structured VoxAtlas feature output.
            
            Raises
            ------
            ValueError
                Raised when required inputs are unavailable for the feature.
            
            Notes
            -----
            Implementations should remain side-effect free and should read dependency outputs from ``feature_input.context['feature_store']`` when needed.
            
            Examples
            --------
            Usage example::
            
                extractor = type(self)()
                output = extractor.compute(feature_input, params)
                print(output)
            """
            base_output = feature_input.context["feature_store"].get(dependency_name)
            derivative = compute_derivative(base_output.values)
            peak_indices = detect_peaks(derivative, params.get("peak_threshold", 0.1))

            if len(base_output.time) > 1:
                frame_sr = 1.0 / float(base_output.time[1] - base_output.time[0])
            else:
                frame_sr = 1.0 / float(params.get("frame_step", 0.010))

            values = compute_peak_rate(peak_indices, frame_sr, len(base_output.values))

            return VectorFeatureOutput(
                feature=self.name,
                unit=base_output.unit,
                time=np.asarray(base_output.time, dtype=np.float32),
                values=np.asarray(values, dtype=np.float32),
            )

    PeakRateExtractor.__name__ = f"{base_key.title().replace('_', '')}PeakRate"
    return PeakRateExtractor


for _base_key, _dependency_name in TRANSFORM_BASES.items():
    _extractor_cls = _make_peak_rate_extractor(_base_key, _dependency_name)
    globals()[_extractor_cls.__name__] = _extractor_cls
    registry.register(_extractor_cls)
