import numpy as np

from voxatlas.acoustic.envelope_utils import compute_derivative, detect_peaks
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


def _make_onset_extractor(base_key, dependency_name):
    class OnsetExtractor(BaseExtractor):
        r"""
        Extract a binary onset contour from an upstream envelope representation.

        This public extractor identifies rapid positive changes in a previously computed envelope contour and stores them as frame-level onset indicators. It gives the VoxAtlas pipeline a sparse event representation that can be quoted directly in timing analyses or aggregated into rates.

        Algorithm
        ---------
        The onset detector is intentionally transparent and follows the implementation literally.

        1. Dependency retrieval
           The base contour :math:`x_t` is loaded from the feature store.

        2. Positive-change detection
           VoxAtlas computes the first difference

           .. math::

              d_t = x_t - x_{t-1},

           and identifies local maxima above a threshold :math:`\theta`.

        3. Binary encoding
           The onset contour is then

           .. math::

              o_t = \mathbf{1}[t \in P],

           where :math:`P` is the set of accepted derivative peaks.

        Notes
        -----
        This extractor depends on one upstream envelope contour and returns a sparse ``VectorFeatureOutput`` aligned to ``frame`` units.

        Attributes
        ----------
        name : str
            Registry key for this extractor. This is derived from the chosen
            dependency and has the form ``"{dependency}.onset"`` (for example,
            ``"acoustic.envelope.oganian.onset"``).
        input_units : str | None
            Required input unit level. ``None`` means this extractor does not
            require linguistic units and instead consumes dependency outputs
            from the feature store.
        output_units : str | None
            Output alignment unit (``"frame"``).
        dependencies : list[str]
            Exactly one upstream contour (``[dependency_name]``), such as
            ``"acoustic.envelope.oganian"`` for ``OganianOnset``.
        default_config : dict
            Default runtime parameters:
            ``frame_length=0.025``, ``frame_step=0.01``,
            ``peak_threshold=0.1``, ``smoothing=1``.

        Examples
        --------
            from voxatlas.features.acoustic.envelope.onset import OganianOnset
            from voxatlas.features.feature_input import FeatureInput

            # Assumes the upstream dependency (``acoustic.envelope.oganian``)
            # has already been computed and is available in the feature store.
            extractor = OganianOnset()
            feature_input = FeatureInput(audio=audio, units=units, context={"feature_store": feature_store})
            output = extractor.compute(feature_input, {})
            print(output)
        """
        name = f"{dependency_name}.onset"
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
            values = np.zeros_like(derivative, dtype=np.float32)
            values[peak_indices] = 1.0

            return VectorFeatureOutput(
                feature=self.name,
                unit=base_output.unit,
                time=np.asarray(base_output.time, dtype=np.float32),
                values=values,
            )

    OnsetExtractor.__name__ = f"{base_key.title().replace('_', '')}Onset"
    return OnsetExtractor


for _base_key, _dependency_name in TRANSFORM_BASES.items():
    _extractor_cls = _make_onset_extractor(_base_key, _dependency_name)
    globals()[_extractor_cls.__name__] = _extractor_cls
    registry.register(_extractor_cls)
