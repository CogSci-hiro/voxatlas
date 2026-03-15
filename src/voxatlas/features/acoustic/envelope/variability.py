import numpy as np

from voxatlas.acoustic.envelope_utils import compute_variability
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


def _make_variability_extractor(base_key, dependency_name):
    class VariabilityExtractor(BaseExtractor):
        r"""
        Extract a frame-aligned variability summary from an upstream envelope contour.

        This public extractor turns a dependency contour into a dispersion statistic that can be attached to every frame. In VoxAtlas terms, it makes a global summary of envelope variation available as a reusable pipeline node rather than forcing each downstream feature to recompute it.

        Algorithm
        ---------
        The variability computation is defined directly on the upstream contour.

        1. Dependency retrieval
           The base contour :math:`x_t` is loaded from the feature store using the configured envelope dependency.

        2. Dispersion estimate
           VoxAtlas computes the population standard deviation

           .. math::

              \sigma_x = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(x_t-\bar{x})^2}.

        3. Broadcasting and packaging
           The scalar value :math:`\sigma_x` is repeated across the original frame grid so the feature remains directly alignable with other frame-level contours.

        Notes
        -----
        This extractor depends on one upstream envelope representation and returns a ``VectorFeatureOutput`` on ``frame`` units.

        Attributes
        ----------
        name : str
            Registry key for this extractor. This is derived from the chosen
            dependency and has the form ``"{dependency}.variability"`` (for
            example, ``"acoustic.envelope.oganian.variability"``).
        input_units : str | None
            Required input unit level. ``None`` means this extractor does not
            require linguistic units and instead consumes dependency outputs
            from the feature store.
        output_units : str | None
            Output alignment unit (``"frame"``).
        dependencies : list[str]
            Exactly one upstream contour (``[dependency_name]``), such as
            ``"acoustic.envelope.oganian"`` for ``OganianVariability``.
        default_config : dict
            Default runtime parameters:
            ``frame_length=0.025``, ``frame_step=0.01``,
            ``peak_threshold=0.1``, ``smoothing=1``.

        Examples
        --------
            from voxatlas.features.acoustic.envelope.variability import OganianVariability
            from voxatlas.features.feature_input import FeatureInput

            # Assumes the upstream dependency (``acoustic.envelope.oganian``)
            # has already been computed and is available in the feature store.
            extractor = OganianVariability()
            feature_input = FeatureInput(audio=audio, units=units, context={"feature_store": feature_store})
            output = extractor.compute(feature_input, {})
            print(output)
        """
        name = f"{dependency_name}.variability"
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
            values = compute_variability(base_output.values)

            return VectorFeatureOutput(
                feature=self.name,
                unit=base_output.unit,
                time=np.asarray(base_output.time, dtype=np.float32),
                values=np.asarray(values, dtype=np.float32),
            )

    VariabilityExtractor.__name__ = f"{base_key.title().replace('_', '')}Variability"
    return VariabilityExtractor


for _base_key, _dependency_name in TRANSFORM_BASES.items():
    _extractor_cls = _make_variability_extractor(_base_key, _dependency_name)
    globals()[_extractor_cls.__name__] = _extractor_cls
    registry.register(_extractor_cls)
