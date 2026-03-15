import numpy as np

from voxatlas.acoustic.envelope_utils import compute_derivative
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


def _make_derivative_extractor(base_key, dependency_name):
    class DerivativeExtractor(BaseExtractor):
        r"""
        Extract a frame-level derivative contour from an upstream envelope feature.

        This public extractor converts an already computed envelope-like contour
        into its first temporal difference. In the VoxAtlas pipeline, it is used
        to expose local change information that can then be reused by onset and
        peak-rate features without recomputing the base envelope.

        Algorithm
        ---------
        The implementation is intentionally simple and closely mirrors the code path.

        1. Dependency retrieval
           The extractor reads the upstream frame-aligned contour :math:`x_t`
           from the feature store. The exact source depends on
           ``dependency_name`` and may be RMS, log energy, Hilbert envelope,
           Praat intensity, or another envelope representation.

        2. Finite differencing
           VoxAtlas applies the backward difference

           .. math::

              d_t = x_t - x_{t-1},

           while preserving the original frame count by prepending the first sample.

        3. Alignment and packaging
           The derivative values remain aligned to the dependency time axis so
           later stages can quote and aggregate the contour without any resampling.

        Notes
        -----
        This extractor depends on exactly one upstream envelope feature and returns
        a ``VectorFeatureOutput`` aligned to ``frame`` units.

        Attributes
        ----------
        name : str
            Registry key for this extractor. This is derived from the chosen
            dependency and has the form ``"{dependency}.derivative"`` (for
            example, ``"acoustic.envelope.oganian.derivative"``).
        input_units : str | None
            Required input unit level. ``None`` means this extractor does not
            require linguistic units and instead consumes dependency outputs
            from the feature store.
        output_units : str | None
            Output alignment unit (``"frame"``).
        dependencies : list[str]
            Exactly one upstream contour (``[dependency_name]``), such as
            ``"acoustic.envelope.oganian"`` for ``OganianDerivative``.
        default_config : dict
            Default runtime parameters:
            ``frame_length=0.025``, ``frame_step=0.01``,
            ``peak_threshold=0.1``, ``smoothing=1``.

        Examples
        --------
        >>> import numpy as np
        >>> from voxatlas.features.acoustic.envelope.derivative import OganianDerivative
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import VectorFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> store = FeatureStore()
        >>> base = VectorFeatureOutput(
        ...     feature="acoustic.envelope.oganian",
        ...     unit="frame",
        ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
        ...     values=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        ... )
        >>> store.add("acoustic.envelope.oganian", base)
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> out = OganianDerivative().compute(feature_input, {})
        >>> out.values.tolist()
        [0.0, 1.0, 0.0]
        """

        name: str = f"{dependency_name}.derivative"
        input_units: str | None = None
        output_units: str | None = "frame"
        dependencies: list[str] = [dependency_name]
        default_config: dict = {
            "frame_length": 0.025,
            "frame_step": 0.010,
            "smoothing": 1,
            "peak_threshold": 0.1,
        }

        def compute(self, feature_input, params):
            """
            Compute the `feature` output for a single stream.

            This method is called by the pipeline after dependency resolution has
            completed. It receives the prepared feature input object together with
            the resolved feature-specific configuration.

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
            Implementations should remain side-effect free and should read dependency
            outputs from ``feature_input.context['feature_store']`` when needed.

            Examples
            --------
            >>> import numpy as np
            >>> from voxatlas.features.acoustic.envelope.derivative import LogEnergyDerivative
            >>> from voxatlas.features.feature_input import FeatureInput
            >>> from voxatlas.features.feature_output import VectorFeatureOutput
            >>> from voxatlas.pipeline.feature_store import FeatureStore
            >>> store = FeatureStore()
            >>> base = VectorFeatureOutput(
            ...     feature="acoustic.envelope.log_energy",
            ...     unit="frame",
            ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
            ...     values=np.array([0.0, 0.0, 2.0], dtype=np.float32),
            ... )
            >>> store.add("acoustic.envelope.log_energy", base)
            >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
            >>> LogEnergyDerivative().compute(feature_input, {}).values.tolist()
            [0.0, 0.0, 2.0]

            >>> import numpy as np
            >>> from voxatlas.features.acoustic.envelope.derivative import PraatIntensityDerivative
            >>> from voxatlas.features.feature_input import FeatureInput
            >>> from voxatlas.features.feature_output import VectorFeatureOutput
            >>> from voxatlas.pipeline.feature_store import FeatureStore
            >>> store = FeatureStore()
            >>> base = VectorFeatureOutput(
            ...     feature="acoustic.envelope.praat_intensity",
            ...     unit="frame",
            ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
            ...     values=np.array([1.0, 3.0, 6.0], dtype=np.float32),
            ... )
            >>> store.add("acoustic.envelope.praat_intensity", base)
            >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
            >>> PraatIntensityDerivative().compute(feature_input, {}).values.tolist()
            [0.0, 2.0, 3.0]

            >>> import numpy as np
            >>> from voxatlas.features.acoustic.envelope.derivative import RmsDerivative
            >>> from voxatlas.features.feature_input import FeatureInput
            >>> from voxatlas.features.feature_output import VectorFeatureOutput
            >>> from voxatlas.pipeline.feature_store import FeatureStore
            >>> store = FeatureStore()
            >>> base = VectorFeatureOutput(
            ...     feature="acoustic.envelope.rms",
            ...     unit="frame",
            ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
            ...     values=np.array([0.5, 0.5, 0.0], dtype=np.float32),
            ... )
            >>> store.add("acoustic.envelope.rms", base)
            >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
            >>> RmsDerivative().compute(feature_input, {}).values.tolist()
            [0.0, 0.0, -0.5]
            """
            base_output = feature_input.context["feature_store"].get(dependency_name)
            values = compute_derivative(base_output.values)

            return VectorFeatureOutput(
                feature=self.name,
                unit=base_output.unit,
                time=np.asarray(base_output.time, dtype=np.float32),
                values=np.asarray(values, dtype=np.float32),
            )

    DerivativeExtractor.__name__ = f"{base_key.title().replace('_', '')}Derivative"
    return DerivativeExtractor


for _base_key, _dependency_name in TRANSFORM_BASES.items():
    _extractor_cls = _make_derivative_extractor(_base_key, _dependency_name)
    globals()[_extractor_cls.__name__] = _extractor_cls
    registry.register(_extractor_cls)
