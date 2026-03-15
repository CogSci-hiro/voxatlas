import numpy as np

from voxatlas.acoustic.spectral_utils import spectral_flux
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class SpectralFluxExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectral.flux`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectral.flux`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor measures inter-frame spectral change using Euclidean distance between adjacent spectra.
    
    1. Frame differencing
       Adjacent magnitude spectra are subtracted along the time axis.
    
    2. Norm computation
       Flux is defined as
    
       .. math::
    
          \mathrm{Flux}_t = \sqrt{\sum_k (S_{t,k} - S_{t-1,k})^2}.
    
    3. Packaging
       The distance value is returned on the frame grid used by the upstream spectrum feature.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.spectral.spectrum'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.features.acoustic.spectral.flux import SpectralFluxExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import MatrixFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> store = FeatureStore()
    >>> spectrum = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    >>> freq = np.array([0.0, 1.0], dtype=np.float32)
    >>> base = MatrixFeatureOutput(
    ...     feature="acoustic.spectral.spectrum",
    ...     unit="frame",
    ...     time=np.array([0.0, 0.01], dtype=np.float32),
    ...     frequency=freq,
    ...     values=spectrum,
    ... )
    >>> store.add("acoustic.spectral.spectrum", base)
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = SpectralFluxExtractor().compute(feature_input, {})
    >>> out.values.tolist()
    [0.0, 1.0]
    """
    name = "acoustic.spectral.flux"
    input_units = None
    output_units = "frame"
    dependencies = ["acoustic.spectral.spectrum"]
    default_config = {}

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
        >>> from voxatlas.features.acoustic.spectral.flux import SpectralFluxExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import MatrixFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> store = FeatureStore()
        >>> spectrum = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        >>> freq = np.array([0.0, 1.0], dtype=np.float32)
        >>> base = MatrixFeatureOutput(
        ...     feature="acoustic.spectral.spectrum",
        ...     unit="frame",
        ...     time=np.array([0.0, 0.01], dtype=np.float32),
        ...     frequency=freq,
        ...     values=spectrum,
        ... )
        >>> store.add("acoustic.spectral.spectrum", base)
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = SpectralFluxExtractor().compute(feature_input, {})
        >>> result.unit
        'frame'
        """
        spectrum_output = feature_input.context["feature_store"].get(
            "acoustic.spectral.spectrum"
        )
        values = spectral_flux(spectrum_output.values)

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(spectrum_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(SpectralFluxExtractor)
