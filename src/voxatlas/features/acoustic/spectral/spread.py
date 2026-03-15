import numpy as np

from voxatlas.acoustic.spectral_utils import spectral_spread
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class SpectralSpreadExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectral.spread`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectral.spread`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor measures second-moment dispersion of spectral energy around the centroid.
    
    1. Centroid preparation
       The spectrum and frequency axis are combined with either a supplied or internally computed centroid :math:`C_t`.
    
    2. Spread computation
       The spread is
    
       .. math::
    
          \mathrm{Spread}_t = \sqrt{\frac{\sum_k S_{t,k}(f_k-C_t)^2}{\sum_k S_{t,k}}}.
    
    3. Packaging
       The resulting contour remains aligned to the source spectrum frames.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.spectral.spectrum'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.features.acoustic.spectral.spread import SpectralSpreadExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import MatrixFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> store = FeatureStore()
    >>> spectrum = MatrixFeatureOutput(
    ...     feature="acoustic.spectral.spectrum",
    ...     unit="frame",
    ...     time=np.array([0.0, 0.01], dtype=np.float32),
    ...     frequency=np.array([0.0, 1000.0, 2000.0], dtype=np.float32),
    ...     values=np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    ... )
    >>> store.add("acoustic.spectral.spectrum", spectrum)
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = SpectralSpreadExtractor().compute(feature_input, {})
    >>> out.values.shape
    (2,)
    >>> float(out.values[0])
    0.0
    >>> round(float(out.values[1]), 1)
    816.5
    """
    name = "acoustic.spectral.spread"
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
        >>> from voxatlas.features.acoustic.spectral.spread import SpectralSpreadExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import MatrixFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> store = FeatureStore()
        >>> spectrum = MatrixFeatureOutput(
        ...     feature="acoustic.spectral.spectrum",
        ...     unit="frame",
        ...     time=np.array([0.0], dtype=np.float32),
        ...     frequency=np.array([0.0, 1000.0, 2000.0], dtype=np.float32),
        ...     values=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        ... )
        >>> store.add("acoustic.spectral.spectrum", spectrum)
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = SpectralSpreadExtractor().compute(feature_input, {})
        >>> result.unit
        'frame'
        """
        spectrum_output = feature_input.context["feature_store"].get(
            "acoustic.spectral.spectrum"
        )
        values = spectral_spread(
            spectrum_output.values,
            spectrum_output.frequency,
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(spectrum_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(SpectralSpreadExtractor)
