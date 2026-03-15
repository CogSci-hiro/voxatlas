import numpy as np

from voxatlas.acoustic.spectral_utils import spectral_centroid
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class SpectralCentroidExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectral.centroid`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectral.centroid`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor computes the first spectral moment of each frame.
    
    1. Dependency retrieval
       A non-negative magnitude spectrum :math:`S_{t,k}` and frequency axis :math:`f_k` are loaded from the upstream spectrum feature.
    
    2. Moment computation
       The centroid is
    
       .. math::
    
          C_t = \frac{\sum_k S_{t,k}f_k}{\sum_k S_{t,k}}.
    
    3. Packaging
       The frame-aligned centroid contour is returned for later aggregation or statistical summarization.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.spectral.spectrum'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.features.acoustic.spectral.centroid import SpectralCentroidExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import MatrixFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> store = FeatureStore()
    >>> spectrum = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    >>> freq = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    >>> base = MatrixFeatureOutput(
    ...     feature="acoustic.spectral.spectrum",
    ...     unit="frame",
    ...     time=np.array([0.0], dtype=np.float32),
    ...     frequency=freq,
    ...     values=spectrum,
    ... )
    >>> store.add("acoustic.spectral.spectrum", base)
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = SpectralCentroidExtractor().compute(feature_input, {})
    >>> out.values.tolist()
    [1.0]
    """
    name = "acoustic.spectral.centroid"
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
        >>> from voxatlas.features.acoustic.spectral.centroid import SpectralCentroidExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import MatrixFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> store = FeatureStore()
        >>> spectrum = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        >>> freq = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        >>> base = MatrixFeatureOutput(
        ...     feature="acoustic.spectral.spectrum",
        ...     unit="frame",
        ...     time=np.array([0.0], dtype=np.float32),
        ...     frequency=freq,
        ...     values=spectrum,
        ... )
        >>> store.add("acoustic.spectral.spectrum", base)
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = SpectralCentroidExtractor().compute(feature_input, {})
        >>> result.unit
        'frame'
        """
        spectrum_output = feature_input.context["feature_store"].get(
            "acoustic.spectral.spectrum"
        )
        values = spectral_centroid(
            spectrum_output.values,
            spectrum_output.frequency,
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(spectrum_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(SpectralCentroidExtractor)
