import numpy as np

from voxatlas.acoustic.spectral_utils import spectral_slope
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class SpectralSlopeExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.spectral.slope`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.spectral.slope`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor follows the standard VoxAtlas feature-computation pattern.
    
    1. Input preparation
       Structured audio, unit tables, and dependency outputs are gathered from ``feature_input``.
    
    2. Feature-specific computation
       The implementation applies the domain-specific transformation required by this extractor.
    
    3. Packaging
       Results are aligned to ``frame`` units and returned as a ``FeatureOutput`` object.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.spectral.spectrum'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.features.acoustic.spectral.slope import SpectralSlopeExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import MatrixFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> store = FeatureStore()
    >>> spectrum = MatrixFeatureOutput(
    ...     feature="acoustic.spectral.spectrum",
    ...     unit="frame",
    ...     time=np.array([0.0, 0.01], dtype=np.float32),
    ...     frequency=np.array([0.0, 1000.0, 2000.0], dtype=np.float32),
    ...     values=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32),
    ... )
    >>> store.add("acoustic.spectral.spectrum", spectrum)
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = SpectralSlopeExtractor().compute(feature_input, {})
    >>> out.values.shape
    (2,)
    >>> float(np.sign(out.values[0]))
    1.0
    >>> float(np.sign(out.values[1]))
    -1.0
    """
    name = "acoustic.spectral.slope"
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
        >>> from voxatlas.features.acoustic.spectral.slope import SpectralSlopeExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import MatrixFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> store = FeatureStore()
        >>> spectrum = MatrixFeatureOutput(
        ...     feature="acoustic.spectral.spectrum",
        ...     unit="frame",
        ...     time=np.array([0.0], dtype=np.float32),
        ...     frequency=np.array([0.0, 1000.0, 2000.0], dtype=np.float32),
        ...     values=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        ... )
        >>> store.add("acoustic.spectral.spectrum", spectrum)
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = SpectralSlopeExtractor().compute(feature_input, {})
        >>> result.unit
        'frame'
        """
        spectrum_output = feature_input.context["feature_store"].get(
            "acoustic.spectral.spectrum"
        )
        values = spectral_slope(
            spectrum_output.values,
            spectrum_output.frequency,
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(spectrum_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(SpectralSlopeExtractor)
