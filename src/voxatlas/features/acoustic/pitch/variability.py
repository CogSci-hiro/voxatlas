import numpy as np

from voxatlas.acoustic.pitch_utils import compute_f0_variability
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class F0VariabilityExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.pitch.f0.variability`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.pitch.f0.variability`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor converts the voiced :math:`f_0` contour into a global dispersion estimate and broadcasts that estimate to voiced frames.
    
    1. Voiced masking
       Only finite, positive :math:`f_0` samples are retained.
    
    2. Dispersion estimate
       The variability value is the population standard deviation
    
       .. math::
    
          \sigma_{f_0} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(f_i-\bar f)^2}.
    
    3. Alignment
       The same scalar dispersion is written back to all voiced frames so that later aggregation stages can preserve temporal alignment while still exposing a summary statistic.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.pitch.f0'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.features.acoustic.pitch.variability import F0VariabilityExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import VectorFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> store = FeatureStore()
    >>> base = VectorFeatureOutput(
    ...     feature="acoustic.pitch.f0",
    ...     unit="frame",
    ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
    ...     values=np.array([100.0, 110.0, np.nan], dtype=np.float32),
    ... )
    >>> store.add("acoustic.pitch.f0", base)
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = F0VariabilityExtractor().compute(feature_input, {})
    >>> float(np.nanmax(out.values))
    5.0
    """
    name = "acoustic.pitch.f0.variability"
    input_units = None
    output_units = "frame"
    dependencies = ["acoustic.pitch.f0"]
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
        >>> from voxatlas.features.acoustic.pitch.variability import F0VariabilityExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import VectorFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> store = FeatureStore()
        >>> base = VectorFeatureOutput(
        ...     feature="acoustic.pitch.f0",
        ...     unit="frame",
        ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
        ...     values=np.array([100.0, 110.0, np.nan], dtype=np.float32),
        ... )
        >>> store.add("acoustic.pitch.f0", base)
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = F0VariabilityExtractor().compute(feature_input, {})
        >>> result.unit
        'frame'
        """
        f0_output = feature_input.context["feature_store"].get("acoustic.pitch.f0")
        values = compute_f0_variability(f0_output.values)

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(f0_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(F0VariabilityExtractor)
