import numpy as np

from voxatlas.acoustic.pitch_utils import compute_f0_derivative
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class F0DerivativeExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.pitch.f0.derivative`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.pitch.f0.derivative`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor computes a first-order temporal difference of the upstream contour while preserving the original frame alignment.
    
    1. Dependency retrieval
       The base frame-level contour is read from the feature store. This may be an :math:`f_0` contour or an amplitude-like envelope, depending on the feature family.
    
    2. Finite difference
       The code applies the backward difference
    
       .. math::
    
          d_t = x_t - x_{t-1},
    
       with missing values preserved when the upstream contour is undefined.
    
    3. Output alignment
       The derivative is emitted at the same frame times as the source contour so later extractors can compare slope-like dynamics and local changes without reindexing.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.pitch.f0'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.features.acoustic.pitch.derivative import F0DerivativeExtractor
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
    >>> out = F0DerivativeExtractor().compute(feature_input, {})
    >>> out.values.tolist()
    [nan, 10.0, nan]
    """
    name = "acoustic.pitch.f0.derivative"
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
        >>> from voxatlas.features.acoustic.pitch.derivative import F0DerivativeExtractor
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
        >>> result = F0DerivativeExtractor().compute(feature_input, {})
        >>> result.unit
        'frame'
        """
        f0_output = feature_input.context["feature_store"].get("acoustic.pitch.f0")
        values = compute_f0_derivative(f0_output.values)

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(f0_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(F0DerivativeExtractor)
