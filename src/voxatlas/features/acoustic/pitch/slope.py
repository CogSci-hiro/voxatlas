import numpy as np

from voxatlas.acoustic.pitch_utils import compute_f0_slope
from voxatlas.features.base_extractor import BaseExtractor
from voxatlas.features.feature_output import VectorFeatureOutput
from voxatlas.registry.feature_registry import registry


class F0SlopeExtractor(BaseExtractor):
    r"""
    Extract the ``acoustic.pitch.f0.slope`` feature within the VoxAtlas pipeline.
    
    This public extractor defines the reusable API for computing ``acoustic.pitch.f0.slope`` from VoxAtlas structured inputs. It consumes ``None`` units and produces values aligned to ``frame`` units, making the extractor a stable pipeline node that can be cited independently of the surrounding execution machinery.
    
    Algorithm
    ---------
    The extractor summarizes local pitch trajectory shape by fitting a straight line inside a sliding voiced window around each frame.
    
    1. Voiced-frame selection
       Only finite, positive :math:`f_0` values are retained. Frames with insufficient voiced support are left undefined.
    
    2. Local regression
       For each frame, the method estimates the least-squares slope
    
       .. math::
    
          \hat\beta = \frac{\sum_i (t_i-\bar t)(f_i-\bar f)}{\sum_i (t_i-\bar t)^2},
    
       over the voiced samples inside the configured neighborhood.
    
    3. Frame-level output
       The resulting slope is returned at frame resolution, making the feature suitable for intonational rise-fall analyses and downstream contour-shape abstractions.
    
    Notes
    -----
    This extractor declares the upstream dependencies ['acoustic.pitch.f0'] and is executed only after those features are available in the pipeline feature store.
    
    Examples
    --------
    >>> import numpy as np
    >>> from voxatlas.features.acoustic.pitch.slope import F0SlopeExtractor
    >>> from voxatlas.features.feature_input import FeatureInput
    >>> from voxatlas.features.feature_output import VectorFeatureOutput
    >>> from voxatlas.pipeline.feature_store import FeatureStore
    >>> store = FeatureStore()
    >>> base = VectorFeatureOutput(
    ...     feature="acoustic.pitch.f0",
    ...     unit="frame",
    ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
    ...     values=np.array([100.0, 110.0, 120.0], dtype=np.float32),
    ... )
    >>> store.add("acoustic.pitch.f0", base)
    >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
    >>> out = F0SlopeExtractor().compute(feature_input, {"window": 3})
    >>> out.values.shape == base.values.shape
    True
    """
    name = "acoustic.pitch.f0.slope"
    input_units = None
    output_units = "frame"
    dependencies = ["acoustic.pitch.f0"]
    default_config = {
        "window": 5,
    }

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
        >>> from voxatlas.features.acoustic.pitch.slope import F0SlopeExtractor
        >>> from voxatlas.features.feature_input import FeatureInput
        >>> from voxatlas.features.feature_output import VectorFeatureOutput
        >>> from voxatlas.pipeline.feature_store import FeatureStore
        >>> store = FeatureStore()
        >>> base = VectorFeatureOutput(
        ...     feature="acoustic.pitch.f0",
        ...     unit="frame",
        ...     time=np.array([0.0, 0.01, 0.02], dtype=np.float32),
        ...     values=np.array([100.0, 110.0, 120.0], dtype=np.float32),
        ... )
        >>> store.add("acoustic.pitch.f0", base)
        >>> feature_input = FeatureInput(audio=None, units=None, context={"feature_store": store})
        >>> result = F0SlopeExtractor().compute(feature_input, {"window": 3})
        >>> result.unit
        'frame'
        """
        f0_output = feature_input.context["feature_store"].get("acoustic.pitch.f0")
        values = compute_f0_slope(
            f0_output.values,
            window=params.get("window", 5),
        )

        return VectorFeatureOutput(
            feature=self.name,
            unit="frame",
            time=np.asarray(f0_output.time, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
        )


registry.register(F0SlopeExtractor)
