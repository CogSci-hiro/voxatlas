from abc import ABC, abstractmethod

from .feature_input import FeatureInput


class BaseExtractor(ABC):
    """
    Abstract base class for public VoxAtlas feature extractors.

    Concrete extractors declare a feature name, the units they consume and
    produce, optional upstream dependencies, and a ``compute`` method. The
    pipeline uses this shared interface to execute every feature family through
    the registry.

    Notes
    -----
    Extractors should remain stateless. Dependency outputs should be read from
    ``feature_input.context["feature_store"]`` rather than stored on the
    extractor instance.

    Examples
    --------
    Usage example::

        class MyExtractor(BaseExtractor):
            name = "custom.example"
            input_units = "token"
            output_units = "token"
            dependencies = []
            default_config = {}

            def compute(self, feature_input, params):
                ...
    """

    name: str
    input_units: str | None = None
    output_units: str | None = None
    dependencies: list[str] = []
    default_config: dict = {}

    @abstractmethod
    def compute(self, feature_input: FeatureInput, params: dict):
        """
        Compute the extractor output for one stream.

        Parameters
        ----------
        feature_input : FeatureInput
            Prepared input bundle containing audio, unit tables, and pipeline
            context.
        params : dict
            Resolved configuration for the extractor.

        Returns
        -------
        object
            Structured VoxAtlas feature output.

        Notes
        -----
        Implementations should raise informative errors when required
        modalities or dependency outputs are unavailable.

        Examples
        --------
        Usage example::

            output = extractor.compute(feature_input, params)
            print(output)
        """
        pass
