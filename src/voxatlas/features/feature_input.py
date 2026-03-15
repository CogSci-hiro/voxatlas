from dataclasses import dataclass

from voxatlas.audio.audio import Audio
from voxatlas.units.units import Units


@dataclass
class FeatureInput:
    """
    Bundle the inputs passed to one extractor invocation.

    Parameters
    ----------
    audio : Audio | None
        Audio stream for the current conversation channel.
    units : Units | None
        Hierarchical unit container for the current stream.
    context : dict
        Shared execution context. The pipeline stores configuration and the
        feature store here.

    Returns
    -------
    FeatureInput
        Dataclass instance used by extractors.

    Notes
    -----
    The context dictionary is the standard place for cross-feature state such
    as dependency outputs and runtime configuration.

    Examples
    --------
    Usage example::

        feature_input = FeatureInput(audio=audio, units=units, context={"feature_store": store})
        print(feature_input.context.keys())
    """

    audio: Audio | None
    units: Units | None
    context: dict
