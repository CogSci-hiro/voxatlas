Feature System
==============

VoxAtlas features are implemented as small, composable **extractors** that
share a common contract. The registry and pipeline use this contract to
discover features, validate metadata (names/units/dependencies), and execute
extractors consistently.

The feature system centers on:

- **extractor classes** that implement one feature
- **structured feature inputs** (audio, units, shared context)
- **typed feature outputs** (scalar/vector/matrix/table/array containers)
- **registry metadata** used for discovery, validation, and dependency planning

Extractor contract (what you implement)
--------------------------------------

All extractors inherit from :class:`~voxatlas.features.base_extractor.BaseExtractor`
and typically define:

- ``name`` (required): fully-qualified feature name like ``"acoustic.pitch.f0"``
- ``input_units`` / ``output_units`` (optional): unit labels such as ``"token"``
  or ``"frame"`` (or ``None`` for audio/global features)
- ``dependencies`` (optional): upstream feature names that must run first
- ``default_config`` (optional): per-feature default parameters
- ``compute(feature_input, params)`` (required): returns a structured feature
  output

Extractors should be stateless. If you need upstream results, read them from
``feature_input.context["feature_store"]`` rather than storing them on the
extractor instance.

Feature inputs (what you receive)
---------------------------------

Each extractor invocation receives a
:class:`~voxatlas.features.feature_input.FeatureInput` bundle:

- ``feature_input.audio``: :class:`~voxatlas.audio.audio.Audio` or ``None``
- ``feature_input.units``: :class:`~voxatlas.units.units.Units` or ``None``
- ``feature_input.context``: shared runtime dictionary (pipeline config + feature store)

The pipeline stores the runtime config and the feature store in the context:

.. code-block:: python

   store = feature_input.context["feature_store"]
   upstream = store.get("syntax.dependencies")
   params = feature_input.context["config"]

Feature outputs (what you return)
---------------------------------

VoxAtlas standardizes common output shapes in
:mod:`voxatlas.features.feature_output`:

- :class:`~voxatlas.features.feature_output.ScalarFeatureOutput`: one scalar per unit
- :class:`~voxatlas.features.feature_output.VectorFeatureOutput`: time-aligned 1D sequence
- :class:`~voxatlas.features.feature_output.MatrixFeatureOutput`: time-frequency matrix
- :class:`~voxatlas.features.feature_output.TableFeatureOutput`: tabular output (DataFrame)
- :class:`~voxatlas.features.feature_output.ArrayFeatureOutput`: raw NumPy array

Most extractors should return one of these dataclasses so downstream consumers
and writers can handle outputs uniformly.

Registry + discovery (how features become runnable)
---------------------------------------------------

VoxAtlas uses a global :class:`~voxatlas.core.registry.FeatureRegistry` instance
to map feature names to extractor classes and metadata.

- **Registration:** feature modules typically call ``registry.register(MyExtractor)``
  at import time.
- **Discovery:** :func:`voxatlas.core.discovery.discover_features` walks the
  ``voxatlas.features`` package and imports modules to trigger registrations.
- **Optional dependencies:** if importing a feature module fails due to a missing
  third-party dependency, discovery records an *unavailable* registry entry (name,
  units, dependencies, missing dependency) so the CLI can still report it.

Configuration and parameters
----------------------------

The pipeline resolves per-feature parameters by merging:

1. an extractor's ``default_config`` (if provided), with
2. user overrides under ``config["feature_config"][<feature_name>]``

See :func:`voxatlas.config.feature_config.resolve_feature_config` for the exact
merge behavior.

Units and alignment
-------------------

Unit labels declared on extractors are validated when the extractor is
registered (for example ``"token"``, ``"word"``, ``"frame"``, or
``"conversation"``). For how unit tables are represented at runtime, see
:doc:`unit_hierarchy`.

At execution time, extractors should still check that required unit tables and
columns exist for the current dataset/stream and raise clear errors when they
do not.

Useful API pages
----------------

- :doc:`../api/generated/voxatlas.features.base_extractor`
- :doc:`../api/generated/voxatlas.features.feature_input`
- :doc:`../api/generated/voxatlas.features.feature_output`
- :doc:`../api/generated/voxatlas.core.registry.FeatureRegistry`
- :func:`voxatlas.core.discovery.discover_features`

Where to go next
----------------

- :doc:`../developer/writing_extractors` for a step-by-step extractor tutorial
- :doc:`pipeline` for how dependencies, parallelism, and caching are executed
