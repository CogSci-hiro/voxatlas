Architecture
============

VoxAtlas is structured as a modular toolkit rather than a single monolithic
feature extractor. The core idea is to keep **feature computation** small and
reusable, while centralizing **discovery, validation, dependency planning, and
execution** in shared infrastructure.

Design Goals
------------

- **Modularity:** each feature is an independent extractor with an explicit
  name, unit alignment, and dependency list.
- **Reproducibility:** the pipeline executes a dependency-sorted plan and can
  optionally cache intermediate results.
- **Extensibility:** adding a new feature should not require editing a central
  switch statement; it should register itself and become discoverable.
- **Graceful optional dependencies:** features that require extra libraries may
  be reported as *unavailable* without breaking discovery.

Package Layout (Conceptual Layers)
----------------------------------

At a high level, the project separates concerns into a few layers:

- ``voxatlas.features``: concrete extractor implementations (acoustic,
  phonology, syntax, morphology, lexical, ...).
- ``voxatlas.features.base_extractor`` / ``voxatlas.features.feature_input`` /
  ``voxatlas.features.feature_output``: the common feature contract and shared
  data containers.
- ``voxatlas.registry`` (backed by ``voxatlas.core``): discovery and registry
  metadata/validation for extractors.
- ``voxatlas.pipeline``: orchestration (dependency planning, parallel
  execution, caching, and result storage).
- Input model helpers:

  - ``voxatlas.audio``: audio container + loading helpers.
  - ``voxatlas.units``: hierarchical unit tables + TextGrid parsing/loading.
  - ``voxatlas.io``: dataset-level loading that pairs audio and alignments.

If you are looking for deeper details, see :doc:`pipeline`,
:doc:`feature_system`, and the developer notes in :doc:`../developer/pipeline_internals`.

End-to-End Data Flow
--------------------

The typical runtime flow is:

1. **Load inputs**

   - Acoustic workflows provide an :class:`~voxatlas.audio.audio.Audio` object.
   - Linguistic/alignment workflows provide a :class:`~voxatlas.units.units.Units`
     object (often sourced from TextGrid files).
   - Many workflows use both modalities.

2. **Load and validate configuration**

   Config is a small YAML or Python mapping that provides a ``features`` list
   and optional pipeline/feature parameters. The recommended entry point is
   :func:`voxatlas.config.load_and_prepare_config`.

3. **Build and run the pipeline**

   :class:`~voxatlas.pipeline.pipeline.VoxAtlasPipeline` validates the requested
   features, resolves dependencies through the registry, builds an
   :class:`~voxatlas.pipeline.execution_plan.ExecutionPlan`, and executes the
   graph layer-by-layer.

4. **Consume results**

   Outputs (requested features and computed dependencies) are stored in a
   :class:`~voxatlas.pipeline.feature_store.FeatureStore`.

The following sketch matches the internal responsibilities:

.. code-block:: text

   audio / units  +  config
          │
          ▼
     Pipeline.run()
       │  ├─ discover + validate features (registry)
       │  ├─ build dependency layers (ExecutionPlan)
       │  ├─ execute each layer (optionally parallel)
       │  ├─ cache (DiskCache)            ┐
       │  └─ store outputs (FeatureStore) ┘
       ▼
   FeatureStore (results)

Core Data Model
---------------

Feature execution uses a small set of shared containers:

- :class:`~voxatlas.features.feature_input.FeatureInput` bundles:

  - ``audio``: the current stream (optional)
  - ``units``: the hierarchical unit tables (optional)
  - ``context``: a shared dictionary that the pipeline uses for cross-feature
    state, including ``config`` and the current ``feature_store``

- Feature outputs are returned as typed dataclasses in
  :mod:`voxatlas.features.feature_output`:

  - :class:`~voxatlas.features.feature_output.ScalarFeatureOutput`
  - :class:`~voxatlas.features.feature_output.VectorFeatureOutput`
  - :class:`~voxatlas.features.feature_output.MatrixFeatureOutput`
  - :class:`~voxatlas.features.feature_output.TableFeatureOutput`
  - :class:`~voxatlas.features.feature_output.ArrayFeatureOutput`

Extractors should retrieve dependency outputs via
``feature_input.context["feature_store"].get("<dependency.name>")`` rather than
recomputing upstream features.

Feature Discovery and the Registry
----------------------------------

VoxAtlas uses a global :class:`~voxatlas.core.registry.FeatureRegistry` instance
to map feature names (for example ``"acoustic.pitch.f0"``) to extractor classes.

- **Registration:** extractors register themselves (typically at import time)
  via ``registry.register(ExtractorClass)``.
- **Discovery:** :func:`voxatlas.core.discovery.discover_features` walks the
  ``voxatlas.features`` package and imports every feature module to trigger
  registrations.
- **Optional dependencies:** if importing a feature module fails due to a
  missing third-party dependency, VoxAtlas records the feature as
  *unavailable* (including its declared name/dependencies/units) so the CLI and
  developer tooling can still report it.

Pipeline Execution, Parallelism, and Caching
--------------------------------------------

Pipeline execution is intentionally simple:

- **Planning:** dependency layers are computed from the declared
  ``BaseExtractor.dependencies`` lists and stored in an
  :class:`~voxatlas.pipeline.execution_plan.ExecutionPlan`. Features in the same
  layer are assumed to have no remaining interdependencies.
- **Execution:** layers are executed in order. Within one layer, the pipeline
  can use process-based parallelism (``n_jobs``) via
  :func:`voxatlas.pipeline.executor.parallel_execute_layer`.
- **Storage:** computed outputs are inserted into a
  :class:`~voxatlas.pipeline.feature_store.FeatureStore` for downstream lookup.
- **Disk cache (optional):** when enabled, :class:`~voxatlas.pipeline.cache.DiskCache`
  stores pickled feature outputs under ``cache_dir/<feature>/<key>.pkl``, where
  the key is derived from the feature name plus hashes of the audio payload and
  the pipeline configuration.

Extension Points
----------------

The primary extension mechanism is writing a new extractor:

1. Implement a class that inherits from
   :class:`~voxatlas.features.base_extractor.BaseExtractor`.
2. Define the class attributes:

   - ``name`` (required): the fully-qualified feature name
   - ``input_units`` / ``output_units`` (optional): declared unit alignment
   - ``dependencies`` (optional): upstream feature names
   - ``default_config`` (optional): per-feature default parameters

3. Implement ``compute(feature_input, params)``.
4. Register the extractor in the global registry.
5. Place it under ``voxatlas.features`` so discovery can import it.

For practical guidance, see :doc:`../developer/writing_extractors`.
