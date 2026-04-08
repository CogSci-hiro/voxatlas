Pipeline Internals
==================

This page describes how :class:`~voxatlas.pipeline.pipeline.VoxAtlasPipeline`
is implemented today: how it plans execution from declared dependencies, how it
executes extractors (including optional parallelism), and how intermediate and
final outputs move through the pipeline.

Main components
---------------

- dependency resolution + execution planning
- layer execution (sequential or process-parallel)
- intermediate storage via :class:`~voxatlas.pipeline.feature_store.FeatureStore`
- optional disk caching via :class:`~voxatlas.pipeline.cache.DiskCache`

High-level data flow
--------------------

At runtime, a pipeline instance is created for a single stream:

.. code-block:: python

   from voxatlas.pipeline import Pipeline
   results = Pipeline(audio=audio, units=units, config=config).run()

The pipeline builds a :class:`~voxatlas.features.feature_input.FeatureInput`
and executes extractors layer-by-layer, inserting each output into the shared
feature store. Downstream extractors can then reuse upstream outputs via
``feature_input.context["feature_store"]``.

.. mermaid::

   graph TD
     cfg["config (features + options)"] --> plan["ExecutionPlan (layers)"]
     reg["FeatureRegistry"] --> plan
     audio["Audio (optional)"] --> input["FeatureInput"]
     units["Units (optional)"] --> input
     store["FeatureStore"] --> input
     plan --> exec["Executor (per layer)"]
     input --> exec
     exec --> store
     cache["DiskCache (optional)"] <--> exec

Dependency resolution and planning
---------------------------------

The pipeline derives ordering from extractor metadata:

- Each extractor can declare a list of upstream dependencies via
  ``BaseExtractor.dependencies``.
- The pipeline validates that requested features (and every dependency) exist
  in the registry.
- A DFS walk produces a dependency map and detects cycles (raises
  ``ValueError``).

Planning produces an :class:`~voxatlas.pipeline.execution_plan.ExecutionPlan`
whose ``layers`` are computed by assigning each feature a level:

- level ``0``: no dependencies
- level ``n``: ``1 + max(level(dep) for dep in dependencies)``

All features at the same level are grouped into the same layer and are assumed
to have no remaining interdependencies.

Notes on ordering:

- The order of features *within* a layer is currently derived from the DFS
  insertion order (which depends on ``config["features"]`` and the declared
  dependency list ordering).
- Only dependency constraints are enforced; if two features have no dependency
  edge between them, their relative order is not semantically meaningful.

FeatureInput and shared state
-----------------------------

Before execution, the pipeline constructs a
:class:`~voxatlas.features.feature_input.FeatureInput` with:

- ``audio`` and ``units`` for the current stream
- ``context["config"]``: the full runtime configuration dict
- ``context["feature_store"]``: the shared feature store

This is the primary mechanism for passing cross-feature state. Extractors
should treat this input as read-only and only communicate results via returned
outputs (and, when necessary, by reading from the feature store).

Execution model (layer-by-layer)
--------------------------------

Execution proceeds in order over ``ExecutionPlan.layers``:

1. For each feature in the current layer, the pipeline checks if it already
   exists in the feature store (some features may appear multiple times via
   dependency traversal).
2. If disk caching is enabled, the pipeline checks for a cached pickle and
   loads it into the feature store when present.
3. Remaining features in the layer are executed via
   :func:`voxatlas.pipeline.executor.parallel_execute_layer`.
4. Each computed output is inserted into the feature store (and saved to cache
   if enabled).

Per-feature parameters are resolved immediately before execution using
:func:`voxatlas.config.feature_config.resolve_feature_config` (merging the
extractor's ``default_config`` with any overrides in
``config["feature_config"][<feature_name>]``).

Parallelism
-----------

Within a dependency layer, the pipeline can execute features in parallel using
process-based parallelism (``concurrent.futures.ProcessPoolExecutor``) when
``pipeline.n_jobs > 1``.

Practical constraints:

- Inputs and outputs must be pickleable to cross process boundaries.
- Worker processes do not share memory; avoid relying on global mutable state.
- Exceptions inside worker processes are re-raised when collecting futures,
  failing the pipeline run.

Caching (DiskCache)
-------------------

When ``pipeline.cache`` is enabled, the pipeline uses
:class:`~voxatlas.pipeline.cache.DiskCache` to store per-feature pickles under
``<cache_dir>/<feature_name>/<key>.pkl``.

The cache key is a SHA-256 hash of:

- the feature name
- an audio hash (waveform bytes + sample rate; or ``"no-audio"`` when audio is
  ``None``)
- a config hash (JSON dump with sorted keys)

Because caching uses Python pickle, cache directories must be treated as
trusted data.

FeatureStore semantics
----------------------

:class:`~voxatlas.pipeline.feature_store.FeatureStore` is a simple in-memory map
from feature name to output object:

- ``add(name, result)`` overwrites any existing value for the same name
- ``exists(name)`` checks presence
- ``get(name)`` raises ``KeyError`` if missing

The store contains both the requested features and any computed dependencies.

Reference pages
---------------

- :doc:`../api/generated/voxatlas.pipeline.pipeline.VoxAtlasPipeline`
- :doc:`../api/generated/voxatlas.pipeline.pipeline.VoxAtlasPipeline.run`
- :doc:`../api/generated/voxatlas.pipeline.execution_plan`
- :doc:`../api/generated/voxatlas.pipeline.executor`
- :doc:`../api/generated/voxatlas.pipeline.feature_store`
- :doc:`../api/generated/voxatlas.pipeline.cache`
