Pipeline
========

The VoxAtlas pipeline executes a *feature graph* for a single stream (for
example one conversation channel), turning configured extractors plus stream
inputs into a :class:`~voxatlas.pipeline.feature_store.FeatureStore` of typed
feature outputs.

Core responsibilities
---------------------

- validate requested features (and discover available extractors)
- resolve dependencies into an execution plan
- execute extractors in dependency order (optionally parallel within a layer)
- cache and reuse intermediate results (optional)
- collect outputs into a structured, queryable result store

Inputs and outputs
------------------

The pipeline is intentionally small and explicit:

- **Inputs**

  - ``audio``: an :class:`~voxatlas.audio.audio.Audio` instance (or ``None``)
  - ``units``: a :class:`~voxatlas.units.units.Units` hierarchy (or ``None``)
  - ``config``: a dictionary containing requested features and options

- **Output**

  - a :class:`~voxatlas.pipeline.feature_store.FeatureStore` mapping feature
    names to feature outputs (including computed dependencies)

Most extractors are written so they can access upstream results via
``feature_input.context["feature_store"]`` instead of recomputing them.

Configuration shape
-------------------

At minimum, a config selects features:

.. code-block:: yaml

   features:
     - acoustic.pitch.dummy

Optional sections control execution and per-feature parameters:

.. code-block:: yaml

   features:
     - acoustic.pitch.dummy

   feature_config:
     acoustic.pitch.dummy:
       example_param: 1

   pipeline:
     n_jobs: 1
     cache: false
     cache_dir: .voxatlas_cache

Notes:

- ``feature_config`` is merged with an extractor's ``default_config`` (when
  provided) by :func:`voxatlas.config.feature_config.resolve_feature_config`.
- ``pipeline.cache_dir`` is only used when ``pipeline.cache`` is enabled.

Execution model
---------------

When you call :meth:`voxatlas.pipeline.pipeline.VoxAtlasPipeline.run`, the
pipeline performs these steps:

1. **Discovery + validation:** imports feature modules and confirms every
   requested feature is registered.
2. **Dependency resolution:** walks each extractor's declared
   ``dependencies`` to build a dependency map and detect cycles.
3. **Planning:** groups features into dependency layers and materializes an
   :class:`~voxatlas.pipeline.execution_plan.ExecutionPlan`.
4. **Execution:** executes layers in order, inserting outputs into the shared
   :class:`~voxatlas.pipeline.feature_store.FeatureStore`.

Parallelism
-----------

Features in the same dependency layer can be executed independently. When
``pipeline.n_jobs > 1``, the pipeline uses process-based parallelism via
:func:`voxatlas.pipeline.executor.parallel_execute_layer`.

Practical implications:

- Parallel execution requires extractor inputs/outputs to be pickleable.
- Extractors should avoid relying on global mutable state (worker processes do
  not share memory).

Caching
-------

When ``pipeline.cache`` is enabled, the pipeline uses
:class:`~voxatlas.pipeline.cache.DiskCache` to load/store feature outputs on
disk as pickles.

- **Default location:** ``.voxatlas_cache`` (overridable via ``cache_dir``)
- **Cache key:** derived from the feature name, an audio hash (waveform bytes +
  sample rate), and a config hash (JSON with sorted keys)

Because the cache uses Python pickle, treat cache directories as trusted data
only.

Accessing results
-----------------

The return value of ``Pipeline(...).run()`` is a feature store:

.. code-block:: python

   from voxatlas.pipeline import Pipeline

   results = Pipeline(audio=audio, units=units, config=config).run()

   if results.exists("acoustic.pitch.dummy"):
       output = results.get("acoustic.pitch.dummy")
       print(output.feature, output.unit)

Key API entry points
--------------------

- :doc:`../api/generated/voxatlas.pipeline.pipeline`
- :doc:`../api/generated/voxatlas.pipeline.execution_plan`
- :doc:`../api/generated/voxatlas.pipeline.executor`
- :doc:`../api/generated/voxatlas.pipeline.feature_store`
- :doc:`../api/generated/voxatlas.pipeline.cache`
