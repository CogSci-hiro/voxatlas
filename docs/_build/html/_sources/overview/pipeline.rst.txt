Pipeline
========

The VoxAtlas pipeline is responsible for turning configured extractors and
inputs into reproducible feature outputs.

Core responsibilities
---------------------

- build an execution plan
- run features in dependency order
- cache reusable intermediate results
- collect outputs into a structured result set

Key API entry points
--------------------

- :doc:`../api/generated/voxatlas.pipeline.pipeline`
- :doc:`../api/generated/voxatlas.pipeline.execution_plan`
- :doc:`../api/generated/voxatlas.pipeline.executor`
