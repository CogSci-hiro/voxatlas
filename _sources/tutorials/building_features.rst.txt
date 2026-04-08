Building Features
=================

New extractors should fit into the shared VoxAtlas feature contract so they
can be discovered and executed by the pipeline.

When adding a feature, define:

- the expected input unit and annotations
- the computation logic
- the output type and metadata
- any dependencies on upstream features

See also
--------

- :doc:`../developer/writing_extractors`
- :doc:`../developer/feature_registry`
