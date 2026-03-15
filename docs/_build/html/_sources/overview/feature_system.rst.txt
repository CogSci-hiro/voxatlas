Feature System
==============

Each VoxAtlas feature extractor follows a common interface so the registry and
pipeline can discover, validate, and execute it consistently.

The feature system centers on:

- extractor classes
- structured feature inputs
- typed feature outputs
- registry metadata and validation

Useful API pages
----------------

- :doc:`../api/generated/voxatlas.features.base_extractor`
- :doc:`../api/generated/voxatlas.features.feature_input`
- :doc:`../api/generated/voxatlas.features.feature_output`
- :doc:`../api/generated/voxatlas.registry.feature_registry`
