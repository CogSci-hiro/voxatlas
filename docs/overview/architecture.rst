Architecture
============

VoxAtlas is structured as a modular toolkit rather than a single monolithic
feature extractor.

At a high level, the package is split into:

- feature implementations under ``voxatlas.features``
- pipeline orchestration under ``voxatlas.pipeline``
- discovery and validation logic under ``voxatlas.registry``
- domain-specific helpers such as ``voxatlas.audio`` and ``voxatlas.units``

This separation keeps feature code focused on computation while the pipeline
handles execution, caching, and organization of outputs.
