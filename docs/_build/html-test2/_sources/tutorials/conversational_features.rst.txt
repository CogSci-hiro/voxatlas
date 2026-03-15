Conversational Features
=======================

VoxAtlas is designed for conversational speech workflows, where feature
extraction often spans multiple analysis levels and data sources.

Typical workflow
----------------

1. Load aligned or annotated conversation data.
2. Select extractors that match the available units and annotations.
3. Run the pipeline over the conversation set.
4. Aggregate outputs for downstream analysis.

Relevant APIs
-------------

- :doc:`../api/generated/voxatlas.units.alignment`
- :doc:`../api/generated/voxatlas.units.alignment_loader`
- :doc:`../api/generated/voxatlas.pipeline.pipeline`
