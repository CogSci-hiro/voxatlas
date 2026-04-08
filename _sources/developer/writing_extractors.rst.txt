Writing Extractors
==================

Extractor implementations should stay focused on feature computation while
leaning on shared pipeline and registry infrastructure for orchestration.

General guidance
----------------

- inherit from the common extractor base when appropriate
- keep dependencies explicit
- return structured outputs
- document parameters, assumptions, and units clearly

Reference pages
---------------

- :doc:`../api/generated/voxatlas.features.base_extractor`
- :doc:`../api/generated/voxatlas.features.feature_input`
- :doc:`../api/generated/voxatlas.features.feature_output`
