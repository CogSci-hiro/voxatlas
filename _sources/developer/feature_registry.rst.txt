Feature Registry
================

The **feature registry** is VoxAtlas’ central lookup table for feature
extractors. It maps a fully-qualified feature name (for example
``"acoustic.pitch.f0"``) to:

- the extractor class that implements the feature (when available), and
- metadata needed by the CLI, discovery tooling, and the pipeline (units,
  dependencies, availability status).

The pipeline uses the registry to resolve extractors and walk declared
dependencies. The CLI uses it to list and inspect features without having to
manually import each feature module.

Responsibilities
----------------

- discover extractor implementations (import-time registration)
- store and expose feature metadata (:class:`~voxatlas.core.registry.FeatureRegistryEntry`)
- validate the extractor contract (name pattern, unit labels, dependencies)
- track *unavailable* features when optional dependencies are missing

Registration
------------

Most features register themselves at import time:

.. code-block:: python

   from voxatlas.core.registry import registry

   registry.register(MyExtractor)

Registration validates the extractor contract (see
:func:`voxatlas.core.registry.validate_extractor_contract`) and stores an entry
containing:

- ``name``
- ``dependencies``
- ``input_units`` / ``output_units``
- ``available`` (``True`` for normal registered extractors)

Discovery
---------

:func:`voxatlas.core.discovery.discover_features` walks the
``voxatlas.features`` package and imports modules to trigger registrations.

Discovery also supports *optional dependencies*:

- If importing a feature module fails because a third-party dependency is
  missing, VoxAtlas records an **unavailable** registry entry using
  :meth:`voxatlas.core.registry.FeatureRegistry.register_unavailable`.
- Unavailable entries keep the declared ``name``/``dependencies``/``units`` so
  the CLI can still report what exists and what is missing.
- When you later install the missing dependency (for example the ``syntax`` or
  ``acoustic`` optional dependency groups), discovery can register the real
  extractor class on the next run.

Validation rules (high level)
-----------------------------

The registry enforces a few consistency rules up front:

- **Feature names:** must be lowercase dot-separated segments (at least two
  segments).
- **Units:** must be one of the supported unit labels (or ``None``).
- **Dependencies:** must be a list of valid feature names.

These checks catch obvious configuration errors early and keep registry metadata
consistent for downstream tooling.

Inspecting features (CLI + Python)
----------------------------------

CLI:

.. code-block:: bash

   voxatlas features list
   voxatlas features info acoustic.pitch.dummy

Python:

.. code-block:: python

   from voxatlas.core.discovery import discover_features
   from voxatlas.core.registry import registry

   discover_features()
   entries = registry.list()
   pitch = registry.by_family("acoustic.pitch")
   grouped = registry.grouped()

Unavailable features
--------------------

When a feature is present in the codebase but cannot be imported, registry
resolution raises :class:`~voxatlas.core.registry.FeatureNotRegisteredError`
with additional context about the missing dependency.

This is also surfaced in the CLI as a status like ``missing:spacy``.

Reference pages
---------------

- :doc:`../api/generated/voxatlas.core.registry.FeatureRegistry`
- :doc:`../api/generated/voxatlas.core.registry.FeatureRegistry.list_features`
- :doc:`../api/generated/voxatlas.core.registry.register_feature`
