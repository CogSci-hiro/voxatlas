Unit Hierarchy
==============

VoxAtlas models conversational analysis at multiple **units of observation**
(turns, words, phonemes, etc.). Many features are defined “per unit”: for
example, a per-word feature produces one value per row in the word table.

VoxAtlas represents these unit tables for a single stream (for example one
conversation channel) with :class:`~voxatlas.units.units.Units`.

Typical examples include:

- conversation-level measurements
- speaker or turn-level measurements
- utterance and sentence-level measurements
- token, word, syllable, or phoneme-level measurements

Extractors declare the unit level they consume and produce via
``BaseExtractor.input_units`` and ``BaseExtractor.output_units``. This metadata
is validated when extractors are registered and is surfaced in the CLI and API
docs. At runtime, extractors should still validate that required tables and
columns are present for the current dataset/stream.

Unit labels
-----------

Extractor unit labels must be one of the supported strings (or ``None``):

- ``conversation``
- ``turn``
- ``ipu``
- ``sentence``
- ``word``
- ``token``
- ``syllable``
- ``phoneme``
- ``frame``

``conversation`` is a *logical* level used for global or summary features; it
is not stored as a table on :class:`~voxatlas.units.units.Units`.

The ``Units`` container
-----------------------

:class:`~voxatlas.units.units.Units` is a lightweight wrapper around a set of
optional Pandas DataFrames:

- ``frames`` (frame-level time grid)
- ``tokens`` (token-level segmentation)
- ``phonemes``
- ``syllables``
- ``words``
- ``sentences``
- ``ipus`` (inter-pausal units)
- ``turns``

Tables are optional. Missing tables are represented as ``None`` and requesting
them via :meth:`~voxatlas.units.units.Units.table` raises ``ValueError``.

Table conventions
-----------------

``Units`` does not enforce a rigid schema, but most features assume a few
common conventions:

- ``id``: unique identifier for the unit row
- ``start`` / ``end``: segment boundaries on a shared timeline (commonly
  seconds)
- optional parent-child links using an ``<parent>_id`` column on the child
  table (for example, syllables can include ``word_id``)

The helper methods :meth:`~voxatlas.units.units.Units.parent` and
:meth:`~voxatlas.units.units.Units.children` implement this naming convention.

Illustrated hierarchy (typical)
-------------------------------

Many datasets provide a hierarchy like the following. The arrows are labeled
with the *child column* that links to the parent (for example, an ``ipu`` row
can carry a ``turn_id`` to identify its parent turn).

.. mermaid::

   graph TD
     turn((turns)) -->|turn_id| ipu([ipus])
     ipu -->|ipu_id| sentence[sentences]
     sentence -->|sentence_id| word[[words]]
     word -->|word_id| syllable([syllables])
     syllable -->|syllable_id| phoneme[/phonemes/]

     token[/tokens/]:::optional
     frames[("frames (time grid)")]:::time

     word -. optional .-> token
     frames -. time-aligned via start/end .-> word

     classDef top fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
     classDef mid fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;
     classDef low fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100;
     classDef time fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C;
     classDef optional fill:#F5F5F5,stroke:#616161,stroke-dasharray:4 3,color:#424242;

     class turn top;
     class ipu,sentence mid;
     class word,syllable,phoneme low;

The exact set of tables and links depends on the dataset. Features should be
written defensively (for example, fall back to time-alignment when explicit
links are unavailable, or raise a clear error when a required mapping is
missing).

Common access patterns
----------------------

.. code-block:: python

   # Get a table (singular/plural labels both work)
   words = units.table("word")

   # Durations from start/end
   word_durations = units.duration("word")

   # Parent ids (requires a <parent>_id column on the child table)
   word_ids_for_syllables = units.parent("syllable", "word")

   # Group children by parent id
   syllables_by_word = units.children("word", "syllable")

See also
--------

- :doc:`../api/generated/voxatlas.units`
- :doc:`../api/generated/voxatlas.units.units`
- :doc:`../api/generated/voxatlas.features.base_extractor`
