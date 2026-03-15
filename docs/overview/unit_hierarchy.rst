Unit Hierarchy
==============

VoxAtlas models conversational analysis at multiple units of observation.

Typical examples include:

- conversation-level measurements
- speaker or turn-level measurements
- utterance and sentence-level measurements
- token, word, syllable, or phoneme-level measurements

This hierarchy helps features declare what they consume and what they produce,
so pipeline validation can keep incompatible combinations from being run
together.

See also
--------

- :doc:`../api/generated/voxatlas.units`
- :doc:`../api/generated/voxatlas.units.units`
