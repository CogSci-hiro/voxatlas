Minimal Example
===============

VoxAtlas is organized around modular feature extractors and a pipeline that
coordinates them.

This minimal example shows the intended workflow at a high level:

1. Load or prepare your conversational data.
2. Choose the feature extractors you want to run.
3. Execute the pipeline.
4. Inspect or export the resulting feature outputs.

Example outline
---------------

The snippet below is fully runnable and uses the built-in
``acoustic.pitch.dummy`` extractor to verify the pipeline path end-to-end.

.. code-block:: python

   import numpy as np

   from voxatlas.audio.audio import Audio
   from voxatlas.pipeline import Pipeline

   audio = Audio(waveform=np.zeros(16000, dtype=np.float32), sample_rate=16000)
   config = {
       "features": ["acoustic.pitch.dummy"],
       "pipeline": {"n_jobs": 1, "cache": False},
   }

   results = Pipeline(audio=audio, units=None, config=config).run()
   output = results.get("acoustic.pitch.dummy")

   print(output.feature, output.unit, float(output.values.iloc[0]))

To run it from a repository checkout without installing the package, execute
this from the repository root:

.. code-block:: bash

   PYTHONPATH=src python - <<'PY'
   import numpy as np

   from voxatlas.audio.audio import Audio
   from voxatlas.pipeline import Pipeline

   audio = Audio(waveform=np.zeros(16000, dtype=np.float32), sample_rate=16000)
   config = {"features": ["acoustic.pitch.dummy"], "pipeline": {"n_jobs": 1, "cache": False}}

   results = Pipeline(audio=audio, units=None, config=config).run()
   output = results.get("acoustic.pitch.dummy")
   print(output.feature, output.unit, float(output.values.iloc[0]))
   PY

Where to go next
----------------

- See :doc:`../overview/pipeline` for the pipeline structure.
- See :doc:`../overview/feature_system` for the extractor model.
- See :doc:`../api/index` for the generated API reference.
