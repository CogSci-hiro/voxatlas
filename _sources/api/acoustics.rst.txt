Acoustics
=========

Acoustic feature extractors grouped by functional family.

Envelope
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~voxatlas.features.acoustic.envelope.hilbert.HilbertEnvelope
   ~voxatlas.features.acoustic.envelope.log_energy.LogEnergyEnvelope
   ~voxatlas.features.acoustic.envelope.oganian.OganianEnvelope
   ~voxatlas.features.acoustic.envelope.praat_intensity.PraatIntensityEnvelope
   ~voxatlas.features.acoustic.envelope.rms.RMSEnvelope
   ~voxatlas.features.acoustic.envelope.varnet.VarnetEnvelope

Envelope Derivative
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~voxatlas.features.acoustic.envelope.derivative.HilbertDerivative
   ~voxatlas.features.acoustic.envelope.derivative.LogEnergyDerivative
   ~voxatlas.features.acoustic.envelope.derivative.OganianDerivative
   ~voxatlas.features.acoustic.envelope.derivative.PraatIntensityDerivative
   ~voxatlas.features.acoustic.envelope.derivative.RmsDerivative
   ~voxatlas.features.acoustic.envelope.derivative.VarnetDerivative

Pitch
-----

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~voxatlas.features.acoustic.pitch.f0.F0Extractor
   ~voxatlas.features.acoustic.pitch.derivative.F0DerivativeExtractor
   ~voxatlas.features.acoustic.pitch.range.F0RangeExtractor
   ~voxatlas.features.acoustic.pitch.slope.F0SlopeExtractor
   ~voxatlas.features.acoustic.pitch.variability.F0VariabilityExtractor
   ~voxatlas.features.acoustic.pitch.contour_shape.F0ContourShapeExtractor

Spectral
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~voxatlas.features.acoustic.spectral.spectrum.SpectrumExtractor
   ~voxatlas.features.acoustic.spectral.centroid.SpectralCentroidExtractor
   ~voxatlas.features.acoustic.spectral.flatness.SpectralFlatnessExtractor
   ~voxatlas.features.acoustic.spectral.flux.SpectralFluxExtractor
   ~voxatlas.features.acoustic.spectral.rolloff.SpectralRolloffExtractor
   ~voxatlas.features.acoustic.spectral.slope.SpectralSlopeExtractor
   ~voxatlas.features.acoustic.spectral.spread.SpectralSpreadExtractor

Spectrogram
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~voxatlas.features.acoustic.spectrogram.mel.MelSpectrogramExtractor
   ~voxatlas.features.acoustic.spectrogram.stft.STFTSpectrogramExtractor

Voice Quality
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~voxatlas.features.acoustic.voice_quality.hnr.HNRExtractor
   ~voxatlas.features.acoustic.voice_quality.jitter.JitterExtractor
   ~voxatlas.features.acoustic.voice_quality.shimmer.ShimmerExtractor
