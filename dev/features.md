
# 1. Acoustic features


## Envelope family
- RMS energy
- log energy
- Praat intensity
- energy derivative
- energy peak rate
- energy variability
- Hilbert envelope
- Oganian envelope
- Varnet envelope

- transforms
	- derivative
	- onset (peak)

## Spectral envelope family

- spectral centroid
- spectral spread
- spectral slope
- spectral rolloff
- spectral flatness
- spectral flux

## Spectrogram family
- cochleogram
- spectrogram
- melspectrogram

## Voice quality family
- jitter
- shimmer
- harmonics-to-noise ratio (HNR)
- Pyannote embedding

## Pitch family
- F0
- F0 derivative
- F0 range
- F0 slope
- F0 contour shape
- pitch variability
- copasul contour

## Rhythm family
- syllable duration
- syllable rate
- pause rate
- nPVI (and variants)
- %V
- %C
- ΔV / ΔC
- VarcoV / VarcoC

# 2. Phonetics/phonology
## 2.1. Articulatory family
- vowel
- consonant
- nasal
- plosive
- fricative
- approximant
- voiced
- voiceless

## 2.2. Formants family
- vowel midpoint
- vowel trajectory
- full vowel average
- mean formant
- median formant
- variance
- onset–mid–offset values
- polynomial fit (F1/F2 trajectories)
- DCT coefficients
- slope of formant movement
- **VSA** (vowel space area)
- **tVSA** (triangle VSA)
- Vowel articulation index (VAI)
- Vowel centralization
## Prosodic phonology family
- stressed syllable
- unstressed syllable
- syllable position in word
- syllable position in IPU
## Phonotactics family
- consonant cluster size
- syllable structure (CV, CVC, CCV…)
## Phonological transition family
- phoneme surprisal (Markov/Hidden Markov)
- phoneme entropy (Markov/Hidden Markov)
  
## Reduction family
- schwa deletion
- vowel reduction

# 3. Morphology
## Inflection family
- tense
- aspect
- mood
- person
- number    
- gender
- case
- definiteness

## Verb morphology family
- finite vs non-finite
- participle
- infinitive

## Agreement family
- subject-verb agreement
- gender agreement

## Derivational morphology family
- prefix presence
- suffix presence
- morphological complexity
- number of morphemes

## Word formation family
- compound word indicator
- clitic presence
  
# 4.Lexical features
## Frequency family
- word frequency (Lexique)
- Zipf frequency
## Lexical richness family
- type/token ratio
- MTLD
- HD-D
## Word properties family
- word length
- syllable count
- phoneme count
## Semantics
- word embedding
- semantic surprisal
- semantic entropy
## Misc family
- function vs content
- animate vs inanimate
- concrete vs abstract

# 5. Syntax
## Local structure family
- POS tag
- dependency label
- head distance
- dependency distance
## Complexity family
- mean dependency length
- clause depth
- parse tree depth
- branching factor
## Clause structure family
- subordinate clause indicator
- relative clause indicator
- coordination indicator
## Phrase structure family
- noun phrase length
- verb phrase length
## Word order family
- subject position
- object position
## Incremental parsing metrics family
- syntactic surprisal
- entropy reduction



# 6. Conversational features
## Timing family
- turn duration
- IPU duration
- pause duration
- silence duration
- overlap duration
## Turn dynamics family
- turn position in conversation
- turn index
- speaker switch indicator
## Interactional metrics family
- response latency
- floor transfer offset
- interruption indicator
## Alignment / entrainment family
- speech rate convergence
- pitch convergence
- lexical alignment
- syntactic alignment
## Backchannel family
- backchannel detection
- filler detection
## Disfluency family
 - filled pauses
 - repairs
 - hesitation

