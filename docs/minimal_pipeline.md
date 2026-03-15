# Minimal Pipeline Guide

This guide shows the simplest current way to run VoxAtlas on real data with:

- an `input_root`
- an `output_root`
- a YAML config file

The example runner auto-discovers conversation ids from the dataset root and processes every conversation it finds.

## 1. Install the project

From the repository root:

```bash
pip install -e .
```

Install the audio backends used by the current loader if they are not already available:

```bash
pip install soundfile moviepy
```

## 2. Prepare the dataset root

The current dataset loader expects this structure:

```text
input_root/
├── audio/
│   conversation01.wav
│   conversation02.wav
└── alignment/
    ├── palign/
    │   ├── conversation01_ch1.TextGrid
    │   ├── conversation01_ch2.TextGrid
    │   ├── conversation02_ch1.TextGrid
    │   └── conversation02_ch2.TextGrid
    ├── syll/
    │   ├── conversation01_ch1.TextGrid
    │   ├── conversation01_ch2.TextGrid
    │   ├── conversation02_ch1.TextGrid
    │   └── conversation02_ch2.TextGrid
    └── ipu/
        ├── conversation01_ch1.TextGrid
        ├── conversation01_ch2.TextGrid
        ├── conversation02_ch1.TextGrid
        └── conversation02_ch2.TextGrid
```

Conversation ids are discovered from:

- `audio/*.wav`
- `alignment/palign/*_ch*.TextGrid`

For example:

- `audio/conversation01.wav`
- `alignment/palign/conversation01_ch1.TextGrid`

produce the conversation id `conversation01`.

## 3. Prepare a minimal config file

A ready-to-use example config is included at:

```text
examples/config.minimal.yaml
```

Its contents are:

```yaml
features:
  - acoustic.pitch.dummy

pipeline:
  n_jobs: 1
  cache: false
```

The built-in `acoustic.pitch.dummy` extractor is only for verifying the pipeline path end-to-end.

## 4. Run the example

From the repository root:

```bash
python examples/run_minimal.py /path/to/input_root /path/to/output_root /path/to/config.yaml
```

Example:

```bash
python examples/run_minimal.py ./dataset ./outputs ./examples/config.minimal.yaml
```

## 5. What the example does

For each discovered conversation, the script:

1. loads the dataset
2. pairs audio and alignment streams
3. runs the pipeline for each stream
4. forces `n_jobs=1`
5. forces `cache=false`
6. writes outputs to the output directory

The selected feature must be compatible with the available modality for each stream.
The built-in `acoustic.pitch.dummy` example feature now works when a stream has:

- audio only
- alignment only
- both audio and alignment

## 6. Output layout

The script writes one directory per conversation and per stream:

```text
output_root/
└── conversation01/
    ├── stream_0_A/
    │   ├── summary.json
    │   └── acoustic_pitch_dummy/
    │       ├── metadata.json
    │       └── values.csv
    └── stream_1_B/
        ├── summary.json
        └── acoustic_pitch_dummy/
            ├── metadata.json
            └── values.csv
```

Current output writing rules:

- `pandas.Series` values are written as `values.csv`
- `numpy.ndarray` values are written as `values.npy`
- basic metadata is written as `metadata.json`

## 7. Current limitations

- The example forces sequential execution with `n_jobs=1`.
- The example disables cache.
- The built-in feature is a dummy feature, not a production extractor.
- The current dataset loader expects the specific `audio/` plus `alignment/{palign,syll,ipu}/` structure above.
