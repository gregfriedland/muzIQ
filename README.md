# muziq-nn

`muziq-nn` is a standalone neural source-tracking repo for music visualization
analysis. It trains label-prediction models from bounded local caches of
instrument notes and MIDI schedules.

The repo is built for the 10 GB local-storage constraint:

- NSynth note WAVs are cached selectively.
- Lakh MIDI files are parsed into compact schedule records.
- Rendered audio and 40-band feature tensors are generated on the fly.
- Train/validation/test leakage is audited by NSynth instrument ID and MIDI hash.

## Setup

```bash
cd /Users/gregfriedland/src/external/muziq-nn
uv venv --python 3.12
uv sync --extra dev
```

## Prepare Data

```bash
uv run python -m muziq_nn.datasets.prepare --storage-budget-gb 10 --download-metadata
uv run python -m muziq_nn.datasets.prepare --storage-budget-gb 10 --build-nsynth-cache
uv run python -m muziq_nn.datasets.prepare --storage-budget-gb 10 --build-midi-cache
uv run python -m muziq_nn.datasets.prepare --audit-leakage
```

The cache builders stream archives and retain only bounded local artifacts.
They do not store the full NSynth corpus or rendered training examples.

## Train

```bash
uv run python -m muziq_nn.training.train \
  --curriculum all \
  --max-hours 8 \
  --generate-on-the-fly
```

The trainer calibrates throughput first, then trims later curriculum stages if
the full plan projects beyond the requested time budget.
