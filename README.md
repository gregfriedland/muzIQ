# muzIQ

`muzIQ` is a standalone neural source-tracking repo for music visualization
analysis. It trains label-prediction models from bounded local caches of
instrument notes and MIDI schedules.

The repo is built for the 10 GB local-storage constraint:

- NSynth note WAVs are cached selectively.
- Lakh MIDI files are parsed into compact schedule records.
- Rendered audio and 40-band feature tensors are generated on the fly.
- Train/validation/test leakage is audited by NSynth instrument ID and MIDI hash.

## Setup

```bash
cd /Users/gregfriedland/src/external/muziq
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

For preemptible GPU pods, enable checkpoint upload so numbered model artifacts
are copied during training:

```bash
uv run python -m muziq_nn.training.train \
  --curriculum all \
  --max-hours 8 \
  --generate-on-the-fly \
  --checkpoint-upload-uri gs://rezo-flyte/scratch/serializable/muziq-nn \
  --checkpoint-upload-run-id "$(date -u +%Y%m%d)-example" \
  --checkpoint-interval-batches 100
```

Uploads are written under
`gs://rezo-flyte/scratch/serializable/muziq-nn/YYYYMMDD/RUN_ID/checkpoints/`.
Each checkpoint has a matching JSON metadata file with checkpoint number,
phase, stage, epoch, batch, loss, and timestamp, plus a `latest.json` pointer.

## Web Source Grid

The web app visualizes realtime source-slot predictions from a trained
checkpoint. It can embed a Spotify track for playback context, but inference
runs on uploaded audio files or browser-captured tab/system audio. The app does
not download or decode Spotify streams directly.

```bash
uv sync --extra dev --extra web
export MUZIQ_NN_CHECKPOINT=/Users/gregfriedland/src/external/muziq/runs/k8s_downloads/20260614-g4spot-889993c-50ep-oomfix-scratch2/checkpoint.pt
uv run muziq-web --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`, load a Spotify track URL if desired, then choose
either an audio file or browser audio capture. The backend expects 16 kHz
float32 mono chunks over `/ws/infer` and returns source activity, family,
onset/offset, confidence, and stable visual position for each source slot.
