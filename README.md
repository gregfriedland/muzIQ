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
scripts/train.sh \
  --curriculum all \
  --max-hours 8 \
  --generate-on-the-fly
```

The trainer calibrates throughput first, then trims later curriculum stages if
the full plan projects beyond the requested time budget.

For preemptible GPU pods, enable checkpoint upload so numbered model artifacts
are copied during training:

```bash
scripts/train.sh --require-g4 \
  --curriculum all \
  --max-hours 8 \
  --generate-on-the-fly \
  --device cuda \
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
checkpoint. The preview is intentionally limited to the Mac `BlackHole 2ch`
system-output capture path so the browser is only a display surface.

```bash
uv sync --extra dev --extra web
scripts/web-detached.sh
```

The script writes `runs/web/preview-8765.pid` and
`runs/web/preview-8765.log`. Pass a checkpoint path as the first argument, or
set `MUZIQ_NN_CHECKPOINT`; otherwise it uses the newest
`runs/k8s_downloads/*/checkpoint.pt`.

Open `http://127.0.0.1:8765` and click `Start BlackHole`. To analyze Mac
speaker output, install `BlackHole 2ch`, create a macOS Multi-Output Device
that includes both your speakers/headphones and `BlackHole 2ch`, then set that
Multi-Output Device as the system output.
The backend captures `BlackHole 2ch` through FFmpeg/AVFoundation, downmixes it
to the 16 kHz mono input expected by the model, and streams source activity,
family, onset/offset, confidence, input level, processing latency, and visual
position back to the grid.
