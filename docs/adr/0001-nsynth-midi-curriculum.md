# ADR 0001: Bounded NSynth + MIDI Source Tracking

**Status:** Proposed
**Date:** 2026-06-13

## Context

The complete NSynth corpus is about 73 GiB, Lakh MIDI is about 1.65 GiB
compressed, and rendered 30-second feature caches would require hundreds of
GiB. The target workstation budget for local artifacts is about 10 GiB.

## Decision

Train from a bounded cache:

- cache selected NSynth note WAVs and manifests;
- parse MIDI into compact schedule records;
- generate mixtures and 40-band frames in memory during training;
- never write rendered examples to disk;
- audit train/validation/test leakage by NSynth instrument ID and MIDI hash.

Use Python 3.12, `uv`, PyTorch, NumPy, SciPy, SoundFile, Mido, Pydantic, Pytest,
Google Cloud Storage, and Ruff.

GPU training on spot/preemptible pods must upload model checkpoints during the
run. The trainer writes numbered checkpoint artifacts and JSON metadata under a
date-versioned prefix:

`gs://rezo-flyte/scratch/serializable/muziq-nn/YYYYMMDD/RUN_ID/checkpoints/`

Each metadata file records checkpoint number, phase, stage, epoch, batch,
examples seen, loss, and creation time. A `latest.json` object is refreshed
after each upload so a revoked pod still leaves the latest reachable artifact.

## Curriculum

| Stage | Train examples / epoch | Epochs |
| --- | ---: | ---: |
| `single_note_all` | 30,000 | 3 |
| `single_instrument_melody` | 60,000 | 3 |
| `simple_duo_trio` | 80,000 | 3 |
| `midi_complex` | 100,000 | 3 |
| `hard_case_finetune` | 40,000 | 2 |

If a 10k-example throughput calibration projects beyond eight hours, trim
`midi_complex`, then `simple_duo_trio`, then `hard_case_finetune`.
