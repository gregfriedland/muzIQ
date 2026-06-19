# Boundary Timing Retrain Launch

## Active Pod

Use only east5 spot G4 pods with opaque UUID names. Do not use central1, and do not put
project names in pod names.

Current active training pod:

```bash
kubectl --context east5 get pod d66c3290-3242-49f2-ad7e-0c7536d5525b -n development
```

Before launching, verify inside the pod:

```bash
kubectl --context east5 -n development exec d66c3290-3242-49f2-ad7e-0c7536d5525b -- nvidia-smi
kubectl --context east5 -n development exec d66c3290-3242-49f2-ad7e-0c7536d5525b -- df -h /workspace
kubectl --context east5 -n development exec d66c3290-3242-49f2-ad7e-0c7536d5525b -- free -h
```

## Remote Repo Setup

Sync `/Users/gregfriedland/src/external/muziq` to `/workspace/muziq` on the pod.

The run needs these local artifacts:

- `runs/eval_notes200_compare_20260619/checkpoints/notes200_checkpoint_000401.pt`
- `runs/preview_phase2_20260619/audio/phase2_single_instrument_melody_test_60s.wav`

## Data Root

Preferred data root:

```bash
runs/nsynth_metadata_800_100_100_notes200_20260619
```

If missing, rebuild it before training:

```bash
scripts/prepare.sh \
  --data-root runs/nsynth_metadata_800_100_100_notes200_20260619 \
  --storage-budget-gb 32 \
  --download-metadata \
  --build-nsynth-cache \
  --nsynth-metadata-instrument-split \
  --nsynth-train-instruments 800 \
  --nsynth-validation-instruments 100 \
  --nsynth-test-instruments 100 \
  --nsynth-notes-per-instrument 200
```

## Training Command

```bash
scripts/train-boundary-timing.sh \
  --require-g4 \
  --data-root runs/nsynth_metadata_800_100_100_notes200_20260619 \
  --run-root runs/boundary_timing_retrain_20260619/train_runs_notes200 \
  --curriculum all \
  --stage-filter single_note_all,single_note_frames_cached,single_instrument_melody,single_instrument_melody_frames_cached \
  --epochs-per-stage 100 \
  --batch-size 256 \
  --device cuda \
  --render-workers 48 \
  --frame-cache-examples-per-stage 120000 \
  --frame-cache-build-batch-size 512 \
  --frame-cache-dtype float16 \
  --frame-cache-phase-jitter-samples 79 \
  --frame-phase-noise-std 0.01 \
  --gpu-frame-extraction \
  --pin-memory \
  --prefetch-batches 2 \
  --mixed-precision \
  --partial-warm-start \
  --warm-start-checkpoint runs/eval_notes200_compare_20260619/checkpoints/notes200_checkpoint_000401.pt \
  --checkpoint-upload-uri gs://rezo-flyte/scratch/serializable/muziq-nn/20260619/boundary-timing-notes200 \
  --checkpoint-upload-run-id boundary-timing-notes200 \
  --checkpoint-interval-batches 100 \
  --boundary-loss-weight 0.5 \
  --boundary-timing-loss-weight 0.25
```

The wrapper automatically runs `scripts/evaluate-boundary-timing-final.sh` after training to produce:

1. validation split threshold calibration;
2. calibrated test split metrics;
3. generated 60s sequence timing metrics;
4. a final summary JSON with artifact paths and key validation/test/60s metrics.

## Expected Final Artifacts

- Final checkpoint: `runs/boundary_timing_retrain_20260619/train_runs_notes200/<stamp>/checkpoint.pt`
- Calibration: `runs/boundary_timing_retrain_20260619/train_runs_notes200/<stamp>/calibration/onset_offset_thresholds.json`
- Test metrics: `runs/boundary_timing_retrain_20260619/train_runs_notes200/<stamp>/eval_test_calibrated/results/heldout_metrics.json`
- 60s metrics: `runs/boundary_timing_retrain_20260619/train_runs_notes200/<stamp>/eval_phase2_preview_60s/boundary_timing_metrics.json`
- Final summary: `runs/boundary_timing_retrain_20260619/train_runs_notes200/<stamp>/final_evaluation_summary.json`
- Uploaded checkpoints under: `gs://rezo-flyte/scratch/serializable/muziq-nn/20260619/boundary-timing-notes200/checkpoints/`
