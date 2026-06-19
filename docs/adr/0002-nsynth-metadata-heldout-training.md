# ADR 0002: NSynth Metadata-Based Held-Out Instrument Training

## Status

Accepted.

## Context

The first stage 1/2 training experiments used note-level separation but did not
guarantee held-out instrument identities. A later audit showed that exact note
names were disjoint, but `instrument_str` values overlapped across train and
test. Those metrics were therefore optimistic for the real target: detecting
and tracking instruments whose timbre was not seen during training.

NSynth provides official metadata in each `examples.json` archive entry. That
metadata includes the canonical global `instrument` identity, `instrument_str`,
instrument family, source, pitch, velocity, and qualities. The filename suffix
alone is not a safe global identity because it is local to a family/source name.

## Decision

Training data preparation must use official NSynth metadata as the identity
source. The target split is:

- 800 train instruments;
- 100 validation instruments;
- 100 test instruments;
- 0 overlap by official `instrument`;
- 0 overlap by `instrument_str`;
- 0 overlap by `note_str`.

The cache remains bounded by selecting a pitch/velocity-diverse subset of notes
per instrument. The first full protocol uses 24 notes per selected instrument,
which gives approximately:

- 19,200 train notes;
- 2,400 validation notes;
- 2,400 test notes.

Rendered 30-second mixtures are still generated on the fly. Rendered audio and
40-band tensors are not materialized.

## Training Changes

Stage 1 and stage 2 are trained only:

1. `single_note_all`
2. `single_instrument_melody`

Each stage is trained for 100 epochs.

The renderer now applies train-time augmentation to improve held-out timbre
generalization:

- small spectral tilt;
- short delay/reverb tail;
- low-level additive noise;
- varied gain through final peak normalization;
- pre-onset and post-offset hard-negative frame sampling.

Boundary labels are no longer exact single-frame labels:

- onset labels use a small window;
- offset labels use a wider window to be lenient for slow-decay instruments;
- evaluation offset recall uses the best offset score inside the tolerance
  window.

The loss keeps the explicit source-count head and cross-entropy count loss,
with strong inactive-slot penalty so extra slots do not fire for one source.

## Commands

Prepare the full metadata split:

```bash
uv run python -m muziq_nn.datasets.prepare \
  --data-root runs/nsynth_metadata_800_100_100_20260618 \
  --download-metadata \
  --build-nsynth-cache \
  --nsynth-metadata-instrument-split \
  --nsynth-train-instruments 800 \
  --nsynth-validation-instruments 100 \
  --nsynth-test-instruments 100 \
  --nsynth-notes-per-instrument 24 \
  --storage-budget-gb 10
```

Train stage 1/2:

```bash
scripts/train.sh --require-g4 \
  --data-root runs/nsynth_metadata_800_100_100_20260618 \
  --run-root runs/stage12_20260618/train_runs_metadata_800_100_100 \
  --stage-filter single_note_all,single_instrument_melody \
  --curriculum-scale 0.04 \
  --epochs-per-stage 100 \
  --batch-size 64 \
  --device cuda \
  --calibration-examples 64 \
  --progress-interval-s 15 \
  --render-workers 1 \
  --model-dim 128 \
  --model-layers 2 \
  --model-heads 4 \
  --identity-dim 16 \
  --activity-pos-weight 10 \
  --inactive-slot-weight 6 \
  --count-loss-weight 8 \
  --family-loss-weight 2.5 \
  --onset-pos-weight 10 \
  --offset-pos-weight 10 \
  --boundary-loss-weight 0.5
```

Evaluate the held-out test instruments:

```bash
uv run python runs/eval_family_20260618/scripts/evaluate_nsynth_family_samples.py \
  --checkpoint <checkpoint.pt> \
  --output-dir runs/eval_family_20260618/metadata_800_100_100_test_eval \
  --samples-per-family 12 \
  --melodies-per-family 4 \
  --activity-threshold 0.35 \
  --offset-tolerance-frames 16 \
  --batch-size 512 \
  --device mps
```

## Results

Pending completion of the full cache build, stage 1/2 training, and held-out
test evaluation.

The strict pre-full-cache baseline, trained from scratch on only 32 train
instrument IDs and tested on 9 held-out IDs, achieved:

- stage 1 source-count accuracy: 1.000;
- stage 1 detection recall: 0.500;
- stage 1 family accuracy on active frames: 0.175;
- stage 1 onset recall: 0.958;
- stage 1 offset recall: 0.217;
- stage 2 source-count accuracy: 1.000;
- stage 2 detection recall: 0.694;
- stage 2 family accuracy on active frames: 0.273;
- stage 2 onset recall: 0.564;
- stage 2 offset recall: 0.564;
- stage 2 ID switches/minute: 0.000.

## Consequences

This protocol makes the test set much harder and more representative of unseen
instruments. Family classification and detection may initially fall compared
with note-held-out metrics, but the result is honest. Future improvements should
be judged against the metadata-held-out validation and test splits, not against
note-held-out examples.

If local disk or time becomes a bottleneck, keep the 800/100/100 instrument
split and reduce only notes per instrument or generated examples per epoch.
