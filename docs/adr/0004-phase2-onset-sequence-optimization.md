# ADR 0004: Phase 2 Onset Sequence Optimization

## Status

Accepted as the current best phase 2 onset-sequence baseline. The run improved
over prior phase 2 attempts but did not meet the target onset sequence AP of
greater than 0.90.

## Date

2026-06-23

## Context

ADR 0003 established that phase 1 can learn reliable onset ranking on the
notes480 metadata-held-out NSynth split. The best phase 1 checkpoint reached
strict validation onset AP 0.9709844589 and onset sequence AP 0.9375238419 on
`single_note_all`.

Phase 2 changes the problem. It trains on `single_instrument_melody`, where the
same source can continue across many notes. The phase 1 checkpoint transfers
good single-note onset ranking, but phase 2 must adapt that ranking to
continuous re-articulation without turning every post-onset shoulder or timbral
change into a new onset.

The first phase 2 onset-sequence-AP run from the phase 1 checkpoint collapsed
relative to phase 1:

| Run | Change | Best onset sequence AP |
| --- | --- | ---: |
| `boundary_sequence_phase2_onset_ap_notes480_20260623` | Uncensored phase 2 | 0.462445 |
| `boundary_sequence_phase2_censored_onset_ap_notes480_20260623` | Censor same-instrument overlap onsets for vocal, flute, brass, string | 0.667826 |

The censored run showed that part of the problem was label ambiguity. Some
families, especially slow or continuous instruments, had same-instrument
re-articulations inside an already active note. Those labels asked the model to
detect a new onset even when the audio did not contain a clear separable attack.
Censoring those ambiguous onsets improved AP, but remaining errors were still
dominated by same-instrument re-articulations and post-onset false positives.

## Decision

Keep phase 2 as a phase-specific fine-tune from the phase 1 notes480 checkpoint,
but add losses that preserve the phase 1 onset ranking while adapting to melody
contexts.

The accepted phase 2 experiment is:

`runs/boundary_sequence_phase2_teacher_post_onset_notes480_20260623`

Key configuration:

- stage: `single_instrument_melody`;
- warm start:
  `runs/boundary_sequence_onset_context_notes480_20260623/train_runs_notes480/20260623-051427/checkpoint.pt`;
- frozen phase 1 onset teacher: same checkpoint;
- early stopping metric: `onset_sequence_average_precision`;
- early stopping patience: 10;
- validation teacher forcing rate: 0.0;
- `--onset-context-sample-prob 0.5`;
- censor same-instrument overlap onset families:
  `vocal,flute,brass,string`;
- phase 1 onset distillation loss weight: 0.25;
- phase 1 onset sequence distillation loss weight: 1.0;
- onset sequence post-onset ranking loss weight: 1.0;
- post-onset negative window: frames 5 through 15 after the onset;
- stronger false-peak and hard-boundary negative mining;
- positive family boosts for the dominant false-negative families:
  `organ`, `reed`, `synth_lead`, `guitar`, and `bass`.

Here, "preserve phase 1 onset ranking" means the phase 2 model is penalized
when its onset logits drift away from the phase 1 model's onset ordering on the
same audio frames. The goal is not to freeze phase 1 behavior. It is to let the
model adapt activity, family, and melody behavior while retaining the phase 1
ability to rank true onset blocks above nearby non-onset frames.

The post-onset ranking loss specifically targets a common phase 2 failure:
high-confidence detections in the decay or timbral shoulder after a real onset.
It ranks accepted onset-block max logits above non-onset frames 5 to 15 frames
after the onset.

## Results

Training exited cleanly with early stopping at epoch 50. The best validation
row was epoch 40.

Best checkpoint:

`gs://rezo-flyte/scratch/serializable/muziq-nn/20260623/boundary-sequence-phase2-teacher-post-onset-notes480/checkpoints/20260623/20260623-214752/checkpoints/checkpoint_000040_phase-epoch_done_stage-single_instrument_melody_epoch-040_batch-000469.pt`

Best validation metrics:

| Metric | Value |
| --- | ---: |
| Epoch | 40 |
| Onset sequence AP | 0.6941309571 |
| Onset sequence best-threshold F1 | 0.6777566671 |
| Onset sequence best threshold | 0.8710937500 |
| Onset sequence best precision | 0.7293371558 |
| Onset sequence best recall | 0.6329900622 |
| Onset AP | 0.3724868000 |
| Source count accuracy | 1.0000000000 |
| Mean predicted active count | 1.0000000000 |
| Family accuracy | 0.5520019531 |

This is a modest improvement over the prior censored phase 2 baseline:

| Run | Best onset sequence AP | Delta vs prior |
| --- | ---: | ---: |
| Uncensored phase 2 | 0.462445 | - |
| Censored phase 2 | 0.667826 | +0.205381 |
| Teacher/post-onset phase 2 | 0.694131 | +0.026305 |

The improvement is real but not sufficient. Phase 2 remains far below the
phase 1 onset sequence AP of 0.937524.

## Error Profile

The post-stop diagnostic used the best epoch 40 checkpoint, the validation
split, and threshold 0.87109375.

Diagnostic artifacts:

`runs/boundary_sequence_phase2_teacher_post_onset_notes480_20260623/analysis/fp_fn_after_early_stop/`

Threshold counts:

| Metric | Value |
| --- | ---: |
| Positive onset blocks | 5632 |
| True positive blocks | 3565 |
| False negative blocks | 2067 |
| False positive frames | 1323 |
| Precision | 0.7293371522 |
| Recall | 0.6329900568 |
| F1 | 0.6777566540 |

Dominant false negatives:

| Family | FN blocks |
| --- | ---: |
| organ | 442 |
| reed | 393 |
| synth_lead | 383 |
| guitar | 305 |
| bass | 258 |
| keyboard | 98 |
| mallet | 80 |

False negatives are still mostly same-instrument re-articulations:

| Situation | FN blocks |
| --- | ---: |
| Same-instrument re-articulation | 1956 |
| Not same-instrument re-articulation | 111 |

Dominant false positives:

| Family | FP frames |
| --- | ---: |
| guitar | 288 |
| reed | 211 |
| organ | 209 |
| string | 138 |
| flute | 112 |
| bass | 93 |
| vocal | 73 |

False positives are still mostly post-onset tail or shoulder detections:

| Timing bucket | FP frames |
| --- | ---: |
| 25-50ms after onset | 725 |
| More than 50ms after onset | 267 |
| More than 50ms before onset | 179 |
| 0-25ms after onset | 83 |
| 25-50ms before onset | 69 |

High-confidence false positives remain common:

| Score bucket | FP frames |
| --- | ---: |
| threshold-0.90 | 220 |
| 0.90-0.95 | 388 |
| 0.95-0.99 | 562 |
| greater than or equal to 0.99 | 153 |

Quality and velocity were unavailable in this diagnostic because
`SourceEventLabelV2` does not carry those metadata fields.

## Consequences

The current phase 2 bottleneck is not source count. Source count accuracy is
1.0 and the mean predicted active count is 1.0. The bottleneck is onset ranking
inside a single active source.

The teacher/post-onset experiment improved AP, so preserving phase 1 onset
ranking is useful. However, it did not change the dominant failure mode enough.
The model still struggles with two opposite cases in the same families:

1. Missing true same-instrument re-articulations for organ, reed, synth lead,
   guitar, and bass.
2. Firing on post-onset shoulders or decay frames for guitar, reed, organ,
   string, and flute.

Those errors imply that phase 2 needs a more direct contrast between true
re-attacks and post-onset shoulders. Generic hard-negative pressure and teacher
distillation are too indirect.

## Follow-Up

The next phase 2 experiment should focus on re-articulation contrast:

- sample paired examples from the same instrument where one frame is a true
  re-attack and another frame is a post-onset shoulder;
- rank true re-attacks above 25-75ms post-onset shoulder frames from the same
  family and source;
- add quality and velocity to `SourceEventLabelV2` or a parallel diagnostic
  metadata table so FP/FN reports can distinguish soft attacks, bright attacks,
  nonlinear envelopes, velocity, and source type;
- consider lowering broad shoulder supervision in phase 2 while keeping the
  phase 1 shoulder definition unchanged for phase 1 AP screening;
- keep early stopping on `onset_sequence_average_precision`, not fixed
  threshold F1.

Do not prioritize wider or deeper architecture for this specific issue. The
phase 1 model already reaches high onset sequence AP, and phase 2 source count
is solved. The missing leverage is discriminating true re-articulation from
post-onset timbral motion in continuous single-instrument melodies.
