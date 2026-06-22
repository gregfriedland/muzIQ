# ADR 0003: Boundary Event Decoder Training for Onset AP

## Status

Accepted. The higher-AP pursuit is still active; the current strict
validation onset AP target of greater than 0.90 has not been met.

## Date

2026-06-22

## Context

The source-tracking model needs reliable note boundary detection on the
metadata-held-out NSynth split. The primary phase 1 screen is validation onset
AP, or average precision: the area under the precision-recall curve when all
event-frame scores are ranked without choosing a fixed threshold. AP is the
right screening metric here because onset frames are rare and a single fixed
threshold can make the model look better or worse for calibration reasons.

The model already used a compact transformer encoder plus a causal boundary
decoder. The decoder is conditioned on event state, which is a small per-frame,
per-source history vector containing prior onset/offset indicators, an active
state, and time-since-event features. Training used scheduled sampling, which
means the model sees a mix of true event history and its own predicted event
history during training.

The critical diagnostic was teacher forcing. Teacher forcing means giving the
decoder the true event history instead of its generated history. With true
history, onset AP was near perfect; with inference-style generated history,
onset AP was much lower:

| Validation mode | Onset AP |
| --- | ---: |
| Generated history, teacher forcing 0.00 | 0.7149 |
| Mixed history, teacher forcing 0.25 | 0.8444 |
| Mixed history, teacher forcing 0.50 | 0.9161 |
| True history, teacher forcing 1.00 | 0.9988 |

This showed that raw capacity was not the main limit. The model could identify
onsets when event history was correct. The failure was exposure bias: the
history available at inference did not match the history quality seen by the
decoder during training.

## Decision

Keep the compact source-tracking transformer:

- model dimension 128;
- 2 encoder layers;
- 4 attention heads;
- 1 event decoder layer;
- 4 event decoder heads;
- stage 1 only for this AP screen: `single_note_all`;
- fixed validation seed set across epochs;
- validation teacher forcing rate 0.0;
- early stopping on validation `onset_average_precision`.

Add a first-pass boundary auxiliary loss. The trainer already performs two
passes when teacher forcing is below 1.0:

1. A first pass with no event-state conditioning predicts a sequence of onset
   and offset logits.
2. The trainer builds predicted event state from that first pass.
3. A second pass uses that predicted event state to make the final source-slot
   boundary predictions.

Before this ADR, the first pass was only used to build detached event state.
The training loss was applied to the second pass. The final design adds an
optional `first_pass_boundary_loss_weight` so the first pass is explicitly
trained as the history generator used at inference.

The auxiliary loss reuses boundary-only terms:

- onset binary loss;
- offset binary loss;
- hard boundary negative loss;
- onset pairwise ranking loss;
- onset softmax loss if enabled;
- onset sequence loss if enabled;
- onset sequence pairwise ranking loss if enabled;
- onset peak shape loss;
- onset event recall loss;
- onset false peak loss.

It intentionally excludes activity, family, and count losses. The first pass is
not a separate source-tracking head; it is only the event-history generator for
the second pass.

## Earlier Phase 1 Screen

The best run when this ADR was first written was:

`runs/boundary_phase1_ap_firstpass050_20260622`

Key configuration:

- `--stage-filter single_note_all`
- `--epochs-per-stage 100`
- `--validation-examples-per-epoch 2048`
- `--validation-interval-epochs 1`
- `--validation-teacher-forcing-rate 0.0`
- `--early-stopping-metric onset_average_precision`
- `--early-stopping-patience 10`
- `--early-stopping-min-delta 0.0005`
- `--first-pass-boundary-loss-weight 0.5`
- `--event-teacher-forcing-start 0.9`
- `--event-teacher-forcing-end 0.0`

Best validation result from that run:

| Metric | Value |
| --- | ---: |
| Epoch | 29 |
| Onset AP | 0.8179067373 |
| Onset best-threshold F1 | 0.7797902822 |
| Onset best threshold | 0.0071105957 |
| Onset best precision | 0.6683006287 |
| Onset best recall | 0.9359267950 |
| Onset sequence AP | 0.1732175052 |
| Offset AP | 0.7690952420 |

Best uploaded checkpoint:

`gs://rezo-flyte/scratch/serializable/muziq-nn/20260622/20260622/boundary-phase1-ap-firstpass050-notes200/checkpoints/checkpoint_000058_phase-epoch_done_stage-single_note_all_epoch-029_batch-000118.pt`

Local pod run directory:

`/workspace/muziq/runs/boundary_phase1_ap_firstpass050_20260622/train_runs_notes200/20260622-012206`

## Current Higher-AP Pursuit State

On 2026-06-22, the AP target was tightened to strict validation onset AP
greater than 0.90 on a notes48 phase 1 screen. The best warm-start checkpoint
for this screen was checkpoint 000099 from:

`runs/boundary_phase1_ap_mixed_warm_notes48_20260622`

Strict validation results for that checkpoint:

| Metric | Value |
| --- | ---: |
| Onset AP | 0.8259952664 |
| Slot-invariant onset AP | 0.8282662034 |
| Onset sequence AP | 0.2569503188 |
| Onset best-threshold F1 | 0.7822698355 |
| Onset best threshold | 0.0062866211 |

Teacher-forcing and event-state diagnostics showed a very high ceiling when
true event history is supplied, but a much lower score when the model must
generate its own history:

| Event-state path | Onset AP |
| --- | ---: |
| No event state | 0.7044475079 |
| Predicted event state | 0.8259952664 |
| True event state | 0.9999342561 |

Teacher-forcing sweep:

| Teacher forcing rate | Onset AP |
| --- | ---: |
| 0.00 | 0.8259952664 |
| 0.25 | 0.9092080593 |
| 0.50 | 0.9664344192 |
| 1.00 | 0.9999342561 |

This made the current bottleneck more specific than "boundary detection" in
general. The audio contains enough onset evidence, and the decoder can use
event history effectively when that history is correct. The remaining failure
is self-generated event history: inference-time history is not good enough to
keep the model above 0.90 AP.

Additional notes48 attempts did not clear the target:

| Run | Change | Best strict onset AP | Result |
| --- | --- | ---: | --- |
| `boundary_phase1_ap_state_noise_notes48_20260622` | Event-state noise and dropout | 0.8229572177 | Rejected |
| `boundary_phase1_ap_flux_notes48_20260622` | Add per-band positive flux input features | 0.8263381124 | Rejected; sequence AP improved but pointwise AP did not |
| `boundary_phase1_ap_firstpass_notes48_20260622` | Stronger first-pass and free-run sequence losses | 0.8268489242 | Rejected; no-state AP improved but predicted-state AP did not |
| `boundary_phase1_ap_softstate_notes48_20260622` | Soft event-state probabilities plus first-pass losses | 0.8161154985 | Rejected |

Final compare for the soft-state run:

| Event-state path | Onset AP |
| --- | ---: |
| No event state | 0.8155629635 |
| Predicted event state | 0.7977325320 |
| True event state | 0.9999929667 |

The next streamlined experiment is a phase 1-only scheduled-sampling curriculum
that anneals event-history teacher forcing from 1.0 to 0.0. This directly
targets exposure bias: the gap between high-AP true-history training behavior
and lower-AP generated-history inference behavior. If that still plateaus
below 0.90 AP, the next larger design change is to replace the externally
constructed event-state feedback with a learned recurrent or causal decoder
state that is trained end to end.

## Experiments Tried

| Change | Result | Decision |
| --- | ---: | --- |
| Baseline scheduled-sampling phase 1, `boundary_phase1_ap_tffast30_20260621` | Best onset AP 0.7427, sequence AP 0.0347 | Good baseline, below target |
| Wider model, `model_dim=192` | Best onset AP about 0.675 | Rejected; capacity increase hurt |
| Deeper event decoder, 2 layers | Best onset AP 0.7044, sequence AP 0.0690 | Rejected; depth hurt |
| Lower event-state thresholds, onset 0.25 and offset 0.03 | Best onset AP 0.7138 | Rejected |
| Lower onset threshold only, onset 0.25 and offset 0.5 | Best onset AP 0.7202, sequence AP 0.0933 | Rejected |
| Soft event-state probabilities | Best onset AP 0.7228, sequence AP 0.0693 | Rejected |
| Dense onset sequence binary loss, weight 0.1 or 0.25 | Best onset AP about 0.617 to 0.695 | Rejected; made final AP worse |
| Onset sequence pairwise ranking loss, weight 0.1 | Best onset AP 0.7110, sequence AP 0.0644 | Rejected |
| Onset sequence pairwise ranking loss, weight 0.2 | Best onset AP 0.6508, sequence AP 0.0146 | Rejected and stopped early |
| Activity-gated onset scoring, `onset * activity` | AP 0.5754 on best baseline checkpoint | Rejected; false positives were not mainly inactive-slot leakage |
| Activity-gated onset scoring, `onset * sqrt(activity)` | AP 0.6693 | Rejected |
| Activity-gated onset scoring, `min(onset, activity)` | AP 0.7213 | Rejected |
| Iterative inference refinement, extra decode passes | Pass 2 AP 0.7427; passes 3 to 5 fell to about 0.724 to 0.730 | Rejected; repeated self-conditioning did not repair history |
| First-pass boundary auxiliary loss, weight 0.5 | Best onset AP 0.8179 | Accepted |

Earlier focal-loss and hard-negative variants improved some batch-level behavior
but did not solve validation onset AP. In particular, hard-negative-heavy
experiments could crush onset recall. The accepted run keeps hard-negative loss
small and uses the first-pass auxiliary loss to train the generated history
path directly.

## Why Not Make the Model Larger?

The teacher-forcing sweep showed a high ceiling with the existing model. Wider
and deeper variants both performed worse. That made a larger architecture a
poor next lever. The useful change was not more capacity; it was aligning the
trained first-pass history generator with the inference-time path.

## Consequences

Future phase 1 AP screens should report:

- threshold-free onset AP;
- best-threshold onset F1, precision, recall, and threshold;
- onset sequence AP;
- offset AP;
- fixed-threshold F1 only as a diagnostic, not as the calibrated score.

When training with scheduled sampling, the first-pass history generator must be
supervised if its predictions are used to build event state. Otherwise the
second pass can learn to use good history while the first pass remains a weak
inference-time history source.

Do not spend more runs on simple width/depth scaling until a new diagnostic
shows capacity is again the bottleneck.

## Follow-Up

The earlier phase 1 screen improved validation AP, but the current greater than
0.90 strict validation AP target is still unmet. Do not proceed to phase 2 as a
production candidate until the phase 1 generated-history path clears the AP
target or a new diagnostic justifies changing the target.
