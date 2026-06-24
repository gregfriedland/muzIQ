#!/usr/bin/env bash
set -euo pipefail

cd /workspace/muziq

RUN_DIR="runs/boundary_sequence_phase12_clean_noisy_notes480_20260624"
DATA_ROOT="runs/nsynth_metadata_800_100_100_notes480_20260623"
STATUS="${RUN_DIR}/train_status.json"
RUN_ROOT="${RUN_DIR}/train_runs_notes480"
GCS_PREFIX="gs://rezo-flyte/scratch/serializable/muziq-nn/20260624/boundary-sequence-phase12-clean-noisy-notes480/checkpoints/"
WARM_START_URI="gs://rezo-flyte/scratch/serializable/muziq-nn/20260623/boundary-sequence-blockloss-mined-positive-notes48/checkpoints/20260623/20260623-041952/checkpoints/checkpoint_000009_phase-epoch_done_stage-single_note_all_epoch-003_batch-000235.pt"
WARM_START_CHECKPOINT="${RUN_DIR}/warm_start/checkpoint_000009_phase-epoch_done_stage-single_note_all_epoch-003_batch-000235.pt"
CENSORED_FAMILIES="vocal,flute,brass,string"
PHASE1_FAMILY_BOOSTS="guitar=2.0,flute=2.0,string=1.6,vocal=1.6,synth_lead=1.4"
PHASE2_FAMILY_BOOSTS="organ=2.2,reed=2.2,synth_lead=2.2,guitar=2.0,bass=1.8,flute=1.6,string=1.4,vocal=1.4"
HARD_CONTEXT_MIX="reattack=0.4,post_onset=0.3,pre_onset=0.15,ordinary=0.15"
HARD_POSITIVE_FAMILIES="organ,reed,synth_lead,guitar,bass"
HARD_NEGATIVE_FAMILIES="guitar,reed,organ,string,flute"
FULL_PHASE_JITTER_SAMPLES=79
FULL_FEATURE_NOISE_STD=0.01

mkdir -p "${RUN_DIR}/logs" "${RUN_ROOT}" "${RUN_DIR}/warm_start"

if pgrep -af "python -m muziq_nn.training.train .*${RUN_ROOT}" >/dev/null; then
  echo "Refusing to start duplicate phase 1/1a/2/2a training." >&2
  pgrep -af "python -m muziq_nn.training.train .*${RUN_ROOT}" >&2 || true
  exit 4
fi

if [[ ! -f "${DATA_ROOT}/manifests/nsynth/metadata_split_summary.json" ]]; then
  python -m muziq_nn.datasets.prepare \
    --data-root "${DATA_ROOT}" \
    --download-metadata \
    --build-nsynth-cache \
    --nsynth-metadata-instrument-split \
    --nsynth-train-instruments 800 \
    --nsynth-validation-instruments 100 \
    --nsynth-test-instruments 100 \
    --nsynth-notes-per-instrument 480 \
    --storage-budget-gb 80 \
    --progress-interval-s 30 \
    > "${RUN_DIR}/logs/prepare_notes480.log" 2>&1
fi

if [[ ! -s "${WARM_START_CHECKPOINT}" ]]; then
  python - <<PY
from pathlib import Path
from google.cloud import storage

uri = "${WARM_START_URI}"
target = Path("${WARM_START_CHECKPOINT}")
target.parent.mkdir(parents=True, exist_ok=True)
bucket_name, blob_name = uri[len("gs://"):].split("/", 1)
storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(target)
PY
fi

python - <<PY
import json
from datetime import UTC, datetime
from pathlib import Path

Path("${STATUS}").write_text(json.dumps({
    "status": "running",
    "kind": "boundary_sequence_phase12_clean_noisy_notes480",
    "started_at": datetime.now(UTC).isoformat(),
    "run_dir": "${RUN_DIR}",
    "run_root": "${RUN_ROOT}",
    "data_root": "${DATA_ROOT}",
    "warm_start_checkpoint": "${WARM_START_CHECKPOINT}",
    "warm_start_checkpoint_uri": "${WARM_START_URI}",
    "gcs_prefix": "${GCS_PREFIX}",
    "phases": [
        {"name": "phase1", "stage": "single_note_all", "phase_jitter_samples": 0, "feature_noise_std": 0.0},
        {"name": "phase1a", "stage": "single_note_all", "phase_jitter_samples": ${FULL_PHASE_JITTER_SAMPLES}, "feature_noise_std": ${FULL_FEATURE_NOISE_STD}},
        {"name": "phase2", "stage": "single_instrument_melody", "phase_jitter_samples": 0, "feature_noise_std": 0.0},
        {"name": "phase2a", "stage": "single_instrument_melody", "phase_jitter_samples": ${FULL_PHASE_JITTER_SAMPLES}, "feature_noise_std": ${FULL_FEATURE_NOISE_STD}}
    ],
    "early_stopping_metric": "onset_sequence_average_precision",
    "early_stopping_patience": 10
}, indent=2, sort_keys=True), encoding="utf-8")
PY

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --run-root "${RUN_ROOT}"
  --epochs-per-stage 60
  --batch-size 128
  --render-workers 48
  --device cuda
  --mixed-precision
  --gpu-frame-extraction
  --training-slice-frames 256
  --training-slice-peak-warmup-frames 512
  --input-novelty-features salience
  --no-resume-optimizer-state
  --checkpoint-upload-uri "${GCS_PREFIX}"
  --checkpoint-interval-batches 500
  --validation-examples-per-epoch 4096
  --validation-interval-epochs 1
  --validation-split validation
  --validation-teacher-forcing-rate 0.0
  --early-stopping-metric onset_sequence_average_precision
  --early-stopping-patience 10
  --early-stopping-min-delta 0.001
  --learning-rate-scale 0.04
  --learning-rate-decay cosine
  --learning-rate-min-fraction 0.1
  --learning-rate-decay-epochs 60
  --model-dim 256
  --model-heads 8
  --model-layers 4
  --event-decoder-layers 2
  --event-decoder-heads 8
  --primary-audio-only-event-decoder
  --event-teacher-forcing-start 1.0
  --event-teacher-forcing-end 0.0
  --onset-shoulder-radius-frames 5
  --offset-label-radius-frames 16
  --boundary-negative-radius-frames 24
  --positive-quality-boosts "fast_decay=2.0,bright=1.8,nonlinear_env=1.8,long_release=1.6,multiphonic=1.5"
  --positive-source-boosts "acoustic=1.0,electronic=1.0,synthetic=1.0"
  --positive-velocity-boosts "50=1.25,75=1.25,100=1.2,127=1.1"
  --activity-pos-weight 10
  --inactive-slot-weight 6
  --family-loss-weight 2.5
  --count-loss-weight 8
  --onset-loss-weight 0.25
  --offset-loss-weight 0.05
  --onset-pos-weight 5
  --offset-pos-weight 5
  --offset-focal-gamma 1.0
  --onset-pairwise-ranking-loss-weight 0.8
  --onset-pairwise-ranking-margin 0.5
  --onset-softmax-loss-weight 0.2
  --onset-sequence-loss-weight 0.75
  --onset-sequence-pairwise-ranking-loss-weight 0.5
  --onset-sequence-block-positive-loss-weight 1.0
  --onset-sequence-block-ranking-loss-weight 1.0
  --onset-nearby-pairwise-ranking-loss-weight 1.0
  --onset-peak-to-shoulder-ranking-loss-weight 0.0
  --onset-peak-loss-weight 0.0
  --onset-shoulder-loss-weight 0.10
  --onset-peak-radius-frames 5
  --onset-event-recall-loss-weight 1.0
)

PHASE1_ARGS=(
  --stage-filter single_note_all
  --onset-context-sample-prob 1.0
  --positive-family-boosts "${PHASE1_FAMILY_BOOSTS}"
  --hard-boundary-negative-loss-weight 0.25
  --hard-boundary-negative-fraction 0.02
  --onset-false-peak-loss-weight 1.0
  --onset-false-peak-fraction 0.02
)

PHASE2_ARGS=(
  --stage-filter single_instrument_melody
  --onset-context-sample-prob 0.5
  --censor-same-instrument-overlap-onset-families "${CENSORED_FAMILIES}"
  --hard-context-sample-prob 1.0
  --hard-context-mix "${HARD_CONTEXT_MIX}"
  --hard-context-positive-families "${HARD_POSITIVE_FAMILIES}"
  --hard-context-negative-families "${HARD_NEGATIVE_FAMILIES}"
  --hard-context-reattack-min-ms 400
  --hard-context-reattack-max-ms 900
  --hard-context-min-remaining-ms 1000
  --hard-context-post-onset-min-ms 25
  --hard-context-post-onset-max-ms 75
  --hard-context-pre-onset-min-ms 25
  --hard-context-pre-onset-max-ms 100
  --positive-family-boosts "${PHASE2_FAMILY_BOOSTS}"
  --hard-boundary-negative-loss-weight 0.35
  --hard-boundary-negative-fraction 0.05
  --onset-sequence-post-onset-ranking-loss-weight 1.0
  --onset-sequence-post-onset-min-frames 5
  --onset-sequence-post-onset-max-frames 15
  --onset-sequence-hard-context-ranking-loss-weight 1.0
  --onset-false-peak-loss-weight 1.5
  --onset-false-peak-fraction 0.05
  --phase1-onset-distillation-loss-weight 0.25
  --phase1-onset-sequence-distillation-loss-weight 1.0
)

run_phase() {
  local phase_name="$1"
  local warm_start="$2"
  local teacher_checkpoint="$3"
  local phase_jitter_samples="$4"
  local feature_noise_std="$5"
  shift 5
  local log="${RUN_DIR}/logs/${phase_name}_$(date -u +%Y%m%dT%H%M%SZ).log"

  python - <<PY
import json
from datetime import UTC, datetime
from pathlib import Path

status_path = Path("${STATUS}")
status = json.loads(status_path.read_text()) if status_path.exists() else {}
status.update({
    "status": "running",
    "current_phase": "${phase_name}",
    "current_phase_started_at": datetime.now(UTC).isoformat(),
    "current_log": "${log}",
    "current_warm_start_checkpoint": "${warm_start}",
    "current_phase1_onset_teacher_checkpoint": "${teacher_checkpoint}",
    "current_phase_jitter_samples": ${phase_jitter_samples},
    "current_feature_noise_std": ${feature_noise_std},
})
status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
PY

  local teacher_args=()
  if [[ -n "${teacher_checkpoint}" ]]; then
    teacher_args=(--phase1-onset-teacher-checkpoint "${teacher_checkpoint}")
  fi

  set +e
  python -m muziq_nn.training.train \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --warm-start-checkpoint "${warm_start}" \
    "${teacher_args[@]}" \
    --phase-jitter-samples "${phase_jitter_samples}" \
    --feature-noise-std "${feature_noise_std}" \
    > "${log}" 2>&1
  local code=$?
  set -e
  if [[ "${code}" -ne 0 ]]; then
    python - <<PY
import json
from datetime import UTC, datetime
from pathlib import Path

status_path = Path("${STATUS}")
status = json.loads(status_path.read_text()) if status_path.exists() else {}
status.update({
    "status": "failed",
    "failed_phase": "${phase_name}",
    "exit_code": ${code},
    "failed_at": datetime.now(UTC).isoformat(),
    "failed_log": "${log}",
})
status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
PY
    exit "${code}"
  fi
  latest_checkpoint="$(python - <<PY
from pathlib import Path

runs = sorted(path for path in Path("${RUN_ROOT}").iterdir() if path.is_dir())
print(runs[-1] / "checkpoint.pt")
PY
)"
  if [[ ! -s "${latest_checkpoint}" ]]; then
    echo "Missing checkpoint after ${phase_name}: ${latest_checkpoint}" >&2
    exit 5
  fi
  python - <<PY
import json
from datetime import UTC, datetime
from pathlib import Path

status_path = Path("${STATUS}")
status = json.loads(status_path.read_text()) if status_path.exists() else {}
completed = status.setdefault("completed_phases", [])
completed.append({
    "name": "${phase_name}",
    "log": "${log}",
    "checkpoint": "${latest_checkpoint}",
    "ended_at": datetime.now(UTC).isoformat(),
    "phase_jitter_samples": ${phase_jitter_samples},
    "feature_noise_std": ${feature_noise_std},
})
status.update({
    "status": "running",
    "last_completed_phase": "${phase_name}",
    "last_completed_checkpoint": "${latest_checkpoint}",
})
status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
PY
  printf '%s\n' "${latest_checkpoint}"
}

PHASE1_CHECKPOINT="$(run_phase phase1 "${WARM_START_CHECKPOINT}" "" 0 0.0 "${PHASE1_ARGS[@]}")"
PHASE1A_CHECKPOINT="$(run_phase phase1a "${PHASE1_CHECKPOINT}" "" "${FULL_PHASE_JITTER_SAMPLES}" "${FULL_FEATURE_NOISE_STD}" "${PHASE1_ARGS[@]}")"
PHASE2_CHECKPOINT="$(run_phase phase2 "${PHASE1A_CHECKPOINT}" "${PHASE1A_CHECKPOINT}" 0 0.0 "${PHASE2_ARGS[@]}")"
PHASE2A_CHECKPOINT="$(run_phase phase2a "${PHASE2_CHECKPOINT}" "${PHASE1A_CHECKPOINT}" "${FULL_PHASE_JITTER_SAMPLES}" "${FULL_FEATURE_NOISE_STD}" "${PHASE2_ARGS[@]}")"

python - <<PY
import json
from datetime import UTC, datetime
from pathlib import Path

status_path = Path("${STATUS}")
status = json.loads(status_path.read_text()) if status_path.exists() else {}
status.update({
    "status": "exited",
    "exit_code": 0,
    "ended_at": datetime.now(UTC).isoformat(),
    "final_checkpoint": "${PHASE2A_CHECKPOINT}",
    "phase1_checkpoint": "${PHASE1_CHECKPOINT}",
    "phase1a_checkpoint": "${PHASE1A_CHECKPOINT}",
    "phase2_checkpoint": "${PHASE2_CHECKPOINT}",
    "phase2a_checkpoint": "${PHASE2A_CHECKPOINT}",
})
status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
PY
