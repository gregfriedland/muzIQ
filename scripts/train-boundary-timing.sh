#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

data_root=""
run_root=""
preview_wav="runs/preview_phase2_20260619/audio/phase2_single_instrument_melody_test_60s.wav"
gcs_checkpoint_prefix=""
training_args=()

arg_value() {
  local key="$1"
  local idx
  for ((idx = 0; idx < ${#training_args[@]}; idx++)); do
    if [[ "${training_args[$idx]}" == "$key" && $((idx + 1)) -lt ${#training_args[@]} ]]; then
      printf '%s\n' "${training_args[$((idx + 1))]}"
      return 0
    fi
    if [[ "${training_args[$idx]}" == "$key="* ]]; then
      printf '%s\n' "${training_args[$idx]#*=}"
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preview-wav)
      preview_wav="$2"
      shift 2
      ;;
    --gcs-checkpoint-prefix)
      gcs_checkpoint_prefix="$2"
      shift 2
      ;;
    *)
      training_args+=("$1")
      shift
      ;;
  esac
done

data_root="$(arg_value --data-root || true)"
run_root="$(arg_value --run-root || true)"
if [[ -z "$data_root" ]]; then
  data_root="data"
fi
if [[ -z "$run_root" ]]; then
  run_root="runs"
fi

scripts/train.sh "${training_args[@]}"

run_dir="$(find "$run_root" -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 ls -td | head -1)"
final_eval_args=(
  --run-dir "$run_dir" \
  --data-root "$data_root" \
  --preview-wav "$preview_wav" \
  --single-notes-per-family 24 \
  --melodies-per-instrument 1 \
  --batch-size 1024 \
  --device "$(arg_value --device || printf 'auto')"
)
if [[ -n "$gcs_checkpoint_prefix" ]]; then
  final_eval_args+=(--gcs-checkpoint-prefix "$gcs_checkpoint_prefix")
fi
scripts/evaluate-boundary-timing-final.sh "${final_eval_args[@]}"
