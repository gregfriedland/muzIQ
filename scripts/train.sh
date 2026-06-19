#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

require_g4=0
training_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --require-g4)
      require_g4=1
      shift
      ;;
    *)
      training_args+=("$1")
      shift
      ;;
  esac
done

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

device="$(arg_value --device || true)"
epochs_per_stage="$(arg_value --epochs-per-stage || true)"
data_root="$(arg_value --data-root || true)"
stage_filter="$(arg_value --stage-filter || true)"

long_run_reasons=()
if [[ -n "$epochs_per_stage" && "$epochs_per_stage" =~ ^[0-9]+$ ]]; then
  if ((epochs_per_stage >= 50)); then
    long_run_reasons+=("${epochs_per_stage}-epoch training")
  fi
fi
if [[ "$data_root" == *"nsynth_metadata_800_100_100"* || "$data_root" == *"metadata_800_100_100"* ]]; then
  long_run_reasons+=("800/100/100 NSynth metadata split")
fi
if [[ "$stage_filter" == *"single_note_all"* && "$stage_filter" == *"single_instrument_melody"* ]]; then
  if [[ -z "$epochs_per_stage" || ! "$epochs_per_stage" =~ ^[0-9]+$ || "$epochs_per_stage" -ge 20 ]]; then
    long_run_reasons+=("full stage 1/2 training")
  fi
fi

if (( ${#long_run_reasons[@]} > 0 )) && [[ "$require_g4" != "1" ]]; then
  if [[ "${MUZIQ_ALLOW_LOCAL_LONG_TRAIN:-0}" != "1" ]]; then
    echo "scripts/train.sh: refusing long training without --require-g4." >&2
    printf 'Reason: %s\n' "${long_run_reasons[@]}" >&2
    echo "Use a G4 pod with: scripts/train.sh --require-g4 ... --device cuda" >&2
    echo "For an intentional local fallback, set MUZIQ_ALLOW_LOCAL_LONG_TRAIN=1." >&2
    exit 2
  fi
fi

if [[ "$require_g4" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "scripts/train.sh: --require-g4 set, but nvidia-smi is not available." >&2
    echo "Run this inside a Running east5 g4-standard-48 RTX Pro 6000 pod." >&2
    exit 2
  fi
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  if [[ ! "$gpu_name" =~ RTX[[:space:]]+(Pro[[:space:]]+)?6000|RTX[[:space:]]+PRO[[:space:]]+6000 ]]; then
    echo "scripts/train.sh: --require-g4 set, but GPU is '$gpu_name'." >&2
    echo "Expected an RTX Pro 6000 G4 pod." >&2
    exit 2
  fi
  if [[ "$device" != "cuda" ]]; then
    echo "scripts/train.sh: --require-g4 requires '--device cuda'." >&2
    exit 2
  fi
else
  if (( ${#long_run_reasons[@]} > 0 )) && [[ "$device" != "cuda" ]]; then
    if [[ "${MUZIQ_ALLOW_LOCAL_LONG_TRAIN:-0}" != "1" ]]; then
      echo "scripts/train.sh: refusing long training on device '${device:-default}'." >&2
      printf 'Reason: %s\n' "${long_run_reasons[@]}" >&2
      echo "Use a G4 pod with: scripts/train.sh --require-g4 ... --device cuda" >&2
      echo "For an intentional local fallback, set MUZIQ_ALLOW_LOCAL_LONG_TRAIN=1." >&2
      exit 2
    fi
  fi
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m muziq_nn.training.train "${training_args[@]}"
fi

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m muziq_nn.training.train "${training_args[@]}"
