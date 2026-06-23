#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_dir=""
checkpoint=""
data_root="data"
preview_wav="runs/preview_phase2_20260619/audio/phase2_single_instrument_melody_test_60s.wav"
device="auto"
batch_size="1024"
single_notes_per_family="24"
melodies_per_instrument="1"
max_melody_instruments="0"
sequence_calibration_max_instruments="8"
sequence_calibration_melodies_per_instrument="1"
activity_threshold="0.35"
onset_threshold="0.35"
offset_threshold="0.35"
onset_tolerance_frames=""
offset_tolerance_frames=""
gcs_checkpoint_prefix="gs://rezo-flyte/scratch/serializable/muziq-nn/20260619/boundary-timing-notes200/checkpoints/"
force=0
skip_preview_if_missing=0

usage() {
  cat <<'EOF'
Usage: scripts/evaluate-boundary-timing-final.sh --run-dir DIR [options]

Runs the full post-training boundary-timing evaluation:
  1. calibrate onset/offset thresholds on the validation split;
  2. evaluate the calibrated model on the test split;
  3. evaluate the generated 60s preview WAV;
  4. write final_evaluation_summary.json with artifact paths and key metrics.

Options:
  --run-dir DIR                     Training run directory. Required.
  --checkpoint PATH                 Model checkpoint. Default: RUN_DIR/checkpoint.pt.
  --data-root DIR                   Dataset root. Default: data.
  --preview-wav PATH                60s preview WAV path.
  --device DEVICE                   auto, cpu, cuda, etc. Default: auto.
  --batch-size N                    Heldout eval batch size. Default: 1024.
  --single-notes-per-family N       Validation/test single-note sample count. Default: 24.
  --melodies-per-instrument N       Validation/test melody sample count. Default: 1.
  --max-melody-instruments N        Cap melody instruments, 0 means all. Default: 0.
  --sequence-calibration-max-instruments N
                                   Cap validation instruments for continuous sequence
                                   threshold calibration. Default: 8.
  --sequence-calibration-melodies-per-instrument N
                                   Validation melodies per instrument for continuous
                                   sequence threshold calibration. Default: 1.
  --activity-threshold X            Activity threshold. Default: 0.35.
  --onset-threshold X               Fallback onset threshold. Default: 0.35.
  --offset-threshold X              Fallback offset threshold. Default: 0.35.
  --onset-tolerance-frames N         Onset probe tolerance override.
  --offset-tolerance-frames N        Offset probe tolerance override.
  --gcs-checkpoint-prefix URI       Checkpoint upload prefix to record in summary.
  --force                           Re-run steps even when outputs exist.
  --skip-preview-if-missing         Skip 60s metrics instead of failing if WAV is absent.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      run_dir="$2"
      shift 2
      ;;
    --checkpoint)
      checkpoint="$2"
      shift 2
      ;;
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --preview-wav)
      preview_wav="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --single-notes-per-family)
      single_notes_per_family="$2"
      shift 2
      ;;
    --melodies-per-instrument)
      melodies_per_instrument="$2"
      shift 2
      ;;
    --max-melody-instruments)
      max_melody_instruments="$2"
      shift 2
      ;;
    --sequence-calibration-max-instruments)
      sequence_calibration_max_instruments="$2"
      shift 2
      ;;
    --sequence-calibration-melodies-per-instrument)
      sequence_calibration_melodies_per_instrument="$2"
      shift 2
      ;;
    --activity-threshold)
      activity_threshold="$2"
      shift 2
      ;;
    --onset-threshold)
      onset_threshold="$2"
      shift 2
      ;;
    --offset-threshold)
      offset_threshold="$2"
      shift 2
      ;;
    --onset-tolerance-frames)
      onset_tolerance_frames="$2"
      shift 2
      ;;
    --offset-tolerance-frames)
      offset_tolerance_frames="$2"
      shift 2
      ;;
    --gcs-checkpoint-prefix)
      gcs_checkpoint_prefix="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --skip-preview-if-missing)
      skip_preview_if_missing=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$run_dir" ]]; then
  echo "--run-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$checkpoint" ]]; then
  checkpoint="$run_dir/checkpoint.pt"
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "checkpoint not found: $checkpoint" >&2
  exit 1
fi
if [[ ! -d "$data_root" ]]; then
  echo "data root not found: $data_root" >&2
  exit 1
fi

run_python() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}" python "$@"
  fi
}

heldout_common=(
  --checkpoint "$checkpoint"
  --data-root "$data_root"
  --single-notes-per-family "$single_notes_per_family"
  --melodies-per-instrument "$melodies_per_instrument"
  --max-melody-instruments "$max_melody_instruments"
  --activity-threshold "$activity_threshold"
  --onset-threshold "$onset_threshold"
  --offset-threshold "$offset_threshold"
  --batch-size "$batch_size"
  --device "$device"
)
if [[ -n "$onset_tolerance_frames" ]]; then
  heldout_common+=(--onset-tolerance-frames "$onset_tolerance_frames")
fi
if [[ -n "$offset_tolerance_frames" ]]; then
  heldout_common+=(--offset-tolerance-frames "$offset_tolerance_frames")
fi

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
calibration="$run_dir/calibration/onset_offset_thresholds.json"
validation_metrics="$run_dir/eval_validation/results/heldout_metrics.json"
test_metrics="$run_dir/eval_test_calibrated/results/heldout_metrics.json"
preview_metrics="$run_dir/eval_phase2_preview_60s/boundary_timing_metrics.json"
summary="$run_dir/final_evaluation_summary.json"

if [[ "$force" -eq 1 || ! -f "$calibration" || ! -f "$validation_metrics" ]]; then
  mkdir -p "$(dirname "$calibration")"
  run_python scripts/evaluate-nsynth-heldout.py \
    "${heldout_common[@]}" \
    --output-dir "$run_dir/eval_validation" \
    --split validation \
    --calibrate-thresholds \
    --calibration-output "$calibration"
else
  echo "validation calibration already exists: $calibration"
fi

sequence_calibration_needed=1
if [[ -f "$calibration" ]]; then
  sequence_calibration_needed="$(
    run_python - "$calibration" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    calibration = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(1)
    raise SystemExit
sequence = calibration.get("sequence")
if isinstance(sequence, dict) and isinstance(sequence.get("global"), dict):
    global_entry = sequence["global"]
    onset = global_entry.get("onset")
    offset = global_entry.get("offset")
    if (
        isinstance(onset, dict)
        and "threshold" in onset
        and isinstance(offset, dict)
        and "threshold" in offset
    ):
        print(0)
        raise SystemExit
print(1)
PY
  )"
fi

if [[ "$force" -eq 1 || "$sequence_calibration_needed" -eq 1 ]]; then
  run_python scripts/evaluate-boundary-timing-sequence.py \
    --checkpoint "$checkpoint" \
    --data-root "$data_root" \
    --calibration-input "$calibration" \
    --calibration-output "$calibration" \
    --calibrate-sequence-thresholds \
    --calibration-split validation \
    --calibration-melodies-per-instrument "$sequence_calibration_melodies_per_instrument" \
    --calibration-max-instruments "$sequence_calibration_max_instruments" \
    --seconds 60 \
    --device "$device" \
    --sample-stride-ms 5
else
  echo "sequence calibration already exists in: $calibration"
fi

if [[ "$force" -eq 1 || ! -f "$test_metrics" ]]; then
  run_python scripts/evaluate-nsynth-heldout.py \
    "${heldout_common[@]}" \
    --output-dir "$run_dir/eval_test_calibrated" \
    --split test \
    --calibration-input "$calibration"
else
  echo "calibrated test metrics already exist: $test_metrics"
fi

if [[ -f "$preview_wav" ]]; then
  if [[ "$force" -eq 1 || ! -f "$preview_metrics" ]]; then
    run_python scripts/evaluate-boundary-timing-sequence.py \
      --checkpoint "$checkpoint" \
      --data-root "$data_root" \
      --wav "$preview_wav" \
      --calibration-input "$calibration" \
      --device "$device" \
      --sample-stride-ms 5 \
      --output-json "$preview_metrics"
  else
    echo "60s preview metrics already exist: $preview_metrics"
  fi
elif [[ "$skip_preview_if_missing" -eq 1 ]]; then
  echo "preview wav not found, skipping 60s sequence metrics: $preview_wav" >&2
else
  echo "preview wav not found: $preview_wav" >&2
  exit 1
fi

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_python - \
  "$summary" \
  "$run_dir" \
  "$checkpoint" \
  "$data_root" \
  "$preview_wav" \
  "$calibration" \
  "$validation_metrics" \
  "$test_metrics" \
  "$preview_metrics" \
  "$gcs_checkpoint_prefix" \
  "$started_at" \
  "$finished_at" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    run_dir,
    checkpoint,
    data_root,
    preview_wav,
    calibration_path,
    validation_metrics_path,
    test_metrics_path,
    preview_metrics_path,
    gcs_checkpoint_prefix,
    started_at,
    finished_at,
) = [Path(value) if idx < 9 else value for idx, value in enumerate(sys.argv[1:])]


def load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compact_phase(heldout: object | None, phase: str) -> dict[str, object]:
    if not isinstance(heldout, dict):
        return {}
    phase_payload = heldout.get(phase)
    if not isinstance(phase_payload, dict):
        return {}
    overall = phase_payload.get("overall")
    if not isinstance(overall, dict):
        return {}
    keys = (
        "active_points",
        "detection_recall",
        "source_count_accuracy",
        "family_accuracy_detected",
        "onset_points",
        "onset_recall",
        "onset_timing_coverage",
        "onset_timing_mae_ms",
        "onset_timing_rmsd_ms",
        "offset_points",
        "offset_recall",
        "offset_timing_coverage",
        "offset_timing_mae_ms",
        "offset_timing_rmsd_ms",
        "id_switches_per_minute",
        "slot_stability",
    )
    return {key: overall.get(key) for key in keys if key in overall}


def compact_heldout(heldout: object | None) -> dict[str, object]:
    return {
        "phase_1_single_note": compact_phase(heldout, "phase_1_single_note"),
        "phase_2_single_instrument_melody": compact_phase(
            heldout,
            "phase_2_single_instrument_melody",
        ),
    }


calibration = load_json(calibration_path)
validation_metrics = load_json(validation_metrics_path)
test_metrics = load_json(test_metrics_path)
preview_metrics = load_json(preview_metrics_path)
artifacts = {
    "checkpoint": str(checkpoint),
    "calibration": str(calibration_path),
    "validation_metrics": str(validation_metrics_path),
    "test_metrics": str(test_metrics_path),
    "preview_60s_metrics": str(preview_metrics_path)
    if preview_metrics_path.exists()
    else None,
    "gcs_checkpoint_prefix": str(gcs_checkpoint_prefix),
}
summary = {
    "started_at": str(started_at),
    "finished_at": str(finished_at),
    "run_dir": str(run_dir),
    "inputs": {
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "preview_wav": str(preview_wav),
    },
    "artifacts": artifacts,
    "calibration": calibration,
    "key_metrics": {
        "validation": compact_heldout(validation_metrics),
        "test": compact_heldout(test_metrics),
        "preview_60s": preview_metrics,
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(artifacts, indent=2, sort_keys=True))
PY

echo "final evaluation summary: $summary"
