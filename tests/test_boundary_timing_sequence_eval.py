from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sequence_eval_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate-boundary-timing-sequence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_boundary_timing_sequence",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prediction_peaks_require_local_maxima_before_refractory_suppression():
    evaluator = _load_sequence_eval_module().BoundaryTimingSequenceEvaluator
    rows = [
        {"frame_idx": 0, "onset_score": 0.1},
        {"frame_idx": 1, "onset_score": 0.7},
        {"frame_idx": 2, "onset_score": 0.9},
        {"frame_idx": 3, "onset_score": 0.8},
        {"frame_idx": 12, "onset_score": 0.65},
        {"frame_idx": 13, "onset_score": 0.6},
        {"frame_idx": 30, "onset_score": 0.95},
        {"frame_idx": 31, "onset_score": 0.9},
    ]

    peaks = evaluator._prediction_peaks(
        rows,
        "onset",
        threshold=0.5,
        refractory=10,
    )

    assert [row["frame_idx"] for row in peaks] == [2, 30]


def test_threshold_prefers_sequence_calibration_over_sparse_probe_calibration():
    evaluator_cls = _load_sequence_eval_module().BoundaryTimingSequenceEvaluator
    evaluator = object.__new__(evaluator_cls)
    evaluator.onset_threshold = None
    evaluator.offset_threshold = None
    evaluator.calibration = {
        "global": {"onset": {"threshold": 0.1}},
        "sequence": {"global": {"onset": {"threshold": 0.7}}},
    }

    assert evaluator._threshold("guitar", "onset") == 0.7


def test_onset_sequence_matching_uses_shoulder_tolerance():
    module = _load_sequence_eval_module()
    evaluator_cls = module.BoundaryTimingSequenceEvaluator

    assert evaluator_cls._match_tolerance_frames("onset") == (
        module.SourceTrackingAudioConfigV2.onset_shoulder_radius_frames
    )


def test_sequence_threshold_search_penalizes_false_peaks():
    module = _load_sequence_eval_module()
    evaluator = object.__new__(module.BoundaryTimingSequenceEvaluator)
    evaluator.event_refractory_ms = 200.0
    evaluator.onset_threshold = None
    evaluator.offset_threshold = None
    hop = module.SourceTrackingAudioConfigV2.hop
    sample_rate = module.SourceTrackingAudioConfigV2.sample_rate
    boundary_frame = 10
    event = module.SourceEventLabelV2(
        source_id=0,
        family="guitar",
        family_index=0,
        start_s=boundary_frame * hop / sample_rate,
        end_s=(boundary_frame + 50) * hop / sample_rate,
    )
    predictions = [
        {
            "frame_idx": 9,
            "time_s": 9 * hop / sample_rate,
            "onset_score": 0.1,
            "onset_delta_frames": 0.0,
        },
        {
            "frame_idx": 10,
            "time_s": 10 * hop / sample_rate,
            "onset_score": 0.9,
            "onset_delta_frames": 0.0,
        },
        {
            "frame_idx": 11,
            "time_s": 11 * hop / sample_rate,
            "onset_score": 0.1,
            "onset_delta_frames": 0.0,
        },
        {
            "frame_idx": 50,
            "time_s": 50 * hop / sample_rate,
            "onset_score": 0.8,
            "onset_delta_frames": 0.0,
        },
        {
            "frame_idx": 51,
            "time_s": 51 * hop / sample_rate,
            "onset_score": 0.1,
            "onset_delta_frames": 0.0,
        },
    ]

    best = evaluator._best_sequence_threshold(
        [([event], predictions, "guitar")],
        "onset",
    )

    assert best["threshold"] == 0.9
    assert best["f1"] == 1.0
    assert best["average_precision"] == 1.0
