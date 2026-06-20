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
