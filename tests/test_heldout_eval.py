from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_best_threshold_matches_brute_force_scan():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate-nsynth-heldout.py"
    spec = importlib.util.spec_from_file_location("evaluate_nsynth_heldout", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evaluator = object.__new__(module.NsynthHeldoutEvaluatorV2)
    evaluator.config = module.HeldoutEvalConfigV2(
        checkpoint=Path("checkpoint.pt"),
        data_root=Path("data"),
        output_dir=Path("out"),
        onset_threshold=0.35,
        offset_threshold=0.35,
    )
    rows = [
        {"probe": "onset", "onset_score": 0.91},
        {"probe": "active", "onset_score": 0.84},
        {"probe": "offset", "onset_score": 0.67},
        {"probe": "onset", "onset_score": 0.67},
        {"probe": "active", "onset_score": 0.51},
        {"probe": "onset", "onset_score": 0.42},
        {"probe": "offset", "onset_score": 0.35},
        {"probe": "active", "onset_score": 0.12},
    ]

    candidates = sorted(
        {0.35, 0.05, 0.1, 0.2, 0.5, 0.65, 0.8}
        | {row["onset_score"] for row in rows}
    )
    brute = {"threshold": 0.35, "f1": -1.0, "recall": 0.0}
    for threshold in candidates:
        tp = fp = fn = positives = 0
        for row in rows:
            positive = row["probe"] == "onset"
            predicted = row["onset_score"] >= threshold
            if positive:
                positives += 1
            if positive and predicted:
                tp += 1
            elif positive and not predicted:
                fn += 1
            elif not positive and predicted:
                fp += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if (f1, recall, threshold) > (brute["f1"], brute["recall"], brute["threshold"]):
            brute = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "positives": positives,
            }

    assert evaluator._best_threshold(rows, "onset") == brute
