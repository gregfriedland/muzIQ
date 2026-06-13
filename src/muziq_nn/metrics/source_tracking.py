"""Label-only metrics for source-tracking predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from muziq_nn.datasets.schema import SourceLabelFramesV2


@dataclass(frozen=True)
class SourcePredictionFramesV2:
    active: np.ndarray
    family: np.ndarray
    source_id: np.ndarray
    onset: np.ndarray
    offset: np.ndarray


@dataclass(frozen=True)
class SourceTrackingMetricsV2:
    source_count_accuracy: float
    activity_f1: float
    family_accuracy: float
    id_switches_per_minute: float
    track_purity: float
    track_fragmentation: float
    onset_f1: float
    offset_f1: float


class SourceTrackingLabelMetricsV2:
    """Compute metrics directly against frame labels."""

    def __init__(self, activity_threshold: float = 0.5):
        self.activity_threshold = activity_threshold

    def evaluate(
        self,
        labels: SourceLabelFramesV2,
        predictions: SourcePredictionFramesV2,
    ) -> SourceTrackingMetricsV2:
        true_active = labels.active > 0.5
        pred_active = predictions.active > self.activity_threshold
        return SourceTrackingMetricsV2(
            source_count_accuracy=self._source_count_accuracy(true_active, pred_active),
            activity_f1=self._binary_f1(true_active, pred_active),
            family_accuracy=self._family_accuracy(
                labels, predictions, true_active, pred_active
            ),
            id_switches_per_minute=self._id_switches_per_minute(labels, predictions),
            track_purity=self._track_purity(labels, predictions),
            track_fragmentation=self._track_fragmentation(labels, predictions),
            onset_f1=self._binary_f1(
                labels.onset > 0.5, predictions.onset > self.activity_threshold
            ),
            offset_f1=self._binary_f1(
                labels.offset > 0.5, predictions.offset > self.activity_threshold
            ),
        )

    @staticmethod
    def _binary_f1(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        tp = float(np.logical_and(true_mask, pred_mask).sum())
        fp = float(np.logical_and(~true_mask, pred_mask).sum())
        fn = float(np.logical_and(true_mask, ~pred_mask).sum())
        denom = 2.0 * tp + fp + fn
        return 1.0 if denom == 0.0 else 2.0 * tp / denom

    @staticmethod
    def _source_count_accuracy(true_active: np.ndarray, pred_active: np.ndarray) -> float:
        return float((true_active.sum(axis=1) == pred_active.sum(axis=1)).mean())

    @staticmethod
    def _family_accuracy(
        labels: SourceLabelFramesV2,
        predictions: SourcePredictionFramesV2,
        true_active: np.ndarray,
        pred_active: np.ndarray,
    ) -> float:
        mask = np.logical_and(true_active, pred_active)
        if not mask.any():
            return 1.0
        return float((labels.family[mask] == predictions.family[mask]).mean())

    @staticmethod
    def _id_switches_per_minute(
        labels: SourceLabelFramesV2,
        predictions: SourcePredictionFramesV2,
    ) -> float:
        switches = 0
        for slot in range(labels.source_id.shape[1]):
            true_ids = labels.source_id[:, slot]
            pred_ids = predictions.source_id[:, slot]
            active = labels.active[:, slot] > 0.5
            last_pair = None
            for true_id, pred_id, is_active in zip(true_ids, pred_ids, active, strict=False):
                if not is_active or true_id < 0 or pred_id < 0:
                    continue
                pair = (int(true_id), int(pred_id))
                if (
                    last_pair is not None
                    and pair[0] == last_pair[0]
                    and pair[1] != last_pair[1]
                ):
                    switches += 1
                last_pair = pair
        minutes = max(float(labels.frame_times[-1]) / 60.0, 1e-6)
        return switches / minutes

    @staticmethod
    def _track_purity(
        labels: SourceLabelFramesV2, predictions: SourcePredictionFramesV2
    ) -> float:
        purities = []
        for pred_id in sorted(set(predictions.source_id[predictions.source_id >= 0].tolist())):
            mask = predictions.source_id == pred_id
            true_ids = labels.source_id[mask]
            true_ids = true_ids[true_ids >= 0]
            if len(true_ids) == 0:
                continue
            counts = np.bincount(true_ids.astype(np.int64))
            purities.append(float(counts.max() / counts.sum()))
        return 1.0 if not purities else float(np.mean(purities))

    @staticmethod
    def _track_fragmentation(
        labels: SourceLabelFramesV2, predictions: SourcePredictionFramesV2
    ) -> float:
        fragments = []
        for true_id in sorted(set(labels.source_id[labels.source_id >= 0].tolist())):
            mask = labels.source_id == true_id
            pred_ids = predictions.source_id[mask]
            pred_ids = pred_ids[pred_ids >= 0]
            fragments.append(float(len(set(pred_ids.tolist()))))
        return 0.0 if not fragments else float(np.mean(fragments) - 1.0)
