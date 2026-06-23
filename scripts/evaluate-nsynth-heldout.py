from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    FamilyVocabularyV2,
    NsynthNoteStoreV2,
    SourceTrackingAudioConfigV2,
    SourceTrackingRendererV2,
)
from muziq_nn.datasets.schema import NsynthNoteV2, SourceEventLabelV2
from muziq_nn.webapp.inference import SourceTrackingCheckpointLoaderV2


@dataclass(frozen=True)
class HeldoutEvalConfigV2:
    checkpoint: Path
    data_root: Path
    output_dir: Path
    single_notes_per_family: int = 0
    melodies_per_instrument: int = 1
    max_melody_instruments: int = 0
    activity_threshold: float = 0.35
    onset_threshold: float = 0.35
    offset_threshold: float = 0.35
    onset_tolerance_frames: int = SourceTrackingAudioConfigV2.onset_shoulder_radius_frames
    offset_tolerance_frames: int = SourceTrackingAudioConfigV2.offset_label_radius_frames
    batch_size: int = 1024
    device: str = "auto"
    split: Literal["train", "validation", "test"] = "test"
    calibration_input: Path | None = None
    calibration_output: Path | None = None
    calibrate_thresholds: bool = False


class NsynthHeldoutEvaluatorV2:
    def __init__(self, config: HeldoutEvalConfigV2):
        self.config = config
        self.families = FamilyVocabularyV2()
        self.rng = np.random.default_rng(20260619)
        self.results_path = config.output_dir / "results" / "heldout_metrics.json"
        self.checkpoint = SourceTrackingCheckpointLoaderV2(
            config.checkpoint,
            device=config.device,
        )
        self.calibration = self._load_calibration(config.calibration_input)

    def run(self) -> dict[str, object]:
        started = time.monotonic()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        index = NsynthIndexV2(self.config.data_root)
        note_store = NsynthNoteStoreV2(index)
        renderer = SourceTrackingRendererV2(note_store)
        notes = index.notes(self.config.split)
        notes_by_family = self._notes_by_family(notes)
        notes_by_instrument = self._notes_by_instrument(notes)
        single_note_rows = self._single_note_eval_rows(renderer, notes_by_family)
        melody_rows = self._melody_eval_rows(renderer, notes_by_instrument)
        if self.config.calibrate_thresholds:
            self.calibration = self._calibrate_thresholds(single_note_rows + melody_rows)
            if self.config.calibration_output is not None:
                self.config.calibration_output.parent.mkdir(parents=True, exist_ok=True)
                self.config.calibration_output.write_text(
                    json.dumps(self.calibration, indent=2), encoding="utf-8"
                )
        payload = {
            "checkpoint": str(self.config.checkpoint),
            "data_root": str(self.config.data_root),
            "split": self.config.split,
            "activity_threshold": self.config.activity_threshold,
            "onset_threshold": self.config.onset_threshold,
            "offset_threshold": self.config.offset_threshold,
            "onset_tolerance_frames": self.config.onset_tolerance_frames,
            "onset_metric_label": "onset_shoulder_radius_frames",
            "offset_tolerance_frames": self.config.offset_tolerance_frames,
            "calibration": self.calibration,
            "note_count": len(notes),
            "instrument_count": len(notes_by_instrument),
            "family_counts": {
                family: len(notes_by_family.get(family, []))
                for family in self.families.families
            },
            "phase_1_single_note": self._summarize_rows(
                single_note_rows, include_switches=False
            ),
            "phase_2_single_instrument_melody": self._summarize_rows(
                melody_rows, include_switches=True
            ),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        self.results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    @staticmethod
    def _notes_by_family(
        notes: list[NsynthNoteV2],
    ) -> dict[str, list[NsynthNoteV2]]:
        grouped: dict[str, list[NsynthNoteV2]] = defaultdict(list)
        for note in notes:
            grouped[note.family].append(note)
        for family_notes in grouped.values():
            family_notes.sort(key=lambda note: note.note_str)
        return dict(grouped)

    @staticmethod
    def _notes_by_instrument(
        notes: list[NsynthNoteV2],
    ) -> dict[str, list[NsynthNoteV2]]:
        grouped: dict[str, list[NsynthNoteV2]] = defaultdict(list)
        for note in notes:
            grouped[note.instrument_str].append(note)
        for instrument_notes in grouped.values():
            instrument_notes.sort(key=lambda note: note.note_str)
        return dict(sorted(grouped.items()))

    def _single_note_eval_rows(
        self,
        renderer: SourceTrackingRendererV2,
        notes_by_family: dict[str, list[NsynthNoteV2]],
    ) -> list[dict[str, object]]:
        rows = []
        for family in self.families.families:
            notes = notes_by_family.get(family, [])
            if self.config.single_notes_per_family > 0:
                notes = notes[: self.config.single_notes_per_family]
            rows.extend(self._single_note_rows(renderer, family, notes))
        return rows

    def _single_note_rows(
        self,
        renderer: SourceTrackingRendererV2,
        family: str,
        notes: list[NsynthNoteV2],
    ) -> list[dict[str, object]]:
        rows = []
        for note_idx, note in enumerate(notes):
            audio, events = self._audio_for_note(renderer, note, start_s=0.75)
            rows.extend(
                self._prediction_rows(
                    renderer,
                    family,
                    f"single_note:{note.note_str}:{note_idx}",
                    audio,
                    events,
                )
            )
        return rows

    def _melody_eval_rows(
        self,
        renderer: SourceTrackingRendererV2,
        notes_by_instrument: dict[str, list[NsynthNoteV2]],
    ) -> list[dict[str, object]]:
        rows = []
        instrument_items = list(notes_by_instrument.items())
        if self.config.max_melody_instruments > 0:
            instrument_items = instrument_items[: self.config.max_melody_instruments]
        for instrument_idx, (instrument, notes) in enumerate(instrument_items):
            for melody_idx in range(self.config.melodies_per_instrument):
                audio, events = self._audio_for_melody(
                    renderer,
                    notes,
                    seed=instrument_idx * 10_003 + melody_idx,
                )
                rows.extend(
                    self._prediction_rows(
                        renderer,
                        notes[0].family,
                        f"melody:{instrument}:{melody_idx}",
                        audio,
                        events,
                    )
                )
        return rows

    def _audio_for_note(
        self,
        renderer: SourceTrackingRendererV2,
        note: NsynthNoteV2,
        start_s: float,
    ) -> tuple[np.ndarray, list[SourceEventLabelV2]]:
        audio = np.zeros(
            int(
                SourceTrackingAudioConfigV2.duration_s
                * SourceTrackingAudioConfigV2.sample_rate
            ),
            dtype=np.float32,
        )
        events: list[SourceEventLabelV2] = []
        renderer._mix_note(audio, note, 0, start_s, events)
        return self._normalize_audio(audio), events

    def _audio_for_melody(
        self,
        renderer: SourceTrackingRendererV2,
        notes: list[NsynthNoteV2],
        seed: int,
    ) -> tuple[np.ndarray, list[SourceEventLabelV2]]:
        rng = np.random.default_rng(91_337 + seed)
        audio = np.zeros(
            int(
                SourceTrackingAudioConfigV2.duration_s
                * SourceTrackingAudioConfigV2.sample_rate
            ),
            dtype=np.float32,
        )
        events: list[SourceEventLabelV2] = []
        period = float(rng.uniform(0.45, 0.95))
        t = 0.5
        while t < SourceTrackingAudioConfigV2.duration_s - 1.0:
            pitch = int(rng.integers(36, 88))
            note = min(
                notes,
                key=lambda item: (abs(item.pitch - pitch), abs(item.velocity - 96)),
            )
            renderer._mix_note(audio, note, 0, t, events)
            t += period + float(rng.uniform(-0.05, 0.05))
        return self._normalize_audio(audio), events

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            return (audio / peak * 0.8).astype(np.float32)
        return audio.astype(np.float32)

    def _prediction_rows(
        self,
        renderer: SourceTrackingRendererV2,
        family: str,
        example_id: str,
        audio: np.ndarray,
        events: list[SourceEventLabelV2],
    ) -> list[dict[str, object]]:
        contexts = []
        row_specs = []
        for event_idx, event in enumerate(events):
            for kind, frame_idx, boundary_frame in self._event_probe_frames(event):
                contexts.append(self._fixed_context(renderer, audio, frame_idx))
                row_specs.append((event_idx, kind, frame_idx, boundary_frame))
        predictions = self._predict_context_batch(contexts)
        rows = []
        true_family_index = self.families.index(family)
        for (event_idx, kind, frame_idx, boundary_frame), pred in zip(
            row_specs, predictions, strict=False
        ):
            rows.append(
                {
                    "family": family,
                    "example_id": example_id,
                    "event_idx": event_idx,
                    "probe": kind,
                    "frame_idx": frame_idx,
                    "boundary_frame": boundary_frame,
                    "true_family_index": true_family_index,
                    **pred,
                }
            )
        return rows

    @staticmethod
    def _fixed_context(
        renderer: SourceTrackingRendererV2,
        audio: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        context = renderer.extractor.extract_context(
            audio,
            end_frame=frame_idx,
            frame_count=256,
            peak_warmup_frames=256,
        )
        if context.shape[0] >= 256:
            return context[-256:].astype(np.float32)
        pad = np.zeros((256 - context.shape[0], context.shape[1]), dtype=np.float32)
        return np.vstack((pad, context)).astype(np.float32)

    def _event_probe_frames(
        self,
        event: SourceEventLabelV2,
    ) -> tuple[tuple[Literal["onset", "active", "offset"], int, int | None], ...]:
        hop = SourceTrackingAudioConfigV2.hop
        sample_rate = SourceTrackingAudioConfigV2.sample_rate
        start_frame = max(0, int(math.floor(event.start_s * sample_rate / hop)))
        end_frame = max(start_frame, int(math.floor(event.end_s * sample_rate / hop)))
        offset_frame = max(start_frame, end_frame - 1)
        mid_frame = (start_frame + end_frame) // 2
        onset_start = max(0, start_frame - self.config.onset_tolerance_frames)
        onset_end = start_frame + self.config.onset_tolerance_frames + 1
        onset_frames = tuple(
            ("onset", frame, start_frame) for frame in range(onset_start, onset_end)
        )
        offset_start = max(0, offset_frame - self.config.offset_tolerance_frames)
        offset_end = offset_frame + self.config.offset_tolerance_frames + 1
        offset_frames = tuple(
            ("offset", frame, offset_frame) for frame in range(offset_start, offset_end)
        )
        return (
            *onset_frames,
            ("active", mid_frame, None),
            *offset_frames,
        )

    def _predict_context_batch(
        self,
        contexts: list[np.ndarray],
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for start in range(0, len(contexts), self.config.batch_size):
            batch = np.asarray(
                contexts[start : start + self.config.batch_size],
                dtype=np.float32,
            )
            tensor = torch.from_numpy(batch).to(self.checkpoint.device)
            with torch.inference_mode():
                outputs = self.checkpoint.predict_sequence(
                    tensor,
                    onset_threshold=self.config.onset_threshold,
                    offset_threshold=self.config.offset_threshold,
                )
            activity = torch.sigmoid(outputs["activity_logits"]).cpu().numpy()
            family_probs = torch.softmax(outputs["family_logits"], dim=-1).cpu().numpy()
            onset = torch.sigmoid(
                self.checkpoint.event_logits(outputs, "onset")
            ).cpu().numpy()
            offset = torch.sigmoid(
                self.checkpoint.event_logits(outputs, "offset")
            ).cpu().numpy()
            onset_delta = outputs["onset_delta"].cpu().numpy()
            offset_delta = outputs["offset_delta"].cpu().numpy()
            count_logits = outputs.get("count_logits")
            predicted_counts = (
                torch.argmax(count_logits, dim=-1).detach().cpu().numpy()
                if count_logits is not None
                and getattr(self.checkpoint, "has_count_head", False)
                else None
            )
            for item_idx in range(activity.shape[0]):
                best_slot = int(np.argmax(activity[item_idx]))
                family_index = int(np.argmax(family_probs[item_idx, best_slot]))
                predicted_active_count = (
                    int(predicted_counts[item_idx])
                    if predicted_counts is not None
                    else int(np.sum(activity[item_idx] >= self.config.activity_threshold))
                )
                out.append(
                    {
                        "predicted_active_count": predicted_active_count,
                        "best_slot": best_slot,
                        "best_activity": float(activity[item_idx, best_slot]),
                        "predicted_family_index": family_index,
                        "predicted_family_confidence": float(
                            family_probs[item_idx, best_slot, family_index]
                        ),
                        "onset_score": float(onset[item_idx, best_slot]),
                        "offset_score": float(offset[item_idx, best_slot]),
                        "onset_delta_frames": float(onset_delta[item_idx, best_slot]),
                        "offset_delta_frames": float(offset_delta[item_idx, best_slot]),
                    }
                )
        return out

    def _summarize_rows(
        self,
        rows: list[dict[str, object]],
        include_switches: bool,
    ) -> dict[str, object]:
        by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_family[str(row["family"])].append(row)
        return {
            "overall": self._summarize_family(rows, include_switches),
            "by_family": {
                family: self._summarize_family(
                    by_family.get(family, []),
                    include_switches,
                )
                for family in self.families.families
            },
        }

    def _summarize_family(
        self,
        rows: list[dict[str, object]],
        include_switches: bool,
    ) -> dict[str, object]:
        if not rows:
            return self._empty_metrics(include_switches)
        active_rows = [row for row in rows if row["probe"] == "active"]
        onset_rows = [row for row in rows if row["probe"] == "onset"]
        offset_events = self._best_rows_by_event(
            [row for row in rows if row["probe"] == "offset"],
            "offset_score",
        )
        detected_active = [
            row
            for row in active_rows
            if float(row["best_activity"]) >= self.config.activity_threshold
        ]
        metrics = {
            "active_points": len(active_rows),
            "detection_recall": self._mean(
                float(row["best_activity"]) >= self.config.activity_threshold
                for row in active_rows
            ),
            "source_count_accuracy": self._mean(
                int(row["predicted_active_count"]) == 1 for row in active_rows
            ),
            "family_accuracy_all_active": self._mean(
                int(row["predicted_family_index"]) == int(row["true_family_index"])
                for row in active_rows
            ),
            "family_accuracy_detected": self._mean(
                int(row["predicted_family_index"]) == int(row["true_family_index"])
                for row in detected_active
            ),
            "mean_predicted_active_count": self._mean_float(
                float(row["predicted_active_count"]) for row in active_rows
            ),
            "predicted_active_count_histogram": self._histogram(
                int(row["predicted_active_count"]) for row in active_rows
            ),
            "dominant_predicted_family": self._dominant_family(active_rows),
            "mean_best_activity": self._mean_float(
                float(row["best_activity"]) for row in active_rows
            ),
            "mean_predicted_family_confidence": self._mean_float(
                float(row["predicted_family_confidence"]) for row in active_rows
            ),
            "onset_points": len(onset_rows),
            "onset_recall": self._mean(
                float(row["onset_score"]) >= self._threshold(str(row["family"]), "onset")
                for row in onset_rows
            ),
            "mean_onset_score": self._mean_float(
                float(row["onset_score"]) for row in onset_rows
            ),
            "offset_points": len(offset_events),
            "offset_recall": self._mean(
                float(row["offset_score"]) >= self._threshold(str(row["family"]), "offset")
                for row in offset_events
            ),
            "mean_offset_score": self._mean_float(
                float(row["offset_score"]) for row in offset_events
            ),
        }
        metrics.update(self._timing_metrics(onset_rows, "onset"))
        metrics.update(self._timing_metrics(offset_events, "offset"))
        if include_switches:
            metrics.update(self._switch_metrics(active_rows))
        return metrics

    def _timing_metrics(
        self,
        rows: list[dict[str, object]],
        kind: Literal["onset", "offset"],
    ) -> dict[str, object]:
        threshold = {
            row_key: self._threshold(str(row["family"]), kind)
            for row_key, row in enumerate(rows)
        }
        matched = [
            row
            for row_key, row in enumerate(rows)
            if float(row[f"{kind}_score"]) >= threshold[row_key]
            and row.get("boundary_frame") is not None
        ]
        deltas = [
            self._predicted_boundary_error_frames(row, kind)
            for row in matched
        ]
        prefix = f"{kind}_timing"
        if not deltas:
            return {
                f"{prefix}_matched_points": 0,
                f"{prefix}_coverage": 0.0 if rows else None,
                f"{prefix}_mae_frames": None,
                f"{prefix}_rmsd_frames": None,
                f"{prefix}_mae_ms": None,
                f"{prefix}_rmsd_ms": None,
            }
        values = np.asarray(deltas, dtype=np.float32)
        hop_ms = (
            1000.0
            * SourceTrackingAudioConfigV2.hop
            / SourceTrackingAudioConfigV2.sample_rate
        )
        return {
            f"{prefix}_matched_points": len(deltas),
            f"{prefix}_coverage": len(deltas) / len(rows) if rows else None,
            f"{prefix}_mae_frames": float(np.mean(np.abs(values))),
            f"{prefix}_rmsd_frames": float(np.sqrt(np.mean(values * values))),
            f"{prefix}_mae_ms": float(np.mean(np.abs(values)) * hop_ms),
            f"{prefix}_rmsd_ms": float(np.sqrt(np.mean(values * values)) * hop_ms),
        }

    @staticmethod
    def _predicted_boundary_error_frames(
        row: dict[str, object],
        kind: Literal["onset", "offset"],
    ) -> float:
        predicted_boundary = float(row["frame_idx"]) - float(row[f"{kind}_delta_frames"])
        return predicted_boundary - float(row["boundary_frame"])

    def _switch_metrics(self, active_rows: list[dict[str, object]]) -> dict[str, float]:
        by_example: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in active_rows:
            if float(row["best_activity"]) >= self.config.activity_threshold:
                by_example[str(row["example_id"])].append(row)
        switch_rates = []
        stability = []
        for rows in by_example.values():
            ordered = sorted(rows, key=lambda row: int(row["frame_idx"]))
            slots = [int(row["best_slot"]) for row in ordered]
            if not slots:
                continue
            switches = sum(
                1 for prev, cur in zip(slots, slots[1:], strict=False) if prev != cur
            )
            switch_rates.append(switches / (SourceTrackingAudioConfigV2.duration_s / 60.0))
            counts = Counter(slots)
            stability.append(counts.most_common(1)[0][1] / len(slots))
        return {
            "id_switches_per_minute": self._mean_float(iter(switch_rates)),
            "slot_stability": self._mean_float(iter(stability)),
        }

    @staticmethod
    def _best_rows_by_event(
        rows: list[dict[str, object]],
        score_key: str,
    ) -> list[dict[str, object]]:
        by_event: dict[tuple[str, int], dict[str, object]] = {}
        for row in rows:
            key = (str(row["example_id"]), int(row["event_idx"]))
            previous = by_event.get(key)
            if previous is None or float(row[score_key]) > float(previous[score_key]):
                by_event[key] = row
        return list(by_event.values())

    def _calibrate_thresholds(
        self,
        rows: list[dict[str, object]],
    ) -> dict[str, object]:
        calibration: dict[str, object] = {
            "split": self.config.split,
            "global": {
                "onset": self._best_threshold(rows, "onset"),
                "offset": self._best_threshold(rows, "offset"),
            },
            "by_family": {},
        }
        by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_family[str(row["family"])].append(row)
        calibration["by_family"] = {
            family: {
                "onset": self._best_threshold(by_family.get(family, []), "onset"),
                "offset": self._best_threshold(by_family.get(family, []), "offset"),
            }
            for family in self.families.families
        }
        return calibration

    def _best_threshold(
        self,
        rows: list[dict[str, object]],
        kind: Literal["onset", "offset"],
    ) -> dict[str, float | int]:
        if not rows:
            return {"threshold": self._default_threshold(kind), "f1": 0.0, "positives": 0}
        score_key = f"{kind}_score"
        scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
        positives_mask = np.asarray([row["probe"] == kind for row in rows], dtype=bool)
        candidates = np.unique(
            np.concatenate(
                (
                    np.asarray(
                        [
                            self._default_threshold(kind),
                            0.05,
                            0.1,
                            0.2,
                            0.35,
                            0.5,
                            0.65,
                            0.8,
                        ],
                        dtype=np.float64,
                    ),
                    scores,
                )
            )
        )
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        positive_prefix = np.concatenate(
            ([0], np.cumsum(positives_mask[order].astype(np.int64)))
        )
        below_threshold = np.searchsorted(sorted_scores, candidates, side="left")
        predicted = scores.size - below_threshold
        positives = int(positive_prefix[-1])
        tp = positives - positive_prefix[below_threshold]
        fp = predicted - tp
        fn = positives - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.divide(
                tp,
                tp + fp,
                out=np.zeros_like(tp, dtype=np.float64),
                where=(tp + fp) > 0,
            )
            recall = np.divide(
                tp,
                tp + fn,
                out=np.zeros_like(tp, dtype=np.float64),
                where=(tp + fn) > 0,
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision, dtype=np.float64),
                where=(precision + recall) > 0,
            )
        best_idx = int(np.lexsort((candidates, recall, f1))[-1])
        return {
            "threshold": float(candidates[best_idx]),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1[best_idx]),
            "positives": positives,
        }

    def _threshold(self, family: str, kind: Literal["onset", "offset"]) -> float:
        by_family = self.calibration.get("by_family") if self.calibration else None
        if isinstance(by_family, dict):
            family_entry = by_family.get(family)
            if isinstance(family_entry, dict):
                kind_entry = family_entry.get(kind)
                if isinstance(kind_entry, dict) and "threshold" in kind_entry:
                    return float(kind_entry["threshold"])
        global_entry = self.calibration.get("global") if self.calibration else None
        if isinstance(global_entry, dict):
            kind_entry = global_entry.get(kind)
            if isinstance(kind_entry, dict) and "threshold" in kind_entry:
                return float(kind_entry["threshold"])
        return self._default_threshold(kind)

    def _default_threshold(self, kind: Literal["onset", "offset"]) -> float:
        if kind == "onset":
            return self.config.onset_threshold
        return self.config.offset_threshold

    @staticmethod
    def _load_calibration(path: Path | None) -> dict[str, object]:
        if path is None:
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _empty_metrics(include_switches: bool) -> dict[str, object]:
        metrics: dict[str, object] = {
            "active_points": 0,
            "detection_recall": None,
            "source_count_accuracy": None,
            "family_accuracy_all_active": None,
            "family_accuracy_detected": None,
            "mean_predicted_active_count": None,
            "predicted_active_count_histogram": {},
            "dominant_predicted_family": None,
            "mean_best_activity": None,
            "mean_predicted_family_confidence": None,
            "onset_points": 0,
            "onset_recall": None,
            "mean_onset_score": None,
            "onset_timing_matched_points": 0,
            "onset_timing_coverage": None,
            "onset_timing_mae_frames": None,
            "onset_timing_rmsd_frames": None,
            "onset_timing_mae_ms": None,
            "onset_timing_rmsd_ms": None,
            "offset_points": 0,
            "offset_recall": None,
            "mean_offset_score": None,
            "offset_timing_matched_points": 0,
            "offset_timing_coverage": None,
            "offset_timing_mae_frames": None,
            "offset_timing_rmsd_frames": None,
            "offset_timing_mae_ms": None,
            "offset_timing_rmsd_ms": None,
        }
        if include_switches:
            metrics["id_switches_per_minute"] = None
            metrics["slot_stability"] = None
        return metrics

    @staticmethod
    def _mean(values) -> float | None:
        collected = [bool(value) for value in values]
        if not collected:
            return None
        return float(sum(collected) / len(collected))

    @staticmethod
    def _mean_float(values) -> float | None:
        collected = [float(value) for value in values]
        if not collected:
            return None
        return float(sum(collected) / len(collected))

    @staticmethod
    def _histogram(values) -> dict[str, int]:
        return {str(key): value for key, value in sorted(Counter(values).items())}

    def _dominant_family(self, rows: list[dict[str, object]]) -> str | None:
        if not rows:
            return None
        counts = Counter(int(row["predicted_family_index"]) for row in rows)
        family_index = counts.most_common(1)[0][0]
        if 0 <= family_index < len(self.families.families):
            return self.families.families[family_index]
        return f"family_{family_index}"


class NsynthHeldoutEvalCliV2:
    @staticmethod
    def run() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--data-root", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--single-notes-per-family", type=int, default=0)
        parser.add_argument("--melodies-per-instrument", type=int, default=1)
        parser.add_argument("--max-melody-instruments", type=int, default=0)
        parser.add_argument("--activity-threshold", type=float, default=0.35)
        parser.add_argument("--onset-threshold", type=float, default=0.35)
        parser.add_argument("--offset-threshold", type=float, default=0.35)
        parser.add_argument(
            "--onset-tolerance-frames",
            type=int,
            default=SourceTrackingAudioConfigV2.onset_shoulder_radius_frames,
        )
        parser.add_argument(
            "--offset-tolerance-frames",
            type=int,
            default=SourceTrackingAudioConfigV2.offset_label_radius_frames,
        )
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--device", default="auto")
        parser.add_argument(
            "--split",
            choices=("train", "validation", "test"),
            default="test",
        )
        parser.add_argument("--calibration-input", type=Path)
        parser.add_argument("--calibration-output", type=Path)
        parser.add_argument("--calibrate-thresholds", action="store_true")
        args = parser.parse_args()
        NsynthHeldoutEvaluatorV2(
            HeldoutEvalConfigV2(
                checkpoint=args.checkpoint,
                data_root=args.data_root,
                output_dir=args.output_dir,
                single_notes_per_family=args.single_notes_per_family,
                melodies_per_instrument=args.melodies_per_instrument,
                max_melody_instruments=args.max_melody_instruments,
                activity_threshold=args.activity_threshold,
                onset_threshold=args.onset_threshold,
                offset_threshold=args.offset_threshold,
                onset_tolerance_frames=args.onset_tolerance_frames,
                offset_tolerance_frames=args.offset_tolerance_frames,
                batch_size=args.batch_size,
                device=args.device,
                split=args.split,
                calibration_input=args.calibration_input,
                calibration_output=args.calibration_output,
                calibrate_thresholds=args.calibrate_thresholds,
            )
        ).run()


if __name__ == "__main__":
    NsynthHeldoutEvalCliV2.run()
