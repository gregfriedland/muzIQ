from __future__ import annotations

import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    FamilyVocabularyV2,
    NsynthNoteStoreV2,
    SourceTrackingAudioConfigV2,
    SourceTrackingRendererV2,
)
from muziq_nn.datasets.schema import SourceEventLabelV2
from muziq_nn.webapp.inference import SourceTrackingCheckpointLoaderV2


class BoundaryTimingSequenceEvaluator:
    def __init__(
        self,
        checkpoint: Path,
        data_root: Path,
        wav: Path,
        output_json: Path,
        calibration_input: Path | None,
        seconds: float,
        test_instrument_index: int,
        device: str,
        sample_stride_ms: float,
        onset_threshold: float | None,
        offset_threshold: float | None,
        event_refractory_ms: float,
    ):
        self.checkpoint = checkpoint
        self.data_root = data_root
        self.wav = wav
        self.output_json = output_json
        self.calibration = self._load_calibration(calibration_input)
        self.seconds = seconds
        self.test_instrument_index = test_instrument_index
        self.device = device
        self.sample_stride_ms = sample_stride_ms
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.event_refractory_ms = event_refractory_ms
        self.families = FamilyVocabularyV2()

    def run(self) -> dict[str, object]:
        events, family = self._events()
        predictions = self._predictions()
        payload = {
            "checkpoint": str(self.checkpoint),
            "wav": str(self.wav),
            "data_root": str(self.data_root),
            "family": family,
            "event_count": len(events),
            "sample_stride_ms": self.sample_stride_ms,
            "calibration": self.calibration,
            "onset": self._event_metrics(events, predictions, family, "onset"),
            "offset": self._event_metrics(events, predictions, family, "offset"),
            "onset_events": self._event_detection_metrics(
                events,
                predictions,
                family,
                "onset",
            ),
            "offset_events": self._event_detection_metrics(
                events,
                predictions,
                family,
                "offset",
            ),
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def _events(self) -> tuple[list[SourceEventLabelV2], str]:
        index = NsynthIndexV2(self.data_root)
        renderer = SourceTrackingRendererV2(NsynthNoteStoreV2(index))
        notes_by_instrument = self._notes_by_instrument(index.notes("test"))
        instrument, notes = list(notes_by_instrument.items())[self.test_instrument_index]
        rng = np.random.default_rng(91337)
        audio = np.zeros(
            int(self.seconds * SourceTrackingAudioConfigV2.sample_rate),
            dtype=np.float32,
        )
        events: list[SourceEventLabelV2] = []
        t = 0.5
        while t < self.seconds - 1.0:
            period = float(rng.uniform(0.45, 0.95))
            pitch = int(rng.integers(36, 88))
            note = min(
                notes,
                key=lambda item: (abs(item.pitch - pitch), abs(item.velocity - 96)),
            )
            renderer._mix_note(audio, note, 0, t, events)
            t += period + float(rng.uniform(-0.05, 0.05))
        return events, notes[0].family

    @staticmethod
    def _notes_by_instrument(notes):
        grouped = defaultdict(list)
        for note in notes:
            grouped[note.instrument_str].append(note)
        for instrument_notes in grouped.values():
            instrument_notes.sort(key=lambda note: note.note_str)
        return dict(sorted(grouped.items()))

    def _predictions(self) -> list[dict[str, object]]:
        sample_rate, audio = self._read_wav()
        hop_ms = (
            1000.0
            * SourceTrackingAudioConfigV2.hop
            / SourceTrackingAudioConfigV2.sample_rate
        )
        sample_stride_frames = max(1, int(round(self.sample_stride_ms / hop_ms)))
        n_frames = int(np.ceil(len(audio) / SourceTrackingAudioConfigV2.hop))
        index = NsynthIndexV2(self.data_root)
        renderer = SourceTrackingRendererV2(NsynthNoteStoreV2(index))
        full = renderer.extractor.extract_context(
            audio,
            end_frame=n_frames - 1,
            frame_count=n_frames,
            peak_warmup_frames=256,
        ).astype(np.float32)
        frame_indices = list(range(0, n_frames, sample_stride_frames))
        if frame_indices[-1] != n_frames - 1:
            frame_indices.append(n_frames - 1)
        contexts = np.zeros((len(frame_indices), 256, full.shape[1]), dtype=np.float32)
        for row_idx, frame_idx in enumerate(frame_indices):
            start = max(0, frame_idx - 255)
            window = full[start : frame_idx + 1]
            contexts[row_idx, -len(window) :] = window
        checkpoint = SourceTrackingCheckpointLoaderV2(self.checkpoint, device=self.device)
        onset_state_threshold = (
            self.onset_threshold if self.onset_threshold is not None else 0.5
        )
        offset_state_threshold = (
            self.offset_threshold if self.offset_threshold is not None else 0.5
        )
        rows = []
        for start in range(0, len(contexts), 1024):
            tensor = torch.from_numpy(contexts[start : start + 1024]).to(
                checkpoint.device
            )
            with torch.inference_mode():
                outputs = checkpoint.predict_sequence(
                    tensor,
                    onset_threshold=onset_state_threshold,
                    offset_threshold=offset_state_threshold,
                )
            activity = torch.sigmoid(outputs["activity_logits"]).cpu().numpy()
            onset = torch.sigmoid(outputs["onset_logits"]).cpu().numpy()
            offset = torch.sigmoid(outputs["offset_logits"]).cpu().numpy()
            onset_delta = outputs["onset_delta"].cpu().numpy()
            offset_delta = outputs["offset_delta"].cpu().numpy()
            for item_idx in range(activity.shape[0]):
                frame_idx = frame_indices[start + item_idx]
                best_slot = int(np.argmax(activity[item_idx]))
                rows.append(
                    {
                        "frame_idx": int(frame_idx),
                        "time_s": frame_idx * SourceTrackingAudioConfigV2.hop / sample_rate,
                        "source_count": int(np.sum(activity[item_idx] >= 0.35)),
                        "best_slot": best_slot,
                        "onset_score": float(onset[item_idx, best_slot]),
                        "offset_score": float(offset[item_idx, best_slot]),
                        "onset_delta_frames": float(onset_delta[item_idx, best_slot]),
                        "offset_delta_frames": float(offset_delta[item_idx, best_slot]),
                    }
                )
        return rows

    def _event_metrics(
        self,
        events: list[SourceEventLabelV2],
        predictions: list[dict[str, object]],
        family: str,
        kind: str,
    ) -> dict[str, object]:
        threshold = self._threshold(family, kind)
        tolerance = (
            SourceTrackingAudioConfigV2.onset_label_radius_frames
            if kind == "onset"
            else SourceTrackingAudioConfigV2.offset_label_radius_frames
        )
        errors = []
        matched = 0
        for event in events:
            boundary = self._boundary_frame(event, kind)
            candidates = [
                row
                for row in predictions
                if abs(int(row["frame_idx"]) - boundary) <= tolerance
                and float(row[f"{kind}_score"]) >= threshold
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: float(row[f"{kind}_score"]))
            predicted_boundary = (
                float(best["frame_idx"]) - float(best[f"{kind}_delta_frames"])
            )
            errors.append(predicted_boundary - boundary)
            matched += 1
        hop_ms = (
            1000.0
            * SourceTrackingAudioConfigV2.hop
            / SourceTrackingAudioConfigV2.sample_rate
        )
        if not errors:
            return {
                "threshold": threshold,
                "events": len(events),
                "matched_events": 0,
                "coverage": 0.0 if events else None,
                "mae_frames": None,
                "rmsd_frames": None,
                "mae_ms": None,
                "rmsd_ms": None,
            }
        values = np.asarray(errors, dtype=np.float32)
        return {
            "threshold": threshold,
            "events": len(events),
            "matched_events": matched,
            "coverage": matched / len(events) if events else None,
            "mae_frames": float(np.mean(np.abs(values))),
            "rmsd_frames": float(np.sqrt(np.mean(values * values))),
            "mae_ms": float(np.mean(np.abs(values)) * hop_ms),
            "rmsd_ms": float(np.sqrt(np.mean(values * values)) * hop_ms),
        }

    def _event_detection_metrics(
        self,
        events: list[SourceEventLabelV2],
        predictions: list[dict[str, object]],
        family: str,
        kind: str,
    ) -> dict[str, object]:
        threshold = self._threshold(family, kind)
        tolerance = (
            SourceTrackingAudioConfigV2.onset_label_radius_frames
            if kind == "onset"
            else SourceTrackingAudioConfigV2.offset_label_radius_frames
        )
        hop_ms = (
            1000.0
            * SourceTrackingAudioConfigV2.hop
            / SourceTrackingAudioConfigV2.sample_rate
        )
        refractory = max(1, int(round(self.event_refractory_ms / hop_ms)))
        peaks = self._prediction_peaks(predictions, kind, threshold, refractory)
        unmatched = set(range(len(events)))
        matched_errors = []
        false_positives = 0
        for peak in peaks:
            predicted_boundary = float(peak["frame_idx"]) - float(
                peak[f"{kind}_delta_frames"]
            )
            if not unmatched:
                false_positives += 1
                continue
            event_idx = min(
                unmatched,
                key=lambda idx: abs(
                    self._boundary_frame(events[idx], kind) - predicted_boundary
                ),
            )
            error = predicted_boundary - self._boundary_frame(events[event_idx], kind)
            if abs(error) <= tolerance:
                matched_errors.append(error)
                unmatched.remove(event_idx)
            else:
                false_positives += 1
        true_positive = len(matched_errors)
        false_negative = len(unmatched)
        precision = (
            true_positive / (true_positive + false_positives)
            if true_positive + false_positives
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision and recall
            else 0.0
        )
        values = np.asarray(matched_errors, dtype=np.float32)
        duration_min = max(1e-6, predictions[-1]["time_s"] / 60.0) if predictions else 1e-6
        return {
            "threshold": threshold,
            "refractory_frames": refractory,
            "refractory_ms": refractory * hop_ms,
            "match_tolerance_frames": tolerance,
            "predicted_events": len(peaks),
            "true_events": len(events),
            "true_positives": true_positive,
            "false_positives": false_positives,
            "false_negatives": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate_per_minute": false_positives / duration_min,
            "mae_ms": float(np.mean(np.abs(values)) * hop_ms) if len(values) else None,
        }

    @staticmethod
    def _prediction_peaks(
        predictions: list[dict[str, object]],
        kind: str,
        threshold: float,
        refractory: int,
    ) -> list[dict[str, object]]:
        candidates = BoundaryTimingSequenceEvaluator._local_peak_candidates(
            predictions,
            kind,
            threshold,
        )
        ordered = sorted(
            candidates,
            key=lambda row: float(row[f"{kind}_score"]),
            reverse=True,
        )
        selected: list[dict[str, object]] = []
        for row in ordered:
            frame_idx = int(row["frame_idx"])
            if any(
                abs(frame_idx - int(existing["frame_idx"])) <= refractory
                for existing in selected
            ):
                continue
            selected.append(row)
        return sorted(selected, key=lambda row: int(row["frame_idx"]))

    @staticmethod
    def _local_peak_candidates(
        predictions: list[dict[str, object]],
        kind: str,
        threshold: float,
    ) -> list[dict[str, object]]:
        candidates = []
        previous_score = float("-inf")
        for idx, row in enumerate(predictions):
            score = float(row[f"{kind}_score"])
            next_score = (
                float(predictions[idx + 1][f"{kind}_score"])
                if idx + 1 < len(predictions)
                else float("-inf")
            )
            if score < threshold:
                previous_score = score
                continue
            if score >= previous_score and score > next_score:
                candidates.append(row)
            previous_score = score
        return candidates

    @staticmethod
    def _boundary_frame(event: SourceEventLabelV2, kind: str) -> int:
        hop = SourceTrackingAudioConfigV2.hop
        sample_rate = SourceTrackingAudioConfigV2.sample_rate
        if kind == "onset":
            return max(0, int(np.floor(event.start_s * sample_rate / hop)))
        end_frame = int(np.ceil(event.end_s * sample_rate / hop))
        return max(0, end_frame - 1)

    def _threshold(self, family: str, kind: str) -> float:
        if kind == "onset" and self.onset_threshold is not None:
            return self.onset_threshold
        if kind == "offset" and self.offset_threshold is not None:
            return self.offset_threshold
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
        return 0.35

    def _read_wav(self) -> tuple[int, np.ndarray]:
        with wave.open(str(self.wav), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if channels != 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        return sample_rate, (pcm.astype(np.float32) / 32767.0)

    @staticmethod
    def _load_calibration(path: Path | None) -> dict[str, object]:
        if path is None:
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


class BoundaryTimingSequenceEvalCli:
    @staticmethod
    def run() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--data-root", type=Path, required=True)
        parser.add_argument("--wav", type=Path, required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--calibration-input", type=Path)
        parser.add_argument("--seconds", type=float, default=60.0)
        parser.add_argument("--test-instrument-index", type=int, default=0)
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--sample-stride-ms", type=float, default=5.0)
        parser.add_argument("--onset-threshold", type=float)
        parser.add_argument("--offset-threshold", type=float)
        parser.add_argument("--event-refractory-ms", type=float, default=200.0)
        args = parser.parse_args()
        BoundaryTimingSequenceEvaluator(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            wav=args.wav,
            output_json=args.output_json,
            calibration_input=args.calibration_input,
            seconds=args.seconds,
            test_instrument_index=args.test_instrument_index,
            device=args.device,
            sample_stride_ms=args.sample_stride_ms,
            onset_threshold=args.onset_threshold,
            offset_threshold=args.offset_threshold,
            event_refractory_ms=args.event_refractory_ms,
        ).run()


if __name__ == "__main__":
    BoundaryTimingSequenceEvalCli.run()
