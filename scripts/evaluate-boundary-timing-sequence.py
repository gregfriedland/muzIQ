from __future__ import annotations

import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    FamilyVocabularyV2,
    NsynthNoteStoreV2,
    SourceTrackingAudioConfigV2,
    SourceTrackingRendererV2,
)
from muziq_nn.datasets.schema import SourceEventLabelV2
from muziq_nn.webapp.inference import RealtimeSourceTrackerV2


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
    ):
        self.checkpoint = checkpoint
        self.data_root = data_root
        self.wav = wav
        self.output_json = output_json
        self.calibration = self._load_calibration(calibration_input)
        self.seconds = seconds
        self.test_instrument_index = test_instrument_index
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
            "calibration": self.calibration,
            "onset": self._event_metrics(events, predictions, family, "onset"),
            "offset": self._event_metrics(events, predictions, family, "offset"),
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
        tracker = RealtimeSourceTrackerV2(self.checkpoint, device="cpu")
        chunk = int(sample_rate * 0.1)
        rows = []
        for start in range(0, len(audio), chunk):
            samples = audio[start : start + chunk]
            prediction = tracker.append_audio(samples, sample_rate=sample_rate)
            sources = list(prediction.sources)
            best = max(sources, key=lambda source: source.activity) if sources else None
            rows.append(
                {
                    "frame_idx": int(
                        round((start + len(samples)) / SourceTrackingAudioConfigV2.hop)
                    ),
                    "time_s": (start + len(samples)) / sample_rate,
                    "source_count": prediction.source_count,
                    "best_slot": best.slot if best is not None else None,
                    "onset_score": best.onset if best is not None else 0.0,
                    "offset_score": best.offset if best is not None else 0.0,
                    "onset_delta_frames": best.onset_delta if best is not None else 0.0,
                    "offset_delta_frames": best.offset_delta if best is not None else 0.0,
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

    @staticmethod
    def _boundary_frame(event: SourceEventLabelV2, kind: str) -> int:
        hop = SourceTrackingAudioConfigV2.hop
        sample_rate = SourceTrackingAudioConfigV2.sample_rate
        if kind == "onset":
            return max(0, int(np.floor(event.start_s * sample_rate / hop)))
        end_frame = int(np.ceil(event.end_s * sample_rate / hop))
        return max(0, end_frame - 1)

    def _threshold(self, family: str, kind: str) -> float:
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
        args = parser.parse_args()
        BoundaryTimingSequenceEvaluator(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            wav=args.wav,
            output_json=args.output_json,
            calibration_input=args.calibration_input,
            seconds=args.seconds,
            test_instrument_index=args.test_instrument_index,
        ).run()


if __name__ == "__main__":
    BoundaryTimingSequenceEvalCli.run()
