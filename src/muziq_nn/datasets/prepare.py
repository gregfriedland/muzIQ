"""Data preparation CLI for bounded NSynth and MIDI caches."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from muziq_nn.datasets.midi import LakhMidiDownloaderV2, MidiScheduleBoundsV2
from muziq_nn.datasets.nsynth import NsynthDownloaderV2, NsynthSplitAuditV2


@dataclass(frozen=True)
class PrepareConfigV2:
    data_root: str = "data"
    storage_budget_gb: float = 10.0
    download_metadata: bool = False
    build_nsynth_cache: bool = False
    build_midi_cache: bool = False
    audit_leakage: bool = False
    nsynth_train_target: int = 25_000
    nsynth_validation_target: int = 2_000
    nsynth_test_target: int = 2_000
    midi_train_target: int = 60_000
    midi_validation_target: int = 4_000
    midi_test_target: int = 4_000
    midi_workers: int = max(1, min(8, os.cpu_count() or 1))
    midi_parse_batch_size: int = 64
    progress_interval_s: float = 10.0
    nsynth_train_archive: str | None = None
    nsynth_validation_archive: str | None = None
    nsynth_test_archive: str | None = None
    midi_archive: str | None = None


class PrepareRunnerV2:
    """Run bounded-cache preparation actions."""

    def __init__(self, config: PrepareConfigV2):
        self.config = config
        self.data_root = Path(config.data_root)

    def run(self) -> dict[str, object]:
        self.data_root.mkdir(parents=True, exist_ok=True)
        actions = []
        if self.config.download_metadata:
            actions.append(self._write_metadata())
        if self.config.build_nsynth_cache:
            actions.append(self._build_nsynth_cache())
        if self.config.build_midi_cache:
            actions.append(self._build_midi_cache())
        if self.config.audit_leakage:
            actions.append(self._audit_leakage())
        payload = {"data_root": str(self.data_root), "actions": actions}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def _write_metadata(self) -> dict[str, object]:
        manifest_root = self.data_root / "manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "storage_budget_gb": self.config.storage_budget_gb,
            "nsynth_urls": NsynthDownloaderV2.ARCHIVE_URLS,
            "lakh_midi_url": LakhMidiDownloaderV2.ARCHIVE_URL,
            "midi_schedule_bounds": {
                "max_events": MidiScheduleBoundsV2.MAX_EVENTS,
                "max_duration_s": MidiScheduleBoundsV2.MAX_DURATION_S,
                "max_tracks": MidiScheduleBoundsV2.MAX_TRACKS,
                "max_payload_bytes": MidiScheduleBoundsV2.MAX_PAYLOAD_BYTES,
            },
            "midi_cache_runtime": {
                "workers": self.config.midi_workers,
                "parse_batch_size": self.config.midi_parse_batch_size,
                "progress_interval_s": self.config.progress_interval_s,
            },
            "note": (
                "Metadata records download sources; cache builders stream "
                "archives on demand."
            ),
        }
        (manifest_root / "sources.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return {"download_metadata": str(manifest_root / "sources.json")}

    def _build_nsynth_cache(self) -> dict[str, object]:
        downloader = NsynthDownloaderV2(self.data_root, self.config.storage_budget_gb)
        sources = {
            "train": self.config.nsynth_train_archive or downloader.ARCHIVE_URLS["train"],
            "validation": self.config.nsynth_validation_archive
            or downloader.ARCHIVE_URLS["validation"],
            "test": self.config.nsynth_test_archive or downloader.ARCHIVE_URLS["test"],
        }
        targets = {
            "train": self.config.nsynth_train_target,
            "validation": self.config.nsynth_validation_target,
            "test": self.config.nsynth_test_target,
        }
        counts = {}
        for split, target in targets.items():
            counts[split] = len(downloader.stream_split(split, sources[split], target))
        downloader.audit_leakage()
        return {"build_nsynth_cache": counts}

    def _build_midi_cache(self) -> dict[str, object]:
        downloader = LakhMidiDownloaderV2(
            self.data_root,
            workers=self.config.midi_workers,
            parse_batch_size=self.config.midi_parse_batch_size,
            progress_interval_s=self.config.progress_interval_s,
        )
        targets = {
            "train": self.config.midi_train_target,
            "validation": self.config.midi_validation_target,
            "test": self.config.midi_test_target,
        }
        downloader.build_cache(self.config.midi_archive, targets)
        counts = {split: len(items) for split, items in downloader.load_manifests().items()}
        return {
            "build_midi_cache": counts,
            "workers": self.config.midi_workers,
            "parse_batch_size": self.config.midi_parse_batch_size,
        }

    def _audit_leakage(self) -> dict[str, object]:
        nsynth = NsynthDownloaderV2(self.data_root, self.config.storage_budget_gb)
        midi = LakhMidiDownloaderV2(self.data_root)
        NsynthSplitAuditV2().audit(nsynth.load_manifests())
        midi.audit_leakage()
        return {"audit_leakage": "ok"}


class PrepareCliV2:
    """Command-line interface for cache preparation."""

    @staticmethod
    def parse(argv: Sequence[str] | None = None) -> PrepareConfigV2:
        parser = argparse.ArgumentParser(description="Prepare bounded muziq-nn caches.")
        parser.add_argument("--data-root", default=PrepareConfigV2.data_root)
        parser.add_argument(
            "--storage-budget-gb", type=float, default=PrepareConfigV2.storage_budget_gb
        )
        parser.add_argument("--download-metadata", action="store_true")
        parser.add_argument("--build-nsynth-cache", action="store_true")
        parser.add_argument("--build-midi-cache", action="store_true")
        parser.add_argument("--audit-leakage", action="store_true")
        parser.add_argument(
            "--nsynth-train-target", type=int, default=PrepareConfigV2.nsynth_train_target
        )
        parser.add_argument(
            "--nsynth-validation-target",
            type=int,
            default=PrepareConfigV2.nsynth_validation_target,
        )
        parser.add_argument(
            "--nsynth-test-target", type=int, default=PrepareConfigV2.nsynth_test_target
        )
        parser.add_argument(
            "--midi-train-target", type=int, default=PrepareConfigV2.midi_train_target
        )
        parser.add_argument(
            "--midi-validation-target",
            type=int,
            default=PrepareConfigV2.midi_validation_target,
        )
        parser.add_argument(
            "--midi-test-target", type=int, default=PrepareConfigV2.midi_test_target
        )
        parser.add_argument("--midi-workers", type=int, default=PrepareConfigV2.midi_workers)
        parser.add_argument(
            "--midi-parse-batch-size",
            type=int,
            default=PrepareConfigV2.midi_parse_batch_size,
        )
        parser.add_argument(
            "--progress-interval-s",
            type=float,
            default=PrepareConfigV2.progress_interval_s,
        )
        parser.add_argument("--nsynth-train-archive")
        parser.add_argument("--nsynth-validation-archive")
        parser.add_argument("--nsynth-test-archive")
        parser.add_argument("--midi-archive")
        return PrepareConfigV2(**vars(parser.parse_args(argv)))

    @staticmethod
    def main(argv: Sequence[str] | None = None) -> int:
        PrepareRunnerV2(PrepareCliV2.parse(argv)).run()
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    return PrepareCliV2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
