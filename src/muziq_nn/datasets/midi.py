"""Lakh MIDI streaming ingestion and compact schedule parsing."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
import urllib.request
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mido

from muziq_nn.datasets.schema import (
    ManifestIOV2,
    MidiNoteEventV2,
    MidiScheduleV2,
    SplitName,
)


class MidiPathsV2:
    """Canonical data paths for compact MIDI schedules."""

    def __init__(self, data_root: Path):
        self.data_root = data_root

    @property
    def manifest_root(self) -> Path:
        return self.data_root / "manifests" / "midi"

    def split_manifest(self, split: SplitName) -> Path:
        return self.manifest_root / f"{split}.jsonl"


class MidiSplitAssignerV2:
    """Assign MIDI files to deterministic hash splits."""

    def split_for_id(self, schedule_id: str) -> SplitName:
        value = int(schedule_id[:8], 16) % 100
        if value < 90:
            return "train"
        if value < 95:
            return "validation"
        return "test"


class MidiInvalidFileV2(ValueError):
    """Raised when Mido cannot load an external MIDI file."""


class MidiScheduleTooLargeV2(ValueError):
    """Raised when a MIDI schedule exceeds cache bounds during parsing."""


class MidiScheduleParserV2:
    """Parse a MIDI byte stream into note events in seconds."""

    DEFAULT_TEMPO = 500_000

    def parse_bytes(
        self, payload: bytes, source_path: str, max_events: int | None = None
    ) -> MidiScheduleV2:
        schedule_id = hashlib.md5(payload).hexdigest()
        split = MidiSplitAssignerV2().split_for_id(schedule_id)
        try:
            midi = mido.MidiFile(file=io.BytesIO(payload))
        except Exception as error:
            raise MidiInvalidFileV2(f"invalid MIDI file {source_path}: {error}") from error
        events: list[MidiNoteEventV2] = []
        duration_s = 0.0
        for track_idx, track in enumerate(midi.tracks):
            events.extend(
                self._parse_track(
                    midi,
                    track,
                    track_idx,
                    self._remaining_events(max_events, len(events), source_path),
                )
            )
        if events:
            duration_s = max(event.end_s for event in events)
        return MidiScheduleV2(
            schedule_id=schedule_id,
            split=split,
            duration_s=float(duration_s),
            source_path=source_path,
            events=tuple(sorted(events, key=lambda event: (event.start_s, event.track))),
        )

    def _parse_track(
        self,
        midi: mido.MidiFile,
        track: mido.MidiTrack,
        track_idx: int,
        max_events: int | None = None,
    ) -> list[MidiNoteEventV2]:
        tempo = self.DEFAULT_TEMPO
        seconds = 0.0
        active: dict[tuple[int | None, int], tuple[float, int, int | None]] = {}
        programs: dict[int | None, int | None] = {}
        events: list[MidiNoteEventV2] = []
        for message in track:
            seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
            if message.type == "set_tempo":
                tempo = int(message.tempo)
                continue
            if message.type == "program_change":
                programs[getattr(message, "channel", None)] = int(message.program)
                continue
            if message.type == "note_on" and message.velocity > 0:
                channel = getattr(message, "channel", None)
                active[(channel, int(message.note))] = (
                    seconds,
                    int(message.velocity),
                    programs.get(channel),
                )
                continue
            if message.type not in ("note_off", "note_on"):
                continue
            channel = getattr(message, "channel", None)
            key = (channel, int(message.note))
            started = active.pop(key, None)
            if started is None:
                continue
            start_s, velocity, program = started
            if seconds <= start_s:
                continue
            events.append(
                MidiNoteEventV2(
                    start_s=float(start_s),
                    end_s=float(seconds),
                    pitch=int(message.note),
                    velocity=velocity,
                    track=track_idx,
                    channel=channel,
                    program=program,
                )
            )
            if max_events is not None and len(events) > max_events:
                raise MidiScheduleTooLargeV2(
                    f"MIDI track {track_idx} exceeds {max_events} parsed events"
                )
        return events

    @staticmethod
    def _remaining_events(
        max_events: int | None, parsed_events: int, source_path: str
    ) -> int | None:
        if max_events is None:
            return None
        remaining = max_events - parsed_events
        if remaining < 0:
            raise MidiScheduleTooLargeV2(
                f"MIDI file {source_path} exceeds {max_events} parsed events"
            )
        return remaining


class MidiSplitAuditV2:
    """Detect MIDI hash leakage across split manifests."""

    def audit(self, split_schedules: dict[SplitName, list[MidiScheduleV2]]) -> None:
        owners: dict[str, set[str]] = {}
        for split, schedules in split_schedules.items():
            for schedule in schedules:
                owners.setdefault(schedule.schedule_id, set()).add(split)
        leaks = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
        if leaks:
            sample = dict(list(sorted(leaks.items()))[:5])
            raise ValueError(f"MIDI hash leakage detected: {sample}")


class MidiInvalidFilePolicyV2:
    """Classify external MIDI-file parse failures that should be skipped."""

    MIDO_MODULE_PREFIX = "mido."
    GENERIC_PARSE_ERRORS = (EOFError, OSError, ValueError, KeyError)

    def should_skip(self, error: Exception) -> bool:
        if isinstance(error, self.GENERIC_PARSE_ERRORS):
            return True
        return error.__class__.__module__.startswith(self.MIDO_MODULE_PREFIX)


class MidiScheduleBoundsV2:
    """Reject schedules too large for bounded 30-second mixture rendering."""

    MAX_EVENTS = 2_048
    MAX_DURATION_S = 600.0
    MAX_TRACKS = 12
    MAX_PAYLOAD_BYTES = 1_000_000

    def accepts(self, schedule: MidiScheduleV2) -> bool:
        return self.reject_reason(schedule) is None

    def reject_reason(self, schedule: MidiScheduleV2) -> str | None:
        if len(schedule.events) == 0:
            return "empty"
        if len(schedule.events) > self.MAX_EVENTS:
            return "too_many_events"
        if schedule.duration_s > self.MAX_DURATION_S:
            return "too_long"
        if len(schedule.track_ids) > self.MAX_TRACKS:
            return "too_many_tracks"
        return None


class MidiCacheBuildStatsV2:
    """Counters emitted while streaming and parsing Lakh MIDI."""

    def __init__(self):
        self.archive_members = 0
        self.midi_members = 0
        self.payload_bytes = 0
        self.batches = 0
        self.parsed = 0
        self.accepted = 0
        self.invalid = 0
        self.too_large = 0
        self.rejected = 0
        self.split_full = 0
        self.payload_too_large = 0

    def as_dict(
        self,
        buckets: dict[SplitName, list[MidiScheduleV2]],
        targets: dict[SplitName, int],
    ) -> dict[str, object]:
        return {
            "archive_members": self.archive_members,
            "midi_members": self.midi_members,
            "payload_mb": round(self.payload_bytes / 1_000_000.0, 3),
            "batches": self.batches,
            "parsed": self.parsed,
            "accepted": self.accepted,
            "invalid": self.invalid,
            "too_large": self.too_large,
            "rejected": self.rejected,
            "split_full": self.split_full,
            "payload_too_large": self.payload_too_large,
            "counts": {split: len(buckets[split]) for split in ("train", "validation", "test")},
            "targets": targets,
        }


class MidiCacheProgressLoggerV2:
    """Emit periodic JSON progress lines for long MIDI archive scans."""

    def __init__(self, interval_s: float = 10.0):
        self.interval_s = interval_s
        self.started = time.monotonic()
        self.last_emit = 0.0

    def emit(
        self,
        event: str,
        stats: MidiCacheBuildStatsV2,
        buckets: dict[SplitName, list[MidiScheduleV2]],
        targets: dict[SplitName, int],
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_emit < self.interval_s:
            return
        self.last_emit = now
        payload = {
            "event": event,
            "elapsed_s": round(now - self.started, 3),
            **stats.as_dict(buckets, targets),
        }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


class LakhMidiDownloaderV2:
    """Stream Lakh MIDI archives into compact split manifests."""

    ARCHIVE_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
    DEFAULT_TARGETS: dict[SplitName, int] = {
        "train": 60_000,
        "validation": 4_000,
        "test": 4_000,
    }

    def __init__(
        self,
        data_root: Path,
        workers: int = 1,
        parse_batch_size: int = 64,
        progress_interval_s: float = 10.0,
    ):
        self.paths = MidiPathsV2(data_root)
        self.parser = MidiScheduleParserV2()
        self.invalid_file_policy = MidiInvalidFilePolicyV2()
        self.schedule_bounds = MidiScheduleBoundsV2()
        self.workers = max(1, workers)
        self.parse_batch_size = max(1, parse_batch_size)
        self.progress_interval_s = progress_interval_s

    def build_cache(
        self, source: str | None = None, targets: dict[SplitName, int] | None = None
    ) -> None:
        target_counts = targets or self.DEFAULT_TARGETS
        buckets: dict[SplitName, list[MidiScheduleV2]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        stats = MidiCacheBuildStatsV2()
        logger = MidiCacheProgressLoggerV2(self.progress_interval_s)
        batch: list[tuple[str, bytes, int]] = []
        executor = (
            ProcessPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        )
        logger.emit("midi_cache_start", stats, buckets, target_counts, force=True)
        with self._open_stream(source or self.ARCHIVE_URL) as stream:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                try:
                    for member in archive:
                        stats.archive_members += 1
                        if self._complete(buckets, target_counts):
                            break
                        if not self._is_midi_member(member):
                            logger.emit(
                                "midi_cache_progress", stats, buckets, target_counts
                            )
                            continue
                        stats.midi_members += 1
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            logger.emit(
                                "midi_cache_progress", stats, buckets, target_counts
                            )
                            continue
                        payload = extracted.read()
                        stats.payload_bytes += len(payload)
                        if len(payload) > self.schedule_bounds.MAX_PAYLOAD_BYTES:
                            stats.payload_too_large += 1
                            logger.emit(
                                "midi_cache_progress", stats, buckets, target_counts
                            )
                            continue
                        batch.append((member.name, payload, self.schedule_bounds.MAX_EVENTS))
                        if len(batch) >= self.parse_batch_size:
                            self._consume_batch(batch, buckets, target_counts, stats, executor)
                            batch.clear()
                        logger.emit("midi_cache_progress", stats, buckets, target_counts)
                    if batch:
                        self._consume_batch(batch, buckets, target_counts, stats, executor)
                        batch.clear()
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True)
        for split, schedules in buckets.items():
            ManifestIOV2.write_midi(self.paths.split_manifest(split), schedules)
        logger.emit("midi_cache_manifests_written", stats, buckets, target_counts, force=True)
        self.audit_leakage()
        logger.emit("midi_cache_done", stats, buckets, target_counts, force=True)

    def _consume_batch(
        self,
        batch: list[tuple[str, bytes, int]],
        buckets: dict[SplitName, list[MidiScheduleV2]],
        targets: dict[SplitName, int],
        stats: MidiCacheBuildStatsV2,
        executor: ProcessPoolExecutor | None,
    ) -> None:
        stats.batches += 1
        if executor is None:
            for task in batch:
                self._consume_parse_result(
                    self._parse_payload(task), buckets, targets, stats
                )
            return
        futures = [
            executor.submit(LakhMidiDownloaderV2._parse_worker, task) for task in batch
        ]
        for future in as_completed(futures):
            self._consume_parse_result(future.result(), buckets, targets, stats)

    def _parse_payload(
        self, task: tuple[str, bytes, int]
    ) -> tuple[str, MidiScheduleV2 | None, str | None]:
        path, payload, max_events = task
        try:
            schedule = self.parser.parse_bytes(payload, path, max_events=max_events)
        except Exception as error:
            if self.invalid_file_policy.should_skip(error):
                reason = "too_large" if isinstance(error, MidiScheduleTooLargeV2) else "invalid"
                return reason, None, error.__class__.__name__
            raise
        reject_reason = self.schedule_bounds.reject_reason(schedule)
        if reject_reason is not None:
            return "rejected", None, reject_reason
        return "parsed", schedule, None

    @staticmethod
    def _parse_worker(
        task: tuple[str, bytes, int]
    ) -> tuple[str, MidiScheduleV2 | None, str | None]:
        downloader = LakhMidiDownloaderV2(Path("."), workers=1)
        return downloader._parse_payload(task)

    def _consume_parse_result(
        self,
        result: tuple[str, MidiScheduleV2 | None, str | None],
        buckets: dict[SplitName, list[MidiScheduleV2]],
        targets: dict[SplitName, int],
        stats: MidiCacheBuildStatsV2,
    ) -> None:
        status, schedule, _reason = result
        if status == "invalid":
            stats.invalid += 1
            return
        if status == "too_large":
            stats.too_large += 1
            return
        if status == "rejected":
            stats.rejected += 1
            return
        if schedule is None:
            stats.invalid += 1
            return
        stats.parsed += 1
        if len(buckets[schedule.split]) >= targets[schedule.split]:
            stats.split_full += 1
            return
        buckets[schedule.split].append(schedule)
        stats.accepted += 1

    def load_manifests(self) -> dict[SplitName, list[MidiScheduleV2]]:
        return {
            split: ManifestIOV2.read_midi(self.paths.split_manifest(split))
            for split in ("train", "validation", "test")
        }

    def manifest_counts(self) -> dict[SplitName, int]:
        return {
            split: self._count_manifest_records(self.paths.split_manifest(split))
            for split in ("train", "validation", "test")
        }

    def audit_leakage(self) -> None:
        owners: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            path = self.paths.split_manifest(split)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    schedule_id = json.loads(line)["schedule_id"]
                    owners.setdefault(schedule_id, set()).add(split)
        leaks = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
        if leaks:
            sample = dict(list(sorted(leaks.items()))[:5])
            raise ValueError(f"MIDI hash leakage detected: {sample}")

    @staticmethod
    def _count_manifest_records(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("rb") as fh:
            return sum(1 for line in fh if line.strip())

    @staticmethod
    def _open_stream(source: str):
        if source.startswith("http://") or source.startswith("https://"):
            return urllib.request.urlopen(source, timeout=60)
        return Path(source).open("rb")

    @staticmethod
    def _is_midi_member(member: tarfile.TarInfo) -> bool:
        suffix = Path(member.name).suffix.lower()
        return member.isfile() and suffix in (".mid", ".midi")

    @staticmethod
    def _complete(
        buckets: dict[SplitName, list[MidiScheduleV2]],
        targets: dict[SplitName, int],
    ) -> bool:
        return all(len(buckets[split]) >= targets[split] for split in targets)


class MidiIndexV2:
    """Read compact MIDI schedule manifests."""

    def __init__(self, data_root: Path):
        self.paths = MidiPathsV2(data_root)
        self.schedules_by_split: dict[SplitName, MidiScheduleSequenceV2] = {
            split: MidiScheduleSequenceV2(self.paths.split_manifest(split))
            for split in ("train", "validation", "test")
        }

    def schedules(self, split: SplitName) -> Sequence[MidiScheduleV2]:
        return self.schedules_by_split[split]

    def write_from_records(self, split: SplitName, schedules: list[MidiScheduleV2]) -> None:
        ManifestIOV2.write_midi(self.paths.split_manifest(split), schedules)
        self.schedules_by_split[split] = MidiScheduleSequenceV2(
            self.paths.split_manifest(split)
        )

    def write_metadata(self) -> None:
        self.paths.manifest_root.mkdir(parents=True, exist_ok=True)
        payload = {
            split: {
                "count": len(schedules),
                "note_events": sum(len(schedule.events) for schedule in schedules),
            }
            for split, schedules in self.schedules_by_split.items()
        }
        (self.paths.manifest_root / "summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class MidiScheduleSequenceV2(Sequence[MidiScheduleV2]):
    """Random-access MIDI manifest view without materializing all schedules."""

    CACHE_SIZE = 128

    def __init__(self, path: Path):
        self.path = path
        self.offsets = self._build_offsets(path)
        self.cache: OrderedDict[int, MidiScheduleV2] = OrderedDict()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int | slice) -> MidiScheduleV2 | list[MidiScheduleV2]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        cached = self.cache.get(index)
        if cached is not None:
            self.cache.move_to_end(index)
            return cached
        schedule = self._read_schedule(index)
        self.cache[index] = schedule
        if len(self.cache) > self.CACHE_SIZE:
            self.cache.popitem(last=False)
        return schedule

    @staticmethod
    def _build_offsets(path: Path) -> tuple[int, ...]:
        if not path.exists():
            return ()
        offsets: list[int] = []
        with path.open("rb") as fh:
            offset = fh.tell()
            line = fh.readline()
            while line:
                if line.strip():
                    offsets.append(offset)
                offset = fh.tell()
                line = fh.readline()
        return tuple(offsets)

    def _read_schedule(self, index: int) -> MidiScheduleV2:
        with self.path.open("rb") as fh:
            fh.seek(self.offsets[index])
            line = fh.readline()
        return MidiScheduleV2.model_validate_json(line)
