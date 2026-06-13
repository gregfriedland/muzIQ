"""Lakh MIDI streaming ingestion and compact schedule parsing."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
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

    def accepts(self, schedule: MidiScheduleV2) -> bool:
        if len(schedule.events) == 0:
            return False
        if len(schedule.events) > self.MAX_EVENTS:
            return False
        if schedule.duration_s > self.MAX_DURATION_S:
            return False
        return len(schedule.track_ids) <= self.MAX_TRACKS


class LakhMidiDownloaderV2:
    """Stream Lakh MIDI archives into compact split manifests."""

    ARCHIVE_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
    DEFAULT_TARGETS: dict[SplitName, int] = {
        "train": 60_000,
        "validation": 4_000,
        "test": 4_000,
    }

    def __init__(self, data_root: Path):
        self.paths = MidiPathsV2(data_root)
        self.parser = MidiScheduleParserV2()
        self.invalid_file_policy = MidiInvalidFilePolicyV2()
        self.schedule_bounds = MidiScheduleBoundsV2()

    def build_cache(
        self, source: str | None = None, targets: dict[SplitName, int] | None = None
    ) -> None:
        target_counts = targets or self.DEFAULT_TARGETS
        buckets: dict[SplitName, list[MidiScheduleV2]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        with self._open_stream(source or self.ARCHIVE_URL) as stream:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    if self._complete(buckets, target_counts):
                        break
                    if not self._is_midi_member(member):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    payload = extracted.read()
                    try:
                        schedule = self.parser.parse_bytes(
                            payload,
                            member.name,
                            max_events=self.schedule_bounds.MAX_EVENTS,
                        )
                    except Exception as error:
                        if self.invalid_file_policy.should_skip(error):
                            continue
                        raise
                    if not self.schedule_bounds.accepts(schedule):
                        continue
                    if len(buckets[schedule.split]) >= target_counts[schedule.split]:
                        continue
                    buckets[schedule.split].append(schedule)
        for split, schedules in buckets.items():
            ManifestIOV2.write_midi(self.paths.split_manifest(split), schedules)
        self.audit_leakage()

    def load_manifests(self) -> dict[SplitName, list[MidiScheduleV2]]:
        return {
            split: ManifestIOV2.read_midi(self.paths.split_manifest(split))
            for split in ("train", "validation", "test")
        }

    def audit_leakage(self) -> None:
        MidiSplitAuditV2().audit(self.load_manifests())

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
        self.schedules_by_split = {
            split: ManifestIOV2.read_midi(self.paths.split_manifest(split))
            for split in ("train", "validation", "test")
        }

    def schedules(self, split: SplitName) -> list[MidiScheduleV2]:
        return self.schedules_by_split[split]

    def write_from_records(self, split: SplitName, schedules: list[MidiScheduleV2]) -> None:
        ManifestIOV2.write_midi(self.paths.split_manifest(split), schedules)
        self.schedules_by_split[split] = schedules

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
