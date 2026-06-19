"""NSynth bounded-cache ingestion."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from muziq_nn.datasets.schema import ManifestIOV2, NsynthNoteV2, SplitName


class NsynthPathsV2:
    """Canonical data paths for the bounded NSynth cache."""

    def __init__(self, data_root: Path):
        self.data_root = data_root

    @property
    def cache_root(self) -> Path:
        return self.data_root / "nsynth_cache"

    @property
    def manifest_root(self) -> Path:
        return self.data_root / "manifests" / "nsynth"

    def split_manifest(self, split: SplitName) -> Path:
        return self.manifest_root / f"{split}.jsonl"

    def split_audio_root(self, split: SplitName) -> Path:
        return self.cache_root / split / "audio"


class NsynthNoteNameV2:
    """Parse fields encoded in NSynth note filenames."""

    @staticmethod
    def from_note_str(note_str: str, split: SplitName, wav_path: Path) -> NsynthNoteV2:
        family, source, instrument_token, pitch_token, velocity_token = NsynthNoteNameV2._parts(
            note_str
        )
        instrument_str = f"{family}_{source}_{instrument_token}"
        return NsynthNoteV2(
            note_str=note_str,
            split=split,
            wav_path=str(wav_path),
            instrument=int(instrument_token),
            instrument_str=instrument_str,
            family=family,
            source=source,
            pitch=int(pitch_token),
            velocity=int(velocity_token),
            qualities=(),
        )

    @staticmethod
    def _parts(note_str: str) -> tuple[str, str, str, str, str]:
        stem = Path(note_str).stem
        left, pitch, velocity = stem.rsplit("-", 2)
        family, source, instrument = left.rsplit("_", 2)
        return family, source, instrument, pitch, velocity


@dataclass(frozen=True)
class NsynthOfficialMetadataRecordV2:
    """One official NSynth metadata row before local split assignment."""

    note_str: str
    source_split: SplitName
    instrument: int
    instrument_str: str
    family: str
    source: str
    pitch: int
    velocity: int
    qualities: tuple[str, ...]

    @staticmethod
    def from_examples_json(
        note_str: str, payload: dict[str, Any], source_split: SplitName
    ) -> NsynthOfficialMetadataRecordV2:
        fallback = NsynthNoteNameV2.from_note_str(note_str, source_split, Path(""))
        instrument_str = str(payload.get("instrument_str") or fallback.instrument_str)
        family = str(payload.get("instrument_family_str") or fallback.family)
        source = str(payload.get("instrument_source_str") or fallback.source)
        return NsynthOfficialMetadataRecordV2(
            note_str=note_str,
            source_split=source_split,
            instrument=int(payload.get("instrument", fallback.instrument)),
            instrument_str=instrument_str,
            family=family,
            source=source,
            pitch=int(payload.get("pitch", fallback.pitch)),
            velocity=int(payload.get("velocity", fallback.velocity)),
            qualities=tuple(str(item) for item in payload.get("qualities_str", ())),
        )

    def to_note(self, split: SplitName, wav_path: Path) -> NsynthNoteV2:
        return NsynthNoteV2(
            note_str=self.note_str,
            split=split,
            wav_path=str(wav_path),
            instrument=self.instrument,
            instrument_str=self.instrument_str,
            family=self.family,
            source=self.source,
            pitch=self.pitch,
            velocity=self.velocity,
            qualities=self.qualities,
        )


class NsynthMetadataInstrumentSplitV2:
    """Deterministically assign official NSynth instruments to local splits."""

    def __init__(
        self,
        train_instruments: int = 800,
        validation_instruments: int = 100,
        test_instruments: int = 100,
        seed: str = "muziq-nsynth-v2",
    ):
        self.targets: dict[SplitName, int] = {
            "train": train_instruments,
            "validation": validation_instruments,
            "test": test_instruments,
        }
        self.seed = seed

    def assign(
        self, records: Iterable[NsynthOfficialMetadataRecordV2]
    ) -> dict[int, SplitName]:
        instruments = sorted({record.instrument for record in records})
        requested = sum(self.targets.values())
        if len(instruments) < requested:
            raise ValueError(
                "Not enough official NSynth instruments for requested split: "
                f"requested={requested}, available={len(instruments)}"
            )
        ordered = sorted(instruments, key=self._instrument_key)
        assignment: dict[int, SplitName] = {}
        cursor = 0
        for split in ("train", "validation", "test"):
            target = self.targets[split]
            for instrument in ordered[cursor : cursor + target]:
                assignment[instrument] = split
            cursor += target
        return assignment

    def _instrument_key(self, instrument: int) -> str:
        return hashlib.sha256(f"{self.seed}:{instrument}".encode()).hexdigest()


class NsynthMetadataNoteSelectorV2:
    """Choose bounded pitch/velocity-diverse notes for each selected instrument."""

    def __init__(self, notes_per_instrument: int = 24, seed: str = "muziq-notes-v2"):
        self.notes_per_instrument = notes_per_instrument
        self.seed = seed

    def select(
        self, records: Iterable[NsynthOfficialMetadataRecordV2]
    ) -> set[str]:
        by_instrument: dict[int, list[NsynthOfficialMetadataRecordV2]] = defaultdict(list)
        for record in records:
            by_instrument[record.instrument].append(record)
        selected: set[str] = set()
        for records_for_instrument in by_instrument.values():
            for record in self._select_for_instrument(records_for_instrument):
                selected.add(record.note_str)
        return selected

    def _select_for_instrument(
        self, records: list[NsynthOfficialMetadataRecordV2]
    ) -> list[NsynthOfficialMetadataRecordV2]:
        buckets: dict[tuple[int, int], list[NsynthOfficialMetadataRecordV2]] = defaultdict(list)
        for record in records:
            buckets[(record.pitch // 12, record.velocity)].append(record)
        for bucket_records in buckets.values():
            bucket_records.sort(key=self._note_key)
        selected: list[NsynthOfficialMetadataRecordV2] = []
        while len(selected) < self.notes_per_instrument and buckets:
            for bucket in sorted(buckets):
                if len(selected) >= self.notes_per_instrument:
                    break
                bucket_records = buckets[bucket]
                if bucket_records:
                    selected.append(bucket_records.pop(0))
            buckets = {key: value for key, value in buckets.items() if value}
        return selected

    def _note_key(self, record: NsynthOfficialMetadataRecordV2) -> str:
        return hashlib.sha256(f"{self.seed}:{record.note_str}".encode()).hexdigest()


class NsynthSplitAuditV2:
    """Detect split leakage by NSynth instrument identity."""

    def audit(self, split_notes: dict[SplitName, Iterable[NsynthNoteV2]]) -> None:
        owners: dict[str, set[str]] = defaultdict(set)
        for split, notes in split_notes.items():
            for note in notes:
                owners[f"id:{note.instrument}"].add(split)
                owners[f"name:{note.instrument_str}"].add(split)
        leaks = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
        if leaks:
            sample = dict(list(sorted(leaks.items()))[:5])
            raise ValueError(f"NSynth instrument leakage detected: {sample}")


class NsynthInstrumentSplitPolicyV2:
    """Assign each NSynth instrument to exactly one training split."""

    TRAIN_BUCKETS = 86
    VALIDATION_BUCKETS = 7
    TOTAL_BUCKETS = 100

    def split_for(self, note: NsynthNoteV2) -> SplitName:
        return self.split_for_instrument(note.instrument)

    def split_for_instrument(self, instrument: int) -> SplitName:
        digest = hashlib.blake2s(str(instrument).encode("utf-8"), digest_size=4).digest()
        bucket = int.from_bytes(digest, byteorder="big") % self.TOTAL_BUCKETS
        if bucket < self.TRAIN_BUCKETS:
            return "train"
        if bucket < self.TRAIN_BUCKETS + self.VALIDATION_BUCKETS:
            return "validation"
        return "test"


class NsynthCacheSelectorV2:
    """Pick a balanced bounded subset from a stream of candidate notes."""

    def __init__(
        self,
        target_notes: int,
        max_notes_per_instrument: int = 80,
        split: SplitName | None = None,
        split_policy: NsynthInstrumentSplitPolicyV2 | None = None,
    ):
        self.target_notes = target_notes
        self.max_notes_per_instrument = max_notes_per_instrument
        self.split = split
        self.split_policy = split_policy or NsynthInstrumentSplitPolicyV2()
        self._count_by_instrument: dict[str, int] = defaultdict(int)
        self._count_by_bucket: dict[tuple[str, str, int, int], int] = defaultdict(int)
        self._selected = 0

    @property
    def complete(self) -> bool:
        return self._selected >= self.target_notes

    def should_keep(self, note: NsynthNoteV2) -> bool:
        if self.complete:
            return False
        if self.split is not None and self.split_policy.split_for(note) != self.split:
            return False
        if self._count_by_instrument[note.instrument_str] >= self.max_notes_per_instrument:
            return False
        bucket = self._bucket(note)
        self._selected += 1
        self._count_by_instrument[note.instrument_str] += 1
        self._count_by_bucket[bucket] += 1
        return True

    @staticmethod
    def _bucket(note: NsynthNoteV2) -> tuple[str, str, int, int]:
        pitch_band = note.pitch // 12
        velocity_band = note.velocity // 32
        return note.family, note.source, pitch_band, velocity_band


class NsynthDownloaderV2:
    """Stream official NSynth JSON/WAV archives into a bounded local cache."""

    ARCHIVE_URLS: dict[SplitName, str] = {
        "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
        "validation": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
        "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
    }
    DEFAULT_TARGETS: dict[SplitName, int] = {
        "train": 25_000,
        "validation": 2_000,
        "test": 2_000,
    }

    def __init__(self, data_root: Path, storage_budget_gb: float = 10.0):
        self.paths = NsynthPathsV2(data_root)
        self.storage_budget_gb = storage_budget_gb

    def build_cache(self, targets: dict[SplitName, int] | None = None) -> None:
        chosen_targets = targets or self.DEFAULT_TARGETS
        for split, target in chosen_targets.items():
            self.stream_split(split, self.ARCHIVE_URLS[split], target)
        self.audit_leakage()

    def stream_split(
        self, split: SplitName, source: str, target_notes: int
    ) -> list[NsynthNoteV2]:
        self.paths.split_audio_root(split).mkdir(parents=True, exist_ok=True)
        selector = NsynthCacheSelectorV2(target_notes, split=split)
        notes: list[NsynthNoteV2] = []
        with self._open_stream(source) as stream:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    if selector.complete:
                        break
                    if not self._is_audio_member(member):
                        continue
                    note_str = Path(member.name).stem
                    target_path = self.paths.split_audio_root(split) / f"{note_str}.wav"
                    note = NsynthNoteNameV2.from_note_str(note_str, split, target_path)
                    if not selector.should_keep(note):
                        continue
                    src = archive.extractfile(member)
                    if src is None:
                        continue
                    with target_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    notes.append(note)
        ManifestIOV2.write_nsynth(self.paths.split_manifest(split), notes)
        return notes

    def load_manifests(self) -> dict[SplitName, list[NsynthNoteV2]]:
        return {
            split: ManifestIOV2.read_nsynth(self.paths.split_manifest(split))
            for split in ("train", "validation", "test")
        }

    def audit_leakage(self) -> None:
        NsynthSplitAuditV2().audit(self.load_manifests())

    @staticmethod
    def _open_stream(source: str):
        if source.startswith("http://") or source.startswith("https://"):
            return urllib.request.urlopen(source, timeout=60)
        return Path(source).open("rb")

    @staticmethod
    def _is_audio_member(member: tarfile.TarInfo) -> bool:
        path = Path(member.name)
        return member.isfile() and path.suffix.lower() == ".wav" and "audio" in path.parts


class NsynthMetadataCacheBuilderV2:
    """Build a bounded NSynth cache from official metadata identity splits."""

    def __init__(
        self,
        data_root: Path,
        sources: dict[SplitName, str] | None = None,
        train_instruments: int = 800,
        validation_instruments: int = 100,
        test_instruments: int = 100,
        notes_per_instrument: int = 24,
        seed: str = "muziq-nsynth-metadata-v2",
        progress_interval_s: float = 30.0,
    ):
        self.paths = NsynthPathsV2(data_root)
        self.sources = sources or NsynthDownloaderV2.ARCHIVE_URLS
        self.splitter = NsynthMetadataInstrumentSplitV2(
            train_instruments=train_instruments,
            validation_instruments=validation_instruments,
            test_instruments=test_instruments,
            seed=seed,
        )
        self.note_selector = NsynthMetadataNoteSelectorV2(
            notes_per_instrument=notes_per_instrument,
            seed=seed,
        )
        self.progress_interval_s = progress_interval_s

    def build_cache(self) -> dict[str, object]:
        metadata = self._load_all_metadata()
        assignment = self.splitter.assign(metadata.values())
        selected_note_names = self.note_selector.select(
            record
            for record in metadata.values()
            if record.instrument in assignment
        )
        self._clear_existing_cache()
        notes = self._extract_selected_audio(metadata, assignment, selected_note_names)
        for split in ("train", "validation", "test"):
            ManifestIOV2.write_nsynth(self.paths.split_manifest(split), notes[split])
        NsynthSplitAuditV2().audit(notes)
        summary = self._summary(notes, assignment, selected_note_names)
        (self.paths.manifest_root / "metadata_split_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def _load_all_metadata(self) -> dict[str, NsynthOfficialMetadataRecordV2]:
        records: dict[str, NsynthOfficialMetadataRecordV2] = {}
        for source_split, source in self.sources.items():
            records.update(self._load_archive_metadata(source_split, source))
        return records

    def _load_archive_metadata(
        self, source_split: SplitName, source: str
    ) -> dict[str, NsynthOfficialMetadataRecordV2]:
        started = monotonic()
        last_progress = started
        members_seen = 0
        with NsynthDownloaderV2._open_stream(source) as stream:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    members_seen += 1
                    now = monotonic()
                    if now - last_progress >= self.progress_interval_s:
                        self._emit_progress(
                            "nsynth_metadata_scan_progress",
                            source_split=source_split,
                            source=source,
                            members_seen=members_seen,
                            elapsed_s=round(now - started, 3),
                        )
                        last_progress = now
                    path = Path(member.name)
                    if not (member.isfile() and path.name == "examples.json"):
                        continue
                    src = archive.extractfile(member)
                    if src is None:
                        break
                    payload = json.load(src)
                    records = {
                        note_str: NsynthOfficialMetadataRecordV2.from_examples_json(
                            note_str, record, source_split
                        )
                        for note_str, record in payload.items()
                    }
                    self._emit_progress(
                        "nsynth_metadata_loaded",
                        source_split=source_split,
                        source=source,
                        records=len(records),
                        members_seen=members_seen,
                        elapsed_s=round(monotonic() - started, 3),
                    )
                    return records
        raise ValueError(f"No examples.json found in NSynth archive {source!r}")

    def _clear_existing_cache(self) -> None:
        if self.paths.cache_root.exists():
            shutil.rmtree(self.paths.cache_root)
        for split in ("train", "validation", "test"):
            self.paths.split_audio_root(split).mkdir(parents=True, exist_ok=True)

    def _extract_selected_audio(
        self,
        metadata: dict[str, NsynthOfficialMetadataRecordV2],
        assignment: dict[int, SplitName],
        selected_note_names: set[str],
    ) -> dict[SplitName, list[NsynthNoteV2]]:
        notes: dict[SplitName, list[NsynthNoteV2]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        remaining = set(selected_note_names)
        for source in self.sources.values():
            if not remaining:
                break
            started = monotonic()
            last_progress = started
            members_seen = 0
            extracted = 0
            with NsynthDownloaderV2._open_stream(source) as stream:
                with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                    for member in archive:
                        members_seen += 1
                        now = monotonic()
                        if now - last_progress >= self.progress_interval_s:
                            self._emit_progress(
                                "nsynth_audio_extract_progress",
                                source=source,
                                members_seen=members_seen,
                                extracted=extracted,
                                remaining=len(remaining),
                                elapsed_s=round(now - started, 3),
                            )
                            last_progress = now
                        if not remaining:
                            break
                        if not NsynthDownloaderV2._is_audio_member(member):
                            continue
                        note_str = Path(member.name).stem
                        if note_str not in remaining:
                            continue
                        record = metadata[note_str]
                        split = assignment[record.instrument]
                        target_path = self.paths.split_audio_root(split) / f"{note_str}.wav"
                        src = archive.extractfile(member)
                        if src is None:
                            continue
                        with target_path.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        notes[split].append(record.to_note(split, target_path))
                        extracted += 1
                        remaining.remove(note_str)
        if remaining:
            sample = sorted(remaining)[:10]
            raise ValueError(f"Missing selected NSynth audio files: {sample}")
        return notes

    @staticmethod
    def _emit_progress(event: str, **payload: object) -> None:
        print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)

    @staticmethod
    def _summary(
        notes: dict[SplitName, list[NsynthNoteV2]],
        assignment: dict[int, SplitName],
        selected_note_names: set[str],
    ) -> dict[str, object]:
        split_instruments: dict[str, set[int]] = defaultdict(set)
        for instrument, split in assignment.items():
            split_instruments[split].add(instrument)
        note_counts = {split: len(items) for split, items in notes.items()}
        family_counts = {
            split: dict(sorted(Counter(note.family for note in items).items()))
            for split, items in notes.items()
        }
        instrument_counts = {
            split: len({note.instrument for note in items})
            for split, items in notes.items()
        }
        return {
            "selected_note_count": len(selected_note_names),
            "note_counts": note_counts,
            "instrument_counts": instrument_counts,
            "assigned_instrument_counts": {
                split: len(split_instruments[split])
                for split in ("train", "validation", "test")
            },
            "family_counts": family_counts,
        }


class NsynthIndexV2:
    """Read cached notes and expose split/family/instrument groupings."""

    def __init__(self, data_root: Path):
        self.paths = NsynthPathsV2(data_root)
        self.notes_by_split = {
            split: ManifestIOV2.read_nsynth(self.paths.split_manifest(split))
            for split in ("train", "validation", "test")
        }

    def notes(self, split: SplitName) -> list[NsynthNoteV2]:
        return self.notes_by_split[split]

    def write_from_records(self, split: SplitName, notes: list[NsynthNoteV2]) -> None:
        ManifestIOV2.write_nsynth(self.paths.split_manifest(split), notes)
        self.notes_by_split[split] = notes

    def load_examples_json(
        self, examples_json: Path, split: SplitName, audio_root: Path
    ) -> list[NsynthNoteV2]:
        payload = json.loads(examples_json.read_text(encoding="utf-8"))
        notes = []
        records = payload.values() if isinstance(payload, dict) else payload
        for record in records:
            note_str = record["note_str"]
            qualities = tuple(record.get("qualities_str", ()))
            notes.append(
                NsynthNoteV2(
                    note_str=note_str,
                    split=split,
                    wav_path=str(audio_root / f"{note_str}.wav"),
                    instrument=int(record["instrument"]),
                    instrument_str=str(record["instrument_str"]),
                    family=str(record["instrument_family_str"]),
                    source=str(record["instrument_source_str"]),
                    pitch=int(record["pitch"]),
                    velocity=int(record["velocity"]),
                    qualities=qualities,
                )
            )
        return notes
