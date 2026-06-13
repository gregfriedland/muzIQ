"""NSynth bounded-cache ingestion."""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

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


class NsynthCacheSelectorV2:
    """Pick a balanced bounded subset from a stream of candidate notes."""

    def __init__(self, target_notes: int, max_notes_per_instrument: int = 80):
        self.target_notes = target_notes
        self.max_notes_per_instrument = max_notes_per_instrument
        self._count_by_instrument: dict[str, int] = defaultdict(int)
        self._count_by_bucket: dict[tuple[str, str, int, int], int] = defaultdict(int)
        self._selected = 0

    @property
    def complete(self) -> bool:
        return self._selected >= self.target_notes

    def should_keep(self, note: NsynthNoteV2) -> bool:
        if self.complete:
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
        selector = NsynthCacheSelectorV2(target_notes)
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
