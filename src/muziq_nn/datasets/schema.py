"""Shared dataset records for bounded-cache source tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SplitName = Literal["train", "validation", "test"]


class NsynthNoteV2(BaseModel):
    """One cached NSynth note and its official metadata."""

    model_config = ConfigDict(frozen=True)

    note_str: str
    split: SplitName
    wav_path: str
    instrument: int
    instrument_str: str
    family: str
    source: str
    pitch: int
    velocity: int
    qualities: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def absolute_wav_path(self) -> Path:
        return Path(self.wav_path).expanduser()


class MidiNoteEventV2(BaseModel):
    """One MIDI note event converted to seconds."""

    model_config = ConfigDict(frozen=True)

    start_s: float
    end_s: float
    pitch: int
    velocity: int
    track: int
    channel: int | None = None
    program: int | None = None


class MidiScheduleV2(BaseModel):
    """Compact parsed MIDI schedule used as timing only."""

    model_config = ConfigDict(frozen=True)

    schedule_id: str
    split: SplitName
    duration_s: float
    source_path: str
    events: tuple[MidiNoteEventV2, ...]

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted({event.track for event in self.events}))


class SourceEventLabelV2(BaseModel):
    """Ground-truth activity interval for one rendered source."""

    model_config = ConfigDict(frozen=True)

    source_id: int
    family: str
    family_index: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class SourceLabelFramesV2:
    active: np.ndarray
    onset: np.ndarray
    offset: np.ndarray
    onset_delta: np.ndarray
    offset_delta: np.ndarray
    onset_timing_mask: np.ndarray
    offset_timing_mask: np.ndarray
    family: np.ndarray
    source_id: np.ndarray
    frame_times: np.ndarray


@dataclass(frozen=True)
class RenderedSourceTrackingExampleV2:
    stage: str
    split: SplitName
    seed: int
    audio: np.ndarray
    frames: np.ndarray
    labels: SourceLabelFramesV2
    events: tuple[SourceEventLabelV2, ...]
    sample_rate: int


class ManifestIOV2:
    """Read and write JSONL manifests without retaining large records in memory."""

    @staticmethod
    def write_nsynth(path: Path, notes: list[NsynthNoteV2]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for note in notes:
                fh.write(note.model_dump_json() + "\n")

    @staticmethod
    def read_nsynth(path: Path) -> list[NsynthNoteV2]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            return [NsynthNoteV2.model_validate_json(line) for line in fh if line.strip()]

    @staticmethod
    def write_midi(path: Path, schedules: list[MidiScheduleV2]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for schedule in schedules:
                fh.write(schedule.model_dump_json() + "\n")

    @staticmethod
    def read_midi(path: Path) -> list[MidiScheduleV2]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            return [MidiScheduleV2.model_validate_json(line) for line in fh if line.strip()]
