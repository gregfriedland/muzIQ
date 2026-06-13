from __future__ import annotations

import io
import tarfile

import pytest
from conftest import TinyCorpusBuilderV2

from muziq_nn.datasets.midi import (
    LakhMidiDownloaderV2,
    MidiScheduleBoundsV2,
    MidiScheduleParserV2,
    MidiSplitAuditV2,
)
from muziq_nn.datasets.schema import MidiNoteEventV2, MidiScheduleV2


class TestMidiParserV2:
    def test_parser_handles_chords_and_tracks(self):
        payload = TinyCorpusBuilderV2._midi_bytes()

        schedule = MidiScheduleParserV2().parse_bytes(payload, "fixture.mid")

        assert len(schedule.events) == 3
        assert sorted({event.track for event in schedule.events}) == [0, 1]
        assert schedule.duration_s > 0.0

    def test_parser_rejects_corrupt_bytes(self):
        with pytest.raises((EOFError, OSError, ValueError)):
            MidiScheduleParserV2().parse_bytes(b"not a midi file", "bad.mid")

    def test_parser_wraps_mido_loader_errors(self, monkeypatch):
        def raise_loader_error(*args, **kwargs):
            raise IndexError("truncated metadata")

        monkeypatch.setattr("muziq_nn.datasets.midi.mido.MidiFile", raise_loader_error)

        with pytest.raises(ValueError, match="invalid MIDI file bad.mid"):
            MidiScheduleParserV2().parse_bytes(b"bad", "bad.mid")

    def test_parser_rejects_event_cap_during_parse(self):
        payload = TinyCorpusBuilderV2._midi_bytes()

        with pytest.raises(ValueError, match="exceeds"):
            MidiScheduleParserV2().parse_bytes(payload, "fixture.mid", max_events=1)

    def test_leakage_audit_rejects_shared_hash(self):
        schedule = MidiScheduleParserV2().parse_bytes(
            TinyCorpusBuilderV2._midi_bytes(), "fixture.mid"
        )
        duplicate = schedule.model_copy(update={"split": "validation"})

        with pytest.raises(ValueError, match="MIDI hash leakage"):
            MidiSplitAuditV2().audit(
                {"train": [schedule], "validation": [duplicate], "test": []}
            )

    def test_downloader_skips_mido_parse_errors(self, tmp_path):
        class SyntheticMidoMetadataErrorV2(Exception):
            pass

        SyntheticMidoMetadataErrorV2.__module__ = "mido.midifiles.meta"

        class ParserWithOneBadMidiV2:
            def __init__(self):
                self.calls = 0

            def parse_bytes(
                self, payload: bytes, source_path: str, max_events: int | None = None
            ):
                self.calls += 1
                if self.calls == 1:
                    raise SyntheticMidoMetadataErrorV2("bad key signature")
                schedule = MidiScheduleParserV2().parse_bytes(payload, source_path)
                return schedule.model_copy(update={"split": "train"})

        archive_path = tmp_path / "lakh_subset.tar.gz"
        good_payload = TinyCorpusBuilderV2._midi_bytes()
        with tarfile.open(archive_path, "w:gz") as archive:
            for name, payload in (("bad.mid", b"bad"), ("good.mid", good_payload)):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

        downloader = LakhMidiDownloaderV2(tmp_path / "data")
        downloader.parser = ParserWithOneBadMidiV2()
        downloader.build_cache(
            str(archive_path), {"train": 1, "validation": 0, "test": 0}
        )

        assert len(downloader.load_manifests()["train"]) == 1

    def test_downloader_rejects_oversized_schedules(self, tmp_path):
        class OversizedParserV2:
            def parse_bytes(
                self, payload: bytes, source_path: str, max_events: int | None = None
            ):
                event = MidiNoteEventV2(
                    start_s=0.0,
                    end_s=0.1,
                    pitch=60,
                    velocity=96,
                    track=0,
                )
                return MidiScheduleV2(
                    schedule_id="0" * 32,
                    split="train",
                    duration_s=1.0,
                    source_path=source_path,
                    events=(event,) * (MidiScheduleBoundsV2.MAX_EVENTS + 1),
                )

        archive_path = tmp_path / "lakh_subset.tar.gz"
        payload = TinyCorpusBuilderV2._midi_bytes()
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo("oversized.mid")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        downloader = LakhMidiDownloaderV2(tmp_path / "data")
        downloader.parser = OversizedParserV2()
        downloader.build_cache(
            str(archive_path), {"train": 1, "validation": 0, "test": 0}
        )

        assert len(downloader.load_manifests()["train"]) == 0
