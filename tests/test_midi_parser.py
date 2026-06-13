from __future__ import annotations

import pytest
from conftest import TinyCorpusBuilderV2

from muziq_nn.datasets.midi import MidiScheduleParserV2, MidiSplitAuditV2


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

    def test_leakage_audit_rejects_shared_hash(self):
        schedule = MidiScheduleParserV2().parse_bytes(
            TinyCorpusBuilderV2._midi_bytes(), "fixture.mid"
        )
        duplicate = schedule.model_copy(update={"split": "validation"})

        with pytest.raises(ValueError, match="MIDI hash leakage"):
            MidiSplitAuditV2().audit(
                {"train": [schedule], "validation": [duplicate], "test": []}
            )
