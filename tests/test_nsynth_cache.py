from __future__ import annotations

import pytest

from muziq_nn.datasets.nsynth import (
    NsynthCacheSelectorV2,
    NsynthInstrumentSplitPolicyV2,
    NsynthMetadataInstrumentSplitV2,
    NsynthMetadataNoteSelectorV2,
    NsynthOfficialMetadataRecordV2,
    NsynthSplitAuditV2,
)
from muziq_nn.datasets.schema import NsynthNoteV2


class TestNsynthCacheV2:
    def test_leakage_audit_rejects_shared_instrument(self):
        train = self._note("train", 1, "bass_acoustic_001")
        validation = self._note("validation", 1, "bass_acoustic_001")

        with pytest.raises(ValueError, match="instrument leakage"):
            NsynthSplitAuditV2().audit(
                {"train": [train], "validation": [validation], "test": []}
            )

    def test_cache_selector_caps_notes_per_instrument(self):
        selector = NsynthCacheSelectorV2(target_notes=10, max_notes_per_instrument=2)
        notes = [self._note("train", 1, "bass_acoustic_001", pitch=48 + i) for i in range(8)]

        kept = [note for note in notes if selector.should_keep(note)]

        assert len(kept) == 2

    def test_cache_selector_rejects_instruments_assigned_to_other_splits(self):
        policy = NsynthInstrumentSplitPolicyV2()
        note = self._note("train", 1, "bass_acoustic_001")
        assigned_split = policy.split_for(note)
        rejected_split = next(
            split for split in ("train", "validation", "test") if split != assigned_split
        )
        selector = NsynthCacheSelectorV2(
            target_notes=1, split=rejected_split, split_policy=policy
        )

        assert not selector.should_keep(note)

    def test_official_metadata_overrides_filename_instrument_suffix(self):
        record = NsynthOfficialMetadataRecordV2.from_examples_json(
            "bass_synthetic_033-060-096",
            {
                "instrument": 417,
                "instrument_str": "bass_synthetic_033",
                "instrument_family_str": "bass",
                "instrument_source_str": "synthetic",
                "pitch": 60,
                "velocity": 96,
                "qualities_str": ["bright"],
            },
            "train",
        )

        note = record.to_note("validation", "/tmp/bass.wav")

        assert note.instrument == 417
        assert note.instrument_str == "bass_synthetic_033"
        assert note.split == "validation"
        assert note.qualities == ("bright",)

    def test_metadata_splitter_assigns_disjoint_official_instruments(self):
        records = [
            self._record(instrument, pitch)
            for instrument in range(6)
            for pitch in range(60, 64)
        ]
        assignment = NsynthMetadataInstrumentSplitV2(
            train_instruments=3,
            validation_instruments=2,
            test_instruments=1,
            seed="test",
        ).assign(records)
        selected = NsynthMetadataNoteSelectorV2(
            notes_per_instrument=2,
            seed="test",
        ).select(records)

        assert set(assignment) == set(range(6))
        assert list(assignment.values()).count("train") == 3
        assert list(assignment.values()).count("validation") == 2
        assert list(assignment.values()).count("test") == 1
        assert len(selected) == 12

    @staticmethod
    def _note(
        split,
        instrument: int,
        instrument_str: str,
        pitch: int = 60,
    ) -> NsynthNoteV2:
        return NsynthNoteV2(
            note_str=f"{instrument_str}-{pitch:03d}-096",
            split=split,
            wav_path="/tmp/nope.wav",
            instrument=instrument,
            instrument_str=instrument_str,
            family=instrument_str.split("_")[0],
            source="acoustic",
            pitch=pitch,
            velocity=96,
            qualities=(),
        )

    @staticmethod
    def _record(instrument: int, pitch: int) -> NsynthOfficialMetadataRecordV2:
        return NsynthOfficialMetadataRecordV2(
            note_str=f"bass_synthetic_{instrument:03d}-{pitch:03d}-096",
            source_split="train",
            instrument=instrument,
            instrument_str=f"bass_synthetic_{instrument:03d}",
            family="bass",
            source="synthetic",
            pitch=pitch,
            velocity=96,
            qualities=(),
        )
