from __future__ import annotations

import pytest

from muziq_nn.datasets.nsynth import NsynthCacheSelectorV2, NsynthSplitAuditV2
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
