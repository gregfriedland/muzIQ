from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from muziq_nn.datasets.midi import MidiIndexV2, MidiScheduleParserV2
from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.schema import NsynthNoteV2


class TinyCorpusBuilderV2:
    """Build tiny fake NSynth/MIDI caches for tests."""

    FAMILIES = ("bass", "guitar", "flute", "vocal")

    def __init__(self, root: Path):
        self.root = root

    def build(self) -> None:
        self._write_nsynth()
        self._write_midi()

    def _write_nsynth(self) -> None:
        index = NsynthIndexV2(self.root)
        for split, instrument_offset in (
            ("train", 0),
            ("validation", 100),
            ("test", 200),
        ):
            notes = []
            audio_root = self.root / "nsynth_cache" / split / "audio"
            audio_root.mkdir(parents=True, exist_ok=True)
            for family_idx, family in enumerate(self.FAMILIES):
                for pitch in (48, 60, 72):
                    instrument = instrument_offset + family_idx
                    note_str = f"{family}_acoustic_{instrument:03d}-{pitch:03d}-096"
                    wav_path = audio_root / f"{note_str}.wav"
                    self._write_tone(wav_path, pitch)
                    notes.append(
                        NsynthNoteV2(
                            note_str=note_str,
                            split=split,
                            wav_path=str(wav_path),
                            instrument=instrument,
                            instrument_str=f"{family}_acoustic_{instrument:03d}",
                            family=family,
                            source="acoustic",
                            pitch=pitch,
                            velocity=96,
                            qualities=(),
                        )
                    )
            index.write_from_records(split, notes)

    def _write_midi(self) -> None:
        midi_bytes = self._midi_bytes()
        schedule = MidiScheduleParserV2().parse_bytes(midi_bytes, "tiny.mid")
        index = MidiIndexV2(self.root)
        index.write_from_records("train", [schedule.model_copy(update={"split": "train"})])
        index.write_from_records(
            "validation", [schedule.model_copy(update={"split": "validation"})]
        )
        index.write_from_records("test", [schedule.model_copy(update={"split": "test"})])

    @staticmethod
    def _write_tone(path: Path, pitch: int) -> None:
        sample_rate = 16_000
        duration = 0.4
        t = np.arange(int(sample_rate * duration)) / sample_rate
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        env = np.exp(-t * 3.0)
        audio = (0.2 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)
        sf.write(path, audio, sample_rate)

    @staticmethod
    def _midi_bytes() -> bytes:
        midi = mido.MidiFile(ticks_per_beat=480)
        track0 = mido.MidiTrack()
        track1 = mido.MidiTrack()
        midi.tracks.append(track0)
        midi.tracks.append(track1)
        track0.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        track0.append(mido.Message("program_change", program=32, channel=0, time=0))
        track0.append(mido.Message("note_on", note=60, velocity=96, channel=0, time=0))
        track0.append(mido.Message("note_on", note=64, velocity=96, channel=0, time=0))
        track0.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
        track0.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=0))
        track1.append(mido.Message("program_change", program=40, channel=1, time=0))
        track1.append(mido.Message("note_on", note=67, velocity=80, channel=1, time=240))
        track1.append(mido.Message("note_off", note=67, velocity=0, channel=1, time=480))
        from io import BytesIO

        buffer = BytesIO()
        midi.save(file=buffer)
        return buffer.getvalue()
