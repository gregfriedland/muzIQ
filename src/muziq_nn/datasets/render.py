"""On-the-fly source-tracking example rendering."""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np
import soundfile as sf
from scipy import signal as sps

from muziq_nn.datasets.midi import MidiIndexV2
from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.schema import (
    MidiScheduleV2,
    NsynthNoteV2,
    RenderedSourceTrackingExampleV2,
    SourceEventLabelV2,
    SourceLabelFramesV2,
    SplitName,
)


class SourceTrackingAudioConfigV2:
    """Audio and frame-shape constants for on-the-fly rendering."""

    sample_rate = 16_000
    duration_s = 30.0
    hop = 80
    win = 512
    bands = 40
    max_sources = 5
    log_min_hz = 40.0
    log_max_hz = 8_000.0


class FamilyVocabularyV2:
    """Stable family-to-index mapping."""

    DEFAULT_FAMILIES = (
        "bass",
        "brass",
        "flute",
        "guitar",
        "keyboard",
        "mallet",
        "organ",
        "reed",
        "string",
        "synth_lead",
        "vocal",
    )

    def __init__(self, families: tuple[str, ...] | None = None):
        self.families = families or self.DEFAULT_FAMILIES
        self.to_index = {family: idx for idx, family in enumerate(self.families)}

    def index(self, family: str) -> int:
        if family not in self.to_index:
            self.to_index[family] = len(self.to_index)
            self.families = tuple(self.to_index)
        return self.to_index[family]


class AudioFrameExtractorV2:
    """Convert rendered audio into 40 log-spaced magnitude frames."""

    def __init__(self, config: type[SourceTrackingAudioConfigV2] = SourceTrackingAudioConfigV2):
        self.config = config
        self._window = np.hanning(config.win).astype(np.float32)
        self._fold = self._build_log_fold()

    def extract(self, audio: np.ndarray) -> np.ndarray:
        n_frames = self._frame_count(audio)
        if n_frames == 0:
            return np.zeros((0, self.config.bands), dtype=np.float32)
        return self._normalize_bands(self._raw_bands(audio, 0, n_frames))

    def extract_context(
        self,
        audio: np.ndarray,
        end_frame: int,
        frame_count: int,
        peak_warmup_frames: int,
    ) -> np.ndarray:
        n_frames = self._frame_count(audio)
        if n_frames == 0 or frame_count <= 0:
            return np.zeros((0, self.config.bands), dtype=np.float32)
        end_frame = min(max(0, int(end_frame)), n_frames - 1)
        output_start = max(0, end_frame - frame_count + 1)
        normalize_start = max(0, output_start - max(0, peak_warmup_frames))
        raw_bands = self._raw_bands(
            audio,
            normalize_start,
            end_frame - normalize_start + 1,
        )
        normalized = self._normalize_bands(raw_bands)
        return normalized[output_start - normalize_start :]

    def _raw_bands(
        self,
        audio: np.ndarray,
        start_frame: int,
        frame_count: int,
    ) -> np.ndarray:
        n_frames = self._frame_count(audio)
        target_samples = n_frames * self.config.hop
        tail = target_samples - len(audio)
        padded = np.pad(
            audio.astype(np.float32, copy=False),
            (self.config.win - self.config.hop, tail),
        )
        window_start = start_frame * self.config.hop
        window_stop = (start_frame + frame_count) * self.config.hop
        windows = np.lib.stride_tricks.sliding_window_view(padded, self.config.win)[
            window_start:window_stop:self.config.hop
        ]
        mag = np.abs(np.fft.rfft(windows * self._window, axis=1))
        return (mag @ self._fold.T).astype(np.float32)

    def _normalize_bands(self, raw_bands: np.ndarray) -> np.ndarray:
        frames = np.zeros_like(raw_bands, dtype=np.float32)
        peak = np.full(self.config.bands, 1e-6, dtype=np.float32)
        for frame_idx, bands in enumerate(raw_bands):
            peak = np.maximum(bands, peak * 0.9996)
            frames[frame_idx] = np.clip(bands / np.maximum(peak, 1e-6), 0.0, 1.0)
        return frames

    def _frame_count(self, audio: np.ndarray) -> int:
        return int(np.ceil(len(audio) / self.config.hop))

    def _build_log_fold(self) -> np.ndarray:
        freqs = np.fft.rfftfreq(self.config.win, 1.0 / self.config.sample_rate)
        edges = np.logspace(
            np.log10(self.config.log_min_hz),
            np.log10(self.config.log_max_hz),
            self.config.bands + 1,
        )
        fold = np.zeros((self.config.bands, len(freqs)), dtype=np.float32)
        for idx in range(self.config.bands):
            bins = np.where((freqs >= edges[idx]) & (freqs < edges[idx + 1]))[0]
            if len(bins) == 0:
                bins = np.array([min(np.searchsorted(freqs, edges[idx]), len(freqs) - 1)])
            fold[idx, bins] = 1.0 / len(bins)
        return fold


class NsynthNoteStoreV2:
    """Lookup and decode cached NSynth notes with a small in-memory LRU."""

    def __init__(
        self,
        index: NsynthIndexV2,
        sample_rate: int = SourceTrackingAudioConfigV2.sample_rate,
        cache_items: int = 256,
    ):
        self.index = index
        self.sample_rate = sample_rate
        self.cache_items = cache_items
        self._decoded: OrderedDict[str, np.ndarray] = OrderedDict()
        self._by_split_instrument: dict[SplitName, dict[str, list[NsynthNoteV2]]] = {
            split: self._group_by_instrument(index.notes(split))
            for split in ("train", "validation", "test")
        }
        self._by_split_family: dict[SplitName, dict[str, list[NsynthNoteV2]]] = {
            split: self._group_by_family(index.notes(split))
            for split in ("train", "validation", "test")
        }

    def sample_instrument_notes(
        self,
        split: SplitName,
        rng: np.random.Generator,
        family: str | None = None,
    ) -> list[NsynthNoteV2]:
        if family is not None and self._by_split_family[split].get(family):
            candidates = self._by_split_family[split][family]
            note = candidates[int(rng.integers(len(candidates)))]
            return self._by_split_instrument[split][note.instrument_str]
        instruments = list(self._by_split_instrument[split].values())
        if not instruments:
            raise ValueError(f"No NSynth notes cached for split {split!r}")
        return instruments[int(rng.integers(len(instruments)))]

    def nearest_pitch(self, notes: list[NsynthNoteV2], pitch: int) -> NsynthNoteV2:
        return min(notes, key=lambda note: (abs(note.pitch - pitch), abs(note.velocity - 96)))

    def decode(self, note: NsynthNoteV2) -> np.ndarray:
        cached = self._decoded.get(note.wav_path)
        if cached is not None:
            self._decoded.move_to_end(note.wav_path)
            return cached.copy()
        audio, sample_rate = sf.read(note.absolute_wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = self._resample(audio.astype(np.float32), sample_rate)
        self._decoded[note.wav_path] = audio
        self._decoded.move_to_end(note.wav_path)
        while len(self._decoded) > self.cache_items:
            self._decoded.popitem(last=False)
        return audio.copy()

    def random_note(self, split: SplitName, rng: np.random.Generator) -> NsynthNoteV2:
        notes = self.index.notes(split)
        if not notes:
            raise ValueError(f"No NSynth notes cached for split {split!r}")
        return notes[int(rng.integers(len(notes)))]

    def _resample(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.sample_rate:
            return audio
        gcd = int(np.gcd(sample_rate, self.sample_rate))
        return sps.resample_poly(audio, self.sample_rate // gcd, sample_rate // gcd).astype(
            np.float32
        )

    @staticmethod
    def _group_by_instrument(notes: list[NsynthNoteV2]) -> dict[str, list[NsynthNoteV2]]:
        grouped: dict[str, list[NsynthNoteV2]] = defaultdict(list)
        for note in notes:
            grouped[note.instrument_str].append(note)
        return dict(grouped)

    @staticmethod
    def _group_by_family(notes: list[NsynthNoteV2]) -> dict[str, list[NsynthNoteV2]]:
        grouped: dict[str, list[NsynthNoteV2]] = defaultdict(list)
        for note in notes:
            grouped[note.family].append(note)
        return dict(grouped)


class MidiScheduleStoreV2:
    """Lookup parsed MIDI schedules by split."""

    def __init__(self, index: MidiIndexV2):
        self.index = index

    def sample(self, split: SplitName, rng: np.random.Generator) -> MidiScheduleV2 | None:
        schedules = self.index.schedules(split)
        if not schedules:
            return None
        return schedules[int(rng.integers(len(schedules)))]


class SourceTrackingRendererV2:
    """Render 30-second source-tracking examples fully in memory."""

    HARD_FAMILIES = (
        ("bass", "keyboard"),
        ("guitar", "mallet"),
        ("flute", "vocal"),
        ("brass", "reed"),
    )

    def __init__(
        self,
        note_store: NsynthNoteStoreV2,
        midi_store: MidiScheduleStoreV2 | None = None,
        config: type[SourceTrackingAudioConfigV2] = SourceTrackingAudioConfigV2,
    ):
        self.note_store = note_store
        self.midi_store = midi_store
        self.config = config
        self.extractor = AudioFrameExtractorV2(config)
        self.families = FamilyVocabularyV2()

    def render(
        self, stage: str, split: SplitName, seed: int
    ) -> RenderedSourceTrackingExampleV2:
        audio, events = self._render_audio_events(stage, split, seed)
        frames = self.extractor.extract(audio)
        labels = self._labels_from_events(events, len(audio))
        return RenderedSourceTrackingExampleV2(
            stage=stage,
            split=split,
            seed=seed,
            audio=audio,
            frames=frames,
            labels=labels,
            events=tuple(events),
            sample_rate=self.config.sample_rate,
        )

    def render_training_slice(
        self,
        stage: str,
        split: SplitName,
        seed: int,
        frame_count: int,
        peak_warmup_frames: int,
    ) -> dict[str, np.ndarray]:
        audio, events = self._render_audio_events(stage, split, seed)
        frame_idx = self._sample_active_frame_from_events(events, len(audio))
        target = self._label_at_frame(events, len(audio), frame_idx)
        return {
            "frames": self.extractor.extract_context(
                audio,
                end_frame=frame_idx,
                frame_count=frame_count,
                peak_warmup_frames=peak_warmup_frames,
            ),
            **target,
        }

    def render_training_audio_slice(
        self,
        stage: str,
        split: SplitName,
        seed: int,
        frame_count: int,
        peak_warmup_frames: int,
    ) -> dict[str, np.ndarray]:
        audio, events = self._render_audio_events(stage, split, seed)
        frame_idx = self._sample_active_frame_from_events(events, len(audio))
        target = self._label_at_frame(events, len(audio), frame_idx)
        return {
            "audio_context": self._audio_context_for_frame(
                audio,
                end_frame=frame_idx,
                frame_count=frame_count,
                peak_warmup_frames=peak_warmup_frames,
            ),
            **target,
        }

    def _render_audio_events(
        self,
        stage: str,
        split: SplitName,
        seed: int,
    ) -> tuple[np.ndarray, list[SourceEventLabelV2]]:
        rng = np.random.default_rng(seed)
        audio = np.zeros(
            int(self.config.duration_s * self.config.sample_rate), dtype=np.float32
        )
        events: list[SourceEventLabelV2] = []
        if stage == "single_note_all":
            self._render_single_note(split, rng, audio, events)
        elif stage == "single_instrument_melody":
            self._render_generated_melody(split, rng, audio, events, source_count=1)
        elif stage == "simple_duo_trio":
            self._render_generated_melody(
                split, rng, audio, events, source_count=int(rng.integers(2, 4))
            )
        elif stage == "midi_complex":
            if not self._render_midi(split, rng, audio, events):
                self._render_generated_melody(split, rng, audio, events, source_count=3)
        elif stage == "hard_case_finetune":
            self._render_hard_case(split, rng, audio, events)
        else:
            raise ValueError(f"Unknown curriculum stage {stage!r}")
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio = (audio / peak * 0.8).astype(np.float32)
        return audio, events

    def _render_single_note(
        self,
        split: SplitName,
        rng: np.random.Generator,
        audio: np.ndarray,
        events: list[SourceEventLabelV2],
    ) -> None:
        note = self.note_store.random_note(split, rng)
        start_s = float(rng.uniform(0.25, 1.25))
        self._mix_note(audio, note, 0, start_s, events)

    def _render_generated_melody(
        self,
        split: SplitName,
        rng: np.random.Generator,
        audio: np.ndarray,
        events: list[SourceEventLabelV2],
        source_count: int,
    ) -> None:
        for source_id in range(min(source_count, self.config.max_sources)):
            notes = self.note_store.sample_instrument_notes(split, rng)
            offset = float(source_id) * 0.17
            period = float(rng.uniform(0.45, 0.95))
            t = 0.5 + offset
            while t < self.config.duration_s - 1.0:
                pitch = int(rng.integers(36, 88))
                note = self.note_store.nearest_pitch(notes, pitch)
                self._mix_note(audio, note, source_id, t, events)
                t += period + float(rng.uniform(-0.05, 0.05))

    def _render_midi(
        self,
        split: SplitName,
        rng: np.random.Generator,
        audio: np.ndarray,
        events: list[SourceEventLabelV2],
    ) -> bool:
        if self.midi_store is None:
            return False
        schedule = self.midi_store.sample(split, rng)
        if schedule is None:
            return False
        tracks = schedule.track_ids[: self.config.max_sources]
        if not tracks:
            return False
        instrument_notes = {
            track: self.note_store.sample_instrument_notes(split, rng) for track in tracks
        }
        source_for_track = {track: idx for idx, track in enumerate(tracks)}
        window_offset = float(
            rng.uniform(0.0, max(0.0, schedule.duration_s - self.config.duration_s))
        )
        for event in schedule.events:
            if event.track not in source_for_track:
                continue
            start_s = event.start_s - window_offset
            if start_s < 0.0 or start_s >= self.config.duration_s:
                continue
            note = self.note_store.nearest_pitch(instrument_notes[event.track], event.pitch)
            self._mix_note(audio, note, source_for_track[event.track], start_s, events)
        return bool(events)

    def _render_hard_case(
        self,
        split: SplitName,
        rng: np.random.Generator,
        audio: np.ndarray,
        events: list[SourceEventLabelV2],
    ) -> None:
        pair = self.HARD_FAMILIES[int(rng.integers(len(self.HARD_FAMILIES)))]
        for source_id, family in enumerate(pair):
            notes = self.note_store.sample_instrument_notes(split, rng, family=family)
            for start_s in np.arange(0.75, self.config.duration_s - 1.0, 0.75):
                pitch = int(rng.integers(36, 88))
                note = self.note_store.nearest_pitch(notes, pitch)
                self._mix_note(audio, note, source_id, float(start_s), events)

    def _mix_note(
        self,
        audio: np.ndarray,
        note: NsynthNoteV2,
        source_id: int,
        start_s: float,
        events: list[SourceEventLabelV2],
    ) -> None:
        note_audio = self._trim_and_fade(self.note_store.decode(note))
        start = int(round(start_s * self.config.sample_rate))
        if start >= len(audio):
            return
        end = min(len(audio), start + len(note_audio))
        if end <= start:
            return
        audio[start:end] += note_audio[: end - start]
        events.append(
            SourceEventLabelV2(
                source_id=source_id,
                family=note.family,
                family_index=self.families.index(note.family),
                start_s=start / self.config.sample_rate,
                end_s=end / self.config.sample_rate,
            )
        )

    def _trim_and_fade(self, audio: np.ndarray) -> np.ndarray:
        max_len = int(4.0 * self.config.sample_rate)
        clipped = audio[:max_len].copy()
        fade = min(len(clipped), int(0.01 * self.config.sample_rate))
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clipped[:fade] *= ramp
            clipped[-fade:] *= ramp[::-1]
        return clipped.astype(np.float32)

    def _labels_from_events(
        self,
        events: list[SourceEventLabelV2],
        n_samples: int,
    ) -> SourceLabelFramesV2:
        n_frames = int(np.ceil(n_samples / self.config.hop))
        shape = (n_frames, self.config.max_sources)
        active = np.zeros(shape, dtype=np.float32)
        onset = np.zeros(shape, dtype=np.float32)
        offset = np.zeros(shape, dtype=np.float32)
        family = np.full(shape, -1, dtype=np.int32)
        source_id = np.full(shape, -1, dtype=np.int32)
        frame_times = np.arange(n_frames, dtype=np.float32) * (
            self.config.hop / self.config.sample_rate
        )
        for event in events:
            if event.source_id >= self.config.max_sources:
                continue
            start, end = self._event_frame_span(event, n_frames)
            if end <= start:
                continue
            active[start:end, event.source_id] = 1.0
            family[start:end, event.source_id] = event.family_index
            source_id[start:end, event.source_id] = event.source_id
            onset[start, event.source_id] = 1.0
            offset[end - 1, event.source_id] = 1.0
        return SourceLabelFramesV2(
            active=active,
            onset=onset,
            offset=offset,
            family=family,
            source_id=source_id,
            frame_times=frame_times,
        )

    def _sample_active_frame_from_events(
        self,
        events: list[SourceEventLabelV2],
        n_samples: int,
    ) -> int:
        n_frames = int(np.ceil(n_samples / self.config.hop))
        active = np.zeros(n_frames, dtype=np.bool_)
        for event in events:
            if event.source_id >= self.config.max_sources:
                continue
            start, end = self._event_frame_span(event, n_frames)
            if end > start:
                active[start:end] = True
        active_frames = np.flatnonzero(active)
        if len(active_frames) == 0:
            return n_frames - 1
        return int(active_frames[len(active_frames) // 2])

    def _label_at_frame(
        self,
        events: list[SourceEventLabelV2],
        n_samples: int,
        frame_idx: int,
    ) -> dict[str, np.ndarray]:
        n_frames = int(np.ceil(n_samples / self.config.hop))
        activity = np.zeros(self.config.max_sources, dtype=np.float32)
        onset = np.zeros(self.config.max_sources, dtype=np.float32)
        offset = np.zeros(self.config.max_sources, dtype=np.float32)
        family = np.full(self.config.max_sources, -1, dtype=np.int32)
        for event in events:
            if event.source_id >= self.config.max_sources:
                continue
            start, end = self._event_frame_span(event, n_frames)
            if start <= frame_idx < end:
                activity[event.source_id] = 1.0
                family[event.source_id] = event.family_index
            if frame_idx == start:
                onset[event.source_id] = 1.0
            if frame_idx == end - 1:
                offset[event.source_id] = 1.0
        return {
            "activity": activity,
            "family": family,
            "onset": onset,
            "offset": offset,
        }

    def _event_frame_span(
        self,
        event: SourceEventLabelV2,
        n_frames: int,
    ) -> tuple[int, int]:
        start = max(
            0, int(np.floor(event.start_s * self.config.sample_rate / self.config.hop))
        )
        end = min(
            n_frames,
            int(np.ceil(event.end_s * self.config.sample_rate / self.config.hop)),
        )
        return start, end

    def _audio_context_for_frame(
        self,
        audio: np.ndarray,
        end_frame: int,
        frame_count: int,
        peak_warmup_frames: int,
    ) -> np.ndarray:
        raw_frame_count = max(1, frame_count + max(0, peak_warmup_frames))
        segment_len = self.config.win + (raw_frame_count - 1) * self.config.hop
        end_sample = (end_frame + 1) * self.config.hop
        start_sample = end_sample - segment_len
        context = np.zeros(segment_len, dtype=np.float32)
        src_start = max(0, start_sample)
        src_end = min(len(audio), end_sample)
        if src_end <= src_start:
            return context
        dst_start = src_start - start_sample
        context[dst_start : dst_start + src_end - src_start] = audio[src_start:src_end]
        return context
