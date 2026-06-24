"""On-the-fly source-tracking example rendering."""

from __future__ import annotations

import os
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
    fine_hop = 40
    hop = 160
    win = 384
    bands = 128
    max_sources = 5
    log_min_hz = 40.0
    log_max_hz = 8_000.0
    log_feature_scale = 1_000.0
    onset_shoulder_ms = 25.0
    onset_shoulder_radius_frames = 3
    offset_label_radius_frames = 16
    boundary_negative_radius_frames = 24
    onset_hard_negative_sample_prob = 0.0
    onset_context_sample_prob = 0.0
    positive_family_boosts = ""
    positive_quality_boosts = ""
    positive_source_boosts = ""
    positive_velocity_boosts = ""
    censor_same_instrument_overlap_onset_families = ""
    hard_context_sample_prob = 0.0
    hard_context_mix = "reattack=0.4,post_onset=0.3,pre_onset=0.15,ordinary=0.15"
    hard_context_positive_families = ""
    hard_context_negative_families = ""
    hard_context_reattack_min_ms = 400.0
    hard_context_reattack_max_ms = 900.0
    hard_context_min_remaining_ms = 1000.0
    hard_context_post_onset_min_ms = 25.0
    hard_context_post_onset_max_ms = 75.0
    hard_context_pre_onset_min_ms = 25.0
    hard_context_pre_onset_max_ms = 100.0


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
    """Legacy FFT frontend kept only for explicit compatibility tests."""

    def __init__(self, config: type[SourceTrackingAudioConfigV2] = SourceTrackingAudioConfigV2):
        if os.environ.get("MUZIQ_ALLOW_LEGACY_FFT_FRONTEND") != "1":
            raise RuntimeError(
                "FFT frontend is disabled. Use LEAF audio-context extraction; set "
                "MUZIQ_ALLOW_LEGACY_FFT_FRONTEND=1 only for legacy compatibility tests."
            )
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
        phase_offset_samples: int = 0,
    ) -> np.ndarray:
        n_frames = self._frame_count(audio)
        if n_frames == 0 or frame_count <= 0:
            return np.zeros((0, self.config.bands), dtype=np.float32)
        phase_offset_samples = int(
            np.clip(phase_offset_samples, 0, max(0, self.config.hop - 1))
        )
        end_frame = min(max(0, int(end_frame)), n_frames - 1)
        output_start = max(0, end_frame - frame_count + 1)
        normalize_start = max(0, output_start - max(0, peak_warmup_frames))
        raw_bands = self._raw_bands(
            audio,
            normalize_start,
            end_frame - normalize_start + 1,
            phase_offset_samples=phase_offset_samples,
        )
        normalized = self._normalize_bands(raw_bands)
        return normalized[output_start - normalize_start :]

    def _raw_bands(
        self,
        audio: np.ndarray,
        start_frame: int,
        frame_count: int,
        phase_offset_samples: int = 0,
    ) -> np.ndarray:
        n_frames = self._frame_count(audio)
        phase_offset_samples = int(
            np.clip(phase_offset_samples, 0, max(0, self.config.hop - 1))
        )
        target_samples = n_frames * self.config.hop + phase_offset_samples
        tail = target_samples - len(audio)
        padded = np.pad(
            audio.astype(np.float32, copy=False),
            (self.config.win - self.config.hop, max(0, tail)),
        )
        window_start = start_frame * self.config.hop + phase_offset_samples
        window_stop = (start_frame + frame_count) * self.config.hop + phase_offset_samples
        windows = np.lib.stride_tricks.sliding_window_view(padded, self.config.win)[
            window_start:window_stop:self.config.hop
        ]
        mag = np.abs(np.fft.rfft(windows * self._window, axis=1))
        return (mag @ self._fold.T).astype(np.float32)

    def _normalize_bands(self, raw_bands: np.ndarray) -> np.ndarray:
        log_bands = np.log1p(raw_bands * self.config.log_feature_scale).astype(
            np.float32
        )
        frames = np.zeros_like(log_bands, dtype=np.float32)
        peak = 1e-6
        for frame_idx, bands in enumerate(log_bands):
            peak = max(float(np.max(bands)), peak * 0.9996)
            frames[frame_idx] = np.clip(bands / max(peak, 1e-6), 0.0, 1.0)
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

    def sample_family(
        self,
        split: SplitName,
        rng: np.random.Generator,
        family_boosts: dict[str, float] | None = None,
    ) -> str:
        families = sorted(
            family for family, notes in self._by_split_family[split].items() if notes
        )
        if not families:
            raise ValueError(f"No NSynth families cached for split {split!r}")
        if family_boosts:
            weights = np.asarray(
                [max(0.0, family_boosts.get(family, 1.0)) for family in families],
                dtype=np.float64,
            )
            total = float(weights.sum())
            if total > 0.0:
                return str(rng.choice(families, p=weights / total))
        return families[int(rng.integers(len(families)))]

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

    def random_note(
        self,
        split: SplitName,
        rng: np.random.Generator,
        family: str | None = None,
        weight_fn=None,
    ) -> NsynthNoteV2:
        notes = (
            self._by_split_family[split].get(family, [])
            if family is not None
            else self.index.notes(split)
        )
        if not notes:
            raise ValueError(f"No NSynth notes cached for split {split!r}")
        if weight_fn is not None:
            weights = np.asarray([max(0.0, float(weight_fn(note))) for note in notes])
            total = float(weights.sum())
            if total > 0.0:
                return notes[int(rng.choice(len(notes), p=weights / total))]
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
        self.extractor: AudioFrameExtractorV2 | None = None
        self.families = FamilyVocabularyV2()
        self.positive_family_boosts = self._parse_weight_spec(
            config.positive_family_boosts
        )
        self.positive_quality_boosts = self._parse_weight_spec(
            config.positive_quality_boosts
        )
        self.positive_source_boosts = self._parse_weight_spec(
            config.positive_source_boosts
        )
        self.positive_velocity_boosts = self._parse_weight_spec(
            config.positive_velocity_boosts
        )
        self.censor_same_instrument_overlap_onset_families = {
            family.strip()
            for family in config.censor_same_instrument_overlap_onset_families.split(",")
            if family.strip()
        }
        self.hard_context_mix = self._parse_weight_spec(config.hard_context_mix)
        self.hard_context_positive_families = self._parse_list_spec(
            config.hard_context_positive_families
        )
        self.hard_context_negative_families = self._parse_list_spec(
            config.hard_context_negative_families
        )

    @staticmethod
    def _parse_weight_spec(spec: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        for item in str(spec or "").split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, value = item.split("=", 1)
            elif ":" in item:
                key, value = item.split(":", 1)
            else:
                key, value = item, "1.0"
            try:
                weights[key.strip()] = max(0.0, float(value))
            except ValueError:
                continue
        return weights

    @staticmethod
    def _parse_list_spec(spec: str) -> set[str]:
        return {item.strip() for item in str(spec or "").split(",") if item.strip()}

    def render(
        self, stage: str, split: SplitName, seed: int
    ) -> RenderedSourceTrackingExampleV2:
        audio, events = self._render_audio_events(stage, split, seed)
        frames = self._legacy_extractor().extract(audio)
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
        phase_offset_samples: int = 0,
    ) -> dict[str, np.ndarray]:
        audio, events = self._render_audio_events(stage, split, seed)
        frame_idx = self._sample_training_frame_from_events(
            events,
            len(audio),
            seed,
            context_frame_count=frame_count,
            use_hard_negatives=split == "train",
        )
        target = self._label_at_frame(events, len(audio), frame_idx)
        context = self._label_context_for_frame(events, len(audio), frame_idx, frame_count)
        hard_context = self._hard_context_masks_for_frame(
            events,
            len(audio),
            frame_idx,
            frame_count,
        )
        return {
            "frames": self._legacy_extractor().extract_context(
                audio,
                end_frame=frame_idx,
                frame_count=frame_count,
                peak_warmup_frames=peak_warmup_frames,
                phase_offset_samples=phase_offset_samples,
            ),
            "frame_idx": np.asarray(frame_idx, dtype=np.int64),
            **context,
            **hard_context,
            **target,
        }

    def _legacy_extractor(self) -> AudioFrameExtractorV2:
        if self.extractor is None:
            self.extractor = AudioFrameExtractorV2(self.config)
        return self.extractor

    def render_training_audio_slice(
        self,
        stage: str,
        split: SplitName,
        seed: int,
        frame_count: int,
        peak_warmup_frames: int,
        phase_offset_samples: int = 0,
    ) -> dict[str, np.ndarray]:
        audio, events = self._render_audio_events(stage, split, seed)
        frame_idx = self._sample_training_frame_from_events(
            events,
            len(audio),
            seed,
            context_frame_count=frame_count,
            use_hard_negatives=split == "train",
        )
        target = self._label_at_frame(events, len(audio), frame_idx)
        context = self._label_context_for_frame(events, len(audio), frame_idx, frame_count)
        hard_context = self._hard_context_masks_for_frame(
            events,
            len(audio),
            frame_idx,
            frame_count,
        )
        return {
            "audio_context": self._audio_context_for_frame(
                audio,
                end_frame=frame_idx,
                frame_count=frame_count,
                peak_warmup_frames=peak_warmup_frames,
                phase_offset_samples=phase_offset_samples,
            ),
            "frame_idx": np.asarray(frame_idx, dtype=np.int64),
            **context,
            **hard_context,
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
        audio = self._augment_mixture(audio, rng)
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio = (audio / peak * 0.8).astype(np.float32)
        return audio, events

    def _augment_mixture(
        self, audio: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        augmented = audio.astype(np.float32, copy=True)
        if rng.random() < 0.8:
            augmented = self._apply_spectral_tilt(augmented, rng)
        if rng.random() < 0.5:
            augmented = self._apply_short_reverb(augmented, rng)
        peak = float(np.max(np.abs(augmented)))
        if peak > 1e-6 and rng.random() < 0.7:
            noise_db = float(rng.uniform(-42.0, -26.0))
            noise_scale = peak * (10.0 ** (noise_db / 20.0))
            augmented = augmented + rng.normal(
                0.0, noise_scale, size=augmented.shape
            ).astype(np.float32)
        return augmented.astype(np.float32)

    @staticmethod
    def _apply_spectral_tilt(
        audio: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        emphasis = float(rng.uniform(-0.35, 0.35))
        shifted = np.concatenate((audio[:1], audio[:-1]))
        tilted = audio + emphasis * (audio - shifted)
        return np.clip(tilted, -2.0, 2.0).astype(np.float32)

    def _apply_short_reverb(
        self, audio: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        delay = int(rng.integers(self.config.sample_rate // 80, self.config.sample_rate // 16))
        decay = float(rng.uniform(0.08, 0.28))
        wet = audio.astype(np.float32, copy=True)
        if delay < len(wet):
            wet[delay:] += audio[:-delay] * decay
        return wet.astype(np.float32)

    def _render_single_note(
        self,
        split: SplitName,
        rng: np.random.Generator,
        audio: np.ndarray,
        events: list[SourceEventLabelV2],
    ) -> None:
        family = self.note_store.sample_family(split, rng, self.positive_family_boosts)
        note = self.note_store.random_note(
            split,
            rng,
            family=family,
            weight_fn=self._positive_note_weight,
        )
        start_s = float(rng.uniform(0.25, 1.25))
        self._mix_note(audio, note, 0, start_s, events)

    def _positive_note_weight(self, note: NsynthNoteV2) -> float:
        weight = self.positive_family_boosts.get(note.family, 1.0)
        weight *= self.positive_source_boosts.get(note.source, 1.0)
        weight *= self.positive_velocity_boosts.get(str(note.velocity), 1.0)
        if note.qualities:
            weight *= max(
                self.positive_quality_boosts.get(quality, 1.0)
                for quality in note.qualities
            )
        else:
            weight *= self.positive_quality_boosts.get("<none>", 1.0)
        return weight

    def _render_generated_melody(
        self,
        split: SplitName,
        rng: np.random.Generator,
        audio: np.ndarray,
        events: list[SourceEventLabelV2],
        source_count: int,
    ) -> None:
        family_order = [
            self.note_store.sample_family(split, rng)
            for _ in range(min(source_count, self.config.max_sources))
        ]
        for source_id in range(min(source_count, self.config.max_sources)):
            notes = self.note_store.sample_instrument_notes(
                split, rng, family=family_order[source_id]
            )
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
                instrument_str=note.instrument_str,
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
        onset_delta = np.zeros(shape, dtype=np.float32)
        offset_delta = np.zeros(shape, dtype=np.float32)
        onset_timing_mask = np.zeros(shape, dtype=np.float32)
        onset_nearby_mask = np.zeros(shape, dtype=np.float32)
        offset_timing_mask = np.zeros(shape, dtype=np.float32)
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
            onset_censored = self._censor_event_onset(
                event,
                events,
                n_frames,
            )
            active[start:end, event.source_id] = 1.0
            family[start:end, event.source_id] = event.family_index
            source_id[start:end, event.source_id] = event.source_id
            onset_nearby_start = max(0, start - self.config.onset_shoulder_radius_frames)
            onset_nearby_end = min(
                n_frames,
                start + self.config.onset_shoulder_radius_frames + 1,
            )
            offset_center = end - 1
            offset_start = max(0, offset_center - self.config.offset_label_radius_frames)
            offset_end = min(
                n_frames, offset_center + self.config.offset_label_radius_frames + 1
            )
            onset_nearby_frames = np.arange(
                onset_nearby_start,
                onset_nearby_end,
                dtype=np.float32,
            )
            if not onset_censored:
                onset[onset_nearby_start:onset_nearby_end, event.source_id] = 1.0
                onset_delta[onset_nearby_start:onset_nearby_end, event.source_id] = (
                    onset_nearby_frames - start
                )
                onset_timing_mask[
                    onset_nearby_start:onset_nearby_end, event.source_id
                ] = 1.0
                onset_nearby_mask[
                    onset_nearby_start:onset_nearby_end, event.source_id
                ] = 1.0
            offset[offset_start:offset_end, event.source_id] = 1.0
            offset_frames = np.arange(offset_start, offset_end, dtype=np.float32)
            offset_delta[offset_start:offset_end, event.source_id] = (
                offset_frames - offset_center
            )
            offset_timing_mask[offset_start:offset_end, event.source_id] = 1.0
        return SourceLabelFramesV2(
            active=active,
            onset=onset,
            offset=offset,
            onset_delta=onset_delta,
            offset_delta=offset_delta,
            onset_timing_mask=onset_timing_mask,
            onset_nearby_mask=onset_nearby_mask,
            offset_timing_mask=offset_timing_mask,
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

    def _sample_training_frame_from_events(
        self,
        events: list[SourceEventLabelV2],
        n_samples: int,
        seed: int,
        *,
        context_frame_count: int | None = None,
        use_hard_negatives: bool = True,
    ) -> int:
        n_frames = int(np.ceil(n_samples / self.config.hop))
        span_rows = [
            (
                event,
                *self._event_frame_span(event, n_frames),
            )
            for event in events
            if event.source_id < self.config.max_sources
        ]
        span_rows = [
            (event, start, end) for event, start, end in span_rows if end > start
        ]
        if not span_rows:
            return n_frames - 1
        rng = np.random.default_rng(seed + 9_973)
        if (
            context_frame_count is not None
            and context_frame_count > 0
            and use_hard_negatives
        ):
            hard_frame = self._sample_hard_context_frame(
                span_rows,
                n_frames,
                rng,
            )
            if hard_frame is not None:
                return hard_frame
        event, start, end = span_rows[int(rng.integers(len(span_rows)))]
        probe = float(rng.random())
        onset_context_prob = min(
            1.0,
            max(0.0, float(self.config.onset_context_sample_prob)),
        )
        if (
            context_frame_count is not None
            and context_frame_count > 0
            and probe < onset_context_prob
        ):
            onset_rows = [
                row
                for row in span_rows
                if not self._censor_event_onset(row[0], events, n_frames)
            ]
            if onset_rows:
                _, start, end = onset_rows[int(rng.integers(len(onset_rows)))]
            hi = min(n_frames, start + int(context_frame_count))
            return int(rng.integers(start, max(start + 1, hi)))
        negative_radius = self.config.boundary_negative_radius_frames
        hard_negative_prob = (
            min(1.0, max(0.0, float(self.config.onset_hard_negative_sample_prob)))
            if use_hard_negatives
            else 0.0
        )
        if probe < hard_negative_prob:
            onset_gap = self.config.onset_shoulder_radius_frames + 1
            if rng.random() < 0.5:
                lo = max(0, start - negative_radius)
                hi = max(lo + 1, start - self.config.onset_shoulder_radius_frames)
                return int(rng.integers(lo, hi))
            lo = min(n_frames - 1, start + onset_gap)
            hi = min(n_frames, end, start + negative_radius + 1)
            if hi > lo:
                return int(rng.integers(lo, hi))
            lo = max(0, start - negative_radius)
            hi = max(lo + 1, start - self.config.onset_shoulder_radius_frames)
            return int(rng.integers(lo, hi))
        if hard_negative_prob > 0.0:
            probe = (probe - hard_negative_prob) / (1.0 - hard_negative_prob)
        if probe < 0.15:
            lo = max(0, start - negative_radius)
            hi = max(lo + 1, start - self.config.onset_shoulder_radius_frames)
            return int(rng.integers(lo, hi))
        if probe < 0.35:
            return int(start)
        if probe < 0.65:
            return int(rng.integers(start, end))
        if probe < 0.85:
            return int(end - 1)
        lo = min(n_frames - 1, end + self.config.offset_label_radius_frames + 1)
        hi = min(n_frames, end + negative_radius)
        if hi <= lo:
            return int(end - 1)
        return int(rng.integers(lo, hi))

    def _sample_hard_context_frame(
        self,
        span_rows: list[tuple[SourceEventLabelV2, int, int]],
        n_frames: int,
        rng: np.random.Generator,
    ) -> int | None:
        sample_prob = min(1.0, max(0.0, float(self.config.hard_context_sample_prob)))
        if sample_prob <= 0.0 or float(rng.random()) >= sample_prob:
            return None
        mode = self._sample_hard_context_mode(rng)
        if mode == "ordinary":
            return None
        candidates = self._hard_context_candidates(span_rows, n_frames)
        selected = candidates.get(mode, [])
        if not selected:
            return None
        return int(selected[int(rng.integers(len(selected)))])

    def _sample_hard_context_mode(self, rng: np.random.Generator) -> str:
        weights = {
            "reattack": self.hard_context_mix.get("reattack", 0.0),
            "post_onset": self.hard_context_mix.get("post_onset", 0.0),
            "pre_onset": self.hard_context_mix.get("pre_onset", 0.0),
            "ordinary": self.hard_context_mix.get("ordinary", 0.0),
        }
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 0.0:
            return "ordinary"
        probe = float(rng.random()) * total
        acc = 0.0
        for mode, weight in weights.items():
            acc += max(0.0, weight)
            if probe <= acc:
                return mode
        return "ordinary"

    def _hard_context_candidates(
        self,
        span_rows: list[tuple[SourceEventLabelV2, int, int]],
        n_frames: int,
    ) -> dict[str, list[int]]:
        positive_families = self.hard_context_positive_families
        negative_families = self.hard_context_negative_families
        candidates: dict[str, list[int]] = {
            "reattack": [],
            "post_onset": [],
            "pre_onset": [],
        }
        for event, start, _end in span_rows:
            if self._censor_event_onset(event, [row[0] for row in span_rows], n_frames):
                continue
            if not positive_families or event.family in positive_families:
                previous = self._previous_same_instrument_event(event, span_rows, start)
                if previous is not None:
                    prev_start, prev_end = previous
                    spacing_ms = (start - prev_start) * self._hop_ms()
                    remaining_ms = (prev_end - start) * self._hop_ms()
                    if (
                        self.config.hard_context_reattack_min_ms
                        <= spacing_ms
                        <= self.config.hard_context_reattack_max_ms
                        and remaining_ms >= self.config.hard_context_min_remaining_ms
                    ):
                        candidates["reattack"].append(start)
            if negative_families and event.family not in negative_families:
                continue
            post_lo, post_hi = self._ms_window_after_onset(
                start,
                self.config.hard_context_post_onset_min_ms,
                self.config.hard_context_post_onset_max_ms,
                n_frames,
            )
            candidates["post_onset"].extend(range(post_lo, post_hi))
            pre_lo, pre_hi = self._ms_window_before_onset(
                start,
                self.config.hard_context_pre_onset_min_ms,
                self.config.hard_context_pre_onset_max_ms,
            )
            candidates["pre_onset"].extend(range(pre_lo, pre_hi))
        return candidates

    def _previous_same_instrument_event(
        self,
        event: SourceEventLabelV2,
        span_rows: list[tuple[SourceEventLabelV2, int, int]],
        start: int,
    ) -> tuple[int, int] | None:
        previous = [
            (other_start, other_end)
            for other, other_start, other_end in span_rows
            if other is not event
            and other.source_id == event.source_id
            and other.instrument_str == event.instrument_str
            and other_start < start
        ]
        if not previous:
            return None
        return max(previous, key=lambda row: row[0])

    def _ms_window_after_onset(
        self,
        start: int,
        min_ms: float,
        max_ms: float,
        n_frames: int,
    ) -> tuple[int, int]:
        shoulder_gap = self.config.onset_shoulder_radius_frames + 1
        lo = start + max(shoulder_gap, int(np.ceil(min_ms / self._hop_ms())))
        hi = start + int(np.floor(max_ms / self._hop_ms())) + 1
        return max(0, min(n_frames, lo)), max(0, min(n_frames, hi))

    def _ms_window_before_onset(
        self,
        start: int,
        min_ms: float,
        max_ms: float,
    ) -> tuple[int, int]:
        shoulder_gap = self.config.onset_shoulder_radius_frames + 1
        lo = start - int(np.floor(max_ms / self._hop_ms()))
        hi = start - max(shoulder_gap, int(np.ceil(min_ms / self._hop_ms()))) + 1
        return max(0, lo), max(0, hi)

    @staticmethod
    def _hop_ms() -> float:
        return (
            SourceTrackingAudioConfigV2.hop
            / SourceTrackingAudioConfigV2.sample_rate
            * 1000.0
        )

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
        onset_delta = np.zeros(self.config.max_sources, dtype=np.float32)
        offset_delta = np.zeros(self.config.max_sources, dtype=np.float32)
        onset_timing_mask = np.zeros(self.config.max_sources, dtype=np.float32)
        onset_nearby_mask = np.zeros(self.config.max_sources, dtype=np.float32)
        offset_timing_mask = np.zeros(self.config.max_sources, dtype=np.float32)
        family = np.full(self.config.max_sources, -1, dtype=np.int32)
        for event in events:
            if event.source_id >= self.config.max_sources:
                continue
            start, end = self._event_frame_span(event, n_frames)
            if start <= frame_idx < end:
                activity[event.source_id] = 1.0
                family[event.source_id] = event.family_index
            onset_censored = self._censor_event_onset(event, events, n_frames)
            onset_frame_delta = frame_idx - start
            if (
                not onset_censored
                and abs(onset_frame_delta) <= self.config.onset_shoulder_radius_frames
            ):
                onset_delta[event.source_id] = float(onset_frame_delta)
                onset_nearby_mask[event.source_id] = 1.0
                onset[event.source_id] = 1.0
                onset_timing_mask[event.source_id] = 1.0
            offset_center = end - 1
            if abs(frame_idx - offset_center) <= self.config.offset_label_radius_frames:
                offset[event.source_id] = 1.0
                offset_delta[event.source_id] = float(frame_idx - offset_center)
                offset_timing_mask[event.source_id] = 1.0
        return {
            "activity": activity,
            "family": family,
            "onset": onset,
            "offset": offset,
            "onset_delta": onset_delta,
            "offset_delta": offset_delta,
            "onset_timing_mask": onset_timing_mask,
            "onset_nearby_mask": onset_nearby_mask,
            "offset_timing_mask": offset_timing_mask,
        }

    def _label_context_for_frame(
        self,
        events: list[SourceEventLabelV2],
        n_samples: int,
        frame_idx: int,
        frame_count: int,
    ) -> dict[str, np.ndarray]:
        labels = self._labels_from_events(events, n_samples)
        start = max(0, frame_idx - frame_count + 1)
        stop = frame_idx + 1
        pad = frame_count - (stop - start)
        shape = (frame_count, self.config.max_sources)

        def window(array: np.ndarray, fill: float = 0.0) -> np.ndarray:
            out = np.full(shape, fill, dtype=array.dtype)
            out[pad:] = array[start:stop]
            return out

        active = window(labels.active)
        onset = window(labels.onset)
        offset = window(labels.offset)
        onset_delta = window(labels.onset_delta)
        offset_delta = window(labels.offset_delta)
        onset_timing_mask = window(labels.onset_timing_mask)
        onset_nearby_mask = window(labels.onset_nearby_mask)
        offset_timing_mask = window(labels.offset_timing_mask)
        family = window(labels.family, fill=-1)
        event_state = self._event_state_context(active, onset, offset)
        return {
            "context_activity": active,
            "context_family": family,
            "context_onset": onset,
            "context_offset": offset,
            "context_onset_delta": onset_delta,
            "context_offset_delta": offset_delta,
            "context_onset_timing_mask": onset_timing_mask,
            "context_onset_nearby_mask": onset_nearby_mask,
            "context_offset_timing_mask": offset_timing_mask,
            "event_state": event_state,
        }

    def _hard_context_masks_for_frame(
        self,
        events: list[SourceEventLabelV2],
        n_samples: int,
        frame_idx: int,
        frame_count: int,
    ) -> dict[str, np.ndarray]:
        n_frames = int(np.ceil(n_samples / self.config.hop))
        start = max(0, frame_idx - frame_count + 1)
        stop = frame_idx + 1
        pad = frame_count - (stop - start)
        shape = (frame_count, self.config.max_sources)
        positive = np.zeros(shape, dtype=np.float32)
        post_negative = np.zeros(shape, dtype=np.float32)
        pre_negative = np.zeros(shape, dtype=np.float32)
        far_negative = np.zeros(shape, dtype=np.float32)
        labels = self._labels_from_events(events, n_samples)
        onset_nearby = np.zeros(shape, dtype=np.float32)
        onset_nearby[pad:] = labels.onset_nearby_mask[start:stop]
        span_rows = [
            (event, *self._event_frame_span(event, n_frames))
            for event in events
            if event.source_id < self.config.max_sources
        ]
        span_rows = [
            (event, event_start, event_end)
            for event, event_start, event_end in span_rows
            if event_end > event_start
        ]
        all_events = [row[0] for row in span_rows]
        for event, event_start, event_end in span_rows:
            if self._censor_event_onset(event, all_events, n_frames):
                continue
            previous = self._previous_same_instrument_event(
                event,
                span_rows,
                event_start,
            )
            if previous is not None:
                prev_start, prev_end = previous
                spacing_ms = (event_start - prev_start) * self._hop_ms()
                remaining_ms = (prev_end - event_start) * self._hop_ms()
                if (
                    (
                        not self.hard_context_positive_families
                        or event.family in self.hard_context_positive_families
                    )
                    and self.config.hard_context_reattack_min_ms
                    <= spacing_ms
                    <= self.config.hard_context_reattack_max_ms
                    and remaining_ms >= self.config.hard_context_min_remaining_ms
                ):
                    self._set_context_frame(positive, start, pad, event_start, event.source_id)
            if (
                self.hard_context_negative_families
                and event.family not in self.hard_context_negative_families
            ):
                continue
            post_lo, post_hi = self._ms_window_after_onset(
                event_start,
                self.config.hard_context_post_onset_min_ms,
                self.config.hard_context_post_onset_max_ms,
                n_frames,
            )
            self._set_context_range(
                post_negative,
                start,
                pad,
                post_lo,
                post_hi,
                event.source_id,
            )
            pre_lo, pre_hi = self._ms_window_before_onset(
                event_start,
                self.config.hard_context_pre_onset_min_ms,
                self.config.hard_context_pre_onset_max_ms,
            )
            self._set_context_range(
                pre_negative,
                start,
                pad,
                pre_lo,
                pre_hi,
                event.source_id,
            )
            sustain_lo = min(
                n_frames,
                event_start + self.config.onset_shoulder_radius_frames + 1,
            )
            sustain_hi = max(sustain_lo, min(n_frames, event_end))
            self._set_context_range(
                far_negative,
                start,
                pad,
                sustain_lo,
                sustain_hi,
                event.source_id,
            )
        far_negative = np.clip(
            far_negative - post_negative - pre_negative - onset_nearby,
            0.0,
            1.0,
        )
        return {
            "hard_context_positive_mask": positive,
            "hard_context_post_onset_negative_mask": post_negative,
            "hard_context_pre_onset_negative_mask": pre_negative,
            "hard_context_far_negative_mask": far_negative,
        }

    @staticmethod
    def _set_context_frame(
        mask: np.ndarray,
        context_start: int,
        context_pad: int,
        global_frame: int,
        source_id: int,
    ) -> None:
        local = context_pad + global_frame - context_start
        if 0 <= local < mask.shape[0]:
            mask[local, source_id] = 1.0

    @staticmethod
    def _set_context_range(
        mask: np.ndarray,
        context_start: int,
        context_pad: int,
        global_start: int,
        global_stop: int,
        source_id: int,
    ) -> None:
        local_start = max(0, context_pad + global_start - context_start)
        local_stop = min(mask.shape[0], context_pad + global_stop - context_start)
        if local_stop > local_start:
            mask[local_start:local_stop, source_id] = 1.0

    @staticmethod
    def _event_state_context(
        active: np.ndarray,
        onset: np.ndarray,
        offset: np.ndarray,
    ) -> np.ndarray:
        frame_count, source_count = active.shape
        state = np.zeros((frame_count, source_count, 5), dtype=np.float32)
        previous_onset = np.vstack(
            [np.zeros((1, source_count), dtype=np.float32), onset[:-1]]
        )
        previous_offset = np.vstack(
            [np.zeros((1, source_count), dtype=np.float32), offset[:-1]]
        )
        previous_active = np.vstack(
            [np.zeros((1, source_count), dtype=np.float32), active[:-1]]
        )
        state[..., 0] = previous_onset
        state[..., 1] = previous_offset
        state[..., 2] = previous_active
        for source_idx in range(source_count):
            since_onset = frame_count
            since_offset = frame_count
            for frame_idx in range(frame_count):
                if previous_onset[frame_idx, source_idx] > 0.5:
                    since_onset = 0
                else:
                    since_onset += 1
                if previous_offset[frame_idx, source_idx] > 0.5:
                    since_offset = 0
                else:
                    since_offset += 1
                state[frame_idx, source_idx, 3] = min(since_onset, frame_count) / frame_count
                state[frame_idx, source_idx, 4] = min(since_offset, frame_count) / frame_count
        return state

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

    def _censor_event_onset(
        self,
        event: SourceEventLabelV2,
        events: list[SourceEventLabelV2],
        n_frames: int,
    ) -> bool:
        if event.family not in self.censor_same_instrument_overlap_onset_families:
            return False
        if not event.instrument_str:
            return False
        start, _ = self._event_frame_span(event, n_frames)
        for other in events:
            if other is event:
                continue
            if other.source_id != event.source_id:
                continue
            if other.instrument_str != event.instrument_str:
                continue
            other_start, other_end = self._event_frame_span(other, n_frames)
            if other_start < start < other_end:
                return True
        return False

    def _audio_context_for_frame(
        self,
        audio: np.ndarray,
        end_frame: int,
        frame_count: int,
        peak_warmup_frames: int,
        phase_offset_samples: int = 0,
    ) -> np.ndarray:
        phase_offset_samples = int(
            np.clip(phase_offset_samples, 0, max(0, self.config.hop - 1))
        )
        raw_frame_count = max(1, frame_count + max(0, peak_warmup_frames))
        segment_len = self.config.win + (raw_frame_count - 1) * self.config.hop
        end_sample = (end_frame + 1) * self.config.hop + phase_offset_samples
        start_sample = end_sample - segment_len
        context = np.zeros(segment_len, dtype=np.float32)
        src_start = max(0, start_sample)
        src_end = min(len(audio), end_sample)
        if src_end <= src_start:
            return context
        dst_start = src_start - start_sample
        context[dst_start : dst_start + src_end - src_start] = audio[src_start:src_end]
        return context
