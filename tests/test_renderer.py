from __future__ import annotations

import numpy as np
import torch
from conftest import TinyCorpusBuilderV2

from muziq_nn.datasets.midi import MidiIndexV2
from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    AudioFrameExtractorV2,
    FamilyVocabularyV2,
    MidiScheduleStoreV2,
    NsynthNoteStoreV2,
    SourceTrackingAudioConfigV2,
    SourceTrackingRendererV2,
)
from muziq_nn.datasets.schema import SourceEventLabelV2
from muziq_nn.training.train import TrainingBatchBuilderV2, TrainingRenderWorkerV2


class TestRendererV2:
    def test_band_normalization_preserves_spectral_shape(self):
        extractor = AudioFrameExtractorV2()
        raw_bands = np.ones((2, SourceTrackingAudioConfigV2.bands), dtype=np.float32)
        raw_bands[0, 3] = 10.0
        raw_bands[0, 7] = 1.0
        raw_bands[1, 3] = 5.0
        raw_bands[1, 7] = 1.0

        normalized = extractor._normalize_bands(raw_bands)

        assert normalized[0, 3] == 1.0
        assert 0.0 < normalized[0, 7] < normalized[0, 3]
        assert normalized[1, 7] < normalized[1, 3]

    def test_renderer_produces_frames_and_labels_without_disk_outputs(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )

        example = renderer.render("midi_complex", "train", seed=3)
        after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

        assert after == before
        assert example.audio.shape[0] == int(
            SourceTrackingAudioConfigV2.duration_s * SourceTrackingAudioConfigV2.sample_rate
        )
        assert example.frames.shape[1] == SourceTrackingAudioConfigV2.bands
        assert example.labels.active.shape[1] == SourceTrackingAudioConfigV2.max_sources
        assert example.labels.active.sum() > 0

    def test_onset_label_uses_shoulder_radius(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        config = type(
            "TestAudioConfig",
            (SourceTrackingAudioConfigV2,),
            {
                "onset_shoulder_radius_frames": 5,
                "boundary_negative_radius_frames": 4,
            },
        )
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
            config=config,
        )
        start_frame = 10
        labels = renderer._labels_from_events(
            [
                SourceEventLabelV2(
                    source_id=0,
                    family="guitar",
                    family_index=3,
                    start_s=start_frame * config.hop / config.sample_rate,
                    end_s=(start_frame + 20) * config.hop / config.sample_rate,
                )
            ],
            n_samples=40 * config.hop,
        )

        assert labels.onset[start_frame - 5 : start_frame + 6, 0].tolist() == [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        assert labels.onset[start_frame - 6, 0] == 0.0
        assert labels.onset[start_frame + 6, 0] == 0.0
        assert labels.onset_nearby_mask[start_frame - 5, 0] == 1.0
        assert labels.onset_nearby_mask[start_frame + 5, 0] == 1.0
        assert labels.onset_nearby_mask[start_frame + 6, 0] == 0.0
        assert labels.onset_delta[start_frame - 5, 0] == -5.0
        assert labels.onset_delta[start_frame + 5, 0] == 5.0

    def test_same_instrument_overlap_onset_can_be_censored(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        config = type(
            "CensorOverlapAudioConfig",
            (SourceTrackingAudioConfigV2,),
            {
                "onset_shoulder_radius_frames": 2,
                "censor_same_instrument_overlap_onset_families": "vocal,flute",
            },
        )
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
            config=config,
        )
        first_start = 10
        second_start = 20
        events = [
            SourceEventLabelV2(
                source_id=0,
                family="vocal",
                family_index=10,
                instrument_str="vocal_acoustic_001",
                start_s=first_start * config.hop / config.sample_rate,
                end_s=50 * config.hop / config.sample_rate,
            ),
            SourceEventLabelV2(
                source_id=0,
                family="vocal",
                family_index=10,
                instrument_str="vocal_acoustic_001",
                start_s=second_start * config.hop / config.sample_rate,
                end_s=60 * config.hop / config.sample_rate,
            ),
        ]
        labels = renderer._labels_from_events(
            events,
            n_samples=80 * config.hop,
        )
        frame_label = renderer._label_at_frame(
            events,
            n_samples=80 * config.hop,
            frame_idx=second_start,
        )

        assert labels.active[second_start, 0] == 1.0
        assert labels.onset[first_start - 2 : first_start + 3, 0].sum() == 5.0
        assert labels.onset[second_start - 2 : second_start + 3, 0].sum() == 0.0
        assert (
            labels.onset_nearby_mask[second_start - 2 : second_start + 3, 0].sum()
            == 0.0
        )
        assert frame_label["activity"][0] == 1.0
        assert frame_label["onset"][0] == 0.0
        assert frame_label["onset_nearby_mask"][0] == 0.0

    def test_different_instrument_overlap_onset_is_not_censored(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        config = type(
            "CensorOverlapAudioConfig",
            (SourceTrackingAudioConfigV2,),
            {
                "onset_shoulder_radius_frames": 2,
                "censor_same_instrument_overlap_onset_families": "vocal,flute",
            },
        )
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
            config=config,
        )
        second_start = 20
        events = [
            SourceEventLabelV2(
                source_id=0,
                family="vocal",
                family_index=10,
                instrument_str="vocal_acoustic_001",
                start_s=10 * config.hop / config.sample_rate,
                end_s=50 * config.hop / config.sample_rate,
            ),
            SourceEventLabelV2(
                source_id=0,
                family="vocal",
                family_index=10,
                instrument_str="vocal_acoustic_002",
                start_s=second_start * config.hop / config.sample_rate,
                end_s=60 * config.hop / config.sample_rate,
            ),
        ]
        labels = renderer._labels_from_events(
            events,
            n_samples=80 * config.hop,
        )
        frame_label = renderer._label_at_frame(
            events,
            n_samples=80 * config.hop,
            frame_idx=second_start,
        )

        assert labels.onset[second_start - 2 : second_start + 3, 0].sum() == 5.0
        assert frame_label["onset"][0] == 1.0

    def test_chords_within_one_midi_track_keep_one_source_id(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )

        example = renderer.render("midi_complex", "train", seed=4)

        active_sources = sorted(
            set(example.labels.source_id[example.labels.source_id >= 0].tolist())
        )
        assert active_sources
        assert min(active_sources) == 0

    def test_training_slice_matches_full_render_labels(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )

        full = renderer.render("single_instrument_melody", "train", seed=5)
        sliced = renderer.render_training_slice(
            "single_instrument_melody",
            "train",
            seed=5,
            frame_count=256,
            peak_warmup_frames=10_000,
        )
        frame_idx = int(sliced["frame_idx"])
        full_slice = {
            "frames": full.frames[max(0, frame_idx - 255) : frame_idx + 1],
            "activity": full.labels.active[frame_idx],
            "family": full.labels.family[frame_idx],
            "onset": full.labels.onset[frame_idx],
            "offset": full.labels.offset[frame_idx],
        }

        assert sliced["frames"].shape == full_slice["frames"].shape
        np.testing.assert_allclose(
            sliced["frames"], full_slice["frames"], rtol=1e-6, atol=1e-6
        )
        assert (sliced["activity"] == full_slice["activity"]).all()
        assert (sliced["family"] == full_slice["family"]).all()
        assert (sliced["onset"] == full_slice["onset"]).all()
        assert (sliced["offset"] == full_slice["offset"]).all()

    def test_onset_context_sampling_keeps_onset_in_sequence_context(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        config = type(
            "OnsetContextAudioConfig",
            (SourceTrackingAudioConfigV2,),
            {"onset_context_sample_prob": 1.0},
        )
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
            config=config,
        )

        for seed in range(20):
            sliced = renderer.render_training_slice(
                "single_note_all",
                "train",
                seed=seed,
                frame_count=32,
                peak_warmup_frames=128,
            )

            assert sliced["context_onset_nearby_mask"].sum() > 0

    def test_audio_training_slice_matches_cpu_frame_slice_after_batching(
        self, tmp_path
    ):
        TinyCorpusBuilderV2(tmp_path).build()
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )
        builder = TrainingBatchBuilderV2(torch.device("cpu"), frame_count=256)

        frame_slice = renderer.render_training_slice(
            "single_instrument_melody",
            "train",
            seed=6,
            frame_count=256,
            peak_warmup_frames=512,
        )
        audio_slice = renderer.render_training_audio_slice(
            "single_instrument_melody",
            "train",
            seed=6,
            frame_count=256,
            peak_warmup_frames=512,
        )

        frame_batch = builder.build_slices([frame_slice])
        audio_batch = builder.build_slices([audio_slice])

        torch.testing.assert_close(
            audio_batch["frames"],
            frame_batch["frames"],
            rtol=5e-4,
            atol=5e-4,
        )
        torch.testing.assert_close(audio_batch["activity"], frame_batch["activity"])
        torch.testing.assert_close(audio_batch["family"], frame_batch["family"])
        torch.testing.assert_close(audio_batch["onset"], frame_batch["onset"])
        torch.testing.assert_close(audio_batch["offset"], frame_batch["offset"])

    def test_audio_training_slice_honors_phase_offset_after_batching(
        self, tmp_path
    ):
        TinyCorpusBuilderV2(tmp_path).build()
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )
        builder = TrainingBatchBuilderV2(torch.device("cpu"), frame_count=256)
        phase_offset_samples = SourceTrackingAudioConfigV2.hop // 2

        frame_slice = renderer.render_training_slice(
            "single_instrument_melody",
            "train",
            seed=6,
            frame_count=256,
            peak_warmup_frames=512,
            phase_offset_samples=phase_offset_samples,
        )
        audio_slice = renderer.render_training_audio_slice(
            "single_instrument_melody",
            "train",
            seed=6,
            frame_count=256,
            peak_warmup_frames=512,
            phase_offset_samples=phase_offset_samples,
        )

        frame_batch = builder.build_slices([frame_slice])
        audio_batch = builder.build_slices([audio_slice])

        torch.testing.assert_close(
            audio_batch["frames"],
            frame_batch["frames"],
            rtol=5e-4,
            atol=5e-4,
        )

    def test_audio_training_worker_honors_phase_offset_after_batching(
        self, tmp_path
    ):
        TinyCorpusBuilderV2(tmp_path).build()
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )
        builder = TrainingBatchBuilderV2(torch.device("cpu"), frame_count=256)
        phase_offset_samples = SourceTrackingAudioConfigV2.hop // 2

        frame_slice = renderer.render_training_slice(
            "single_instrument_melody",
            "train",
            seed=6,
            frame_count=256,
            peak_warmup_frames=512,
            phase_offset_samples=phase_offset_samples,
        )
        audio_slice = TrainingRenderWorkerV2.render_slice(
            (
                str(tmp_path),
                "single_instrument_melody",
                "train",
                6,
                256,
                512,
                True,
                phase_offset_samples,
            )
        )

        frame_batch = builder.build_slices([frame_slice])
        audio_batch = builder.build_slices([audio_slice])

        torch.testing.assert_close(
            audio_batch["frames"],
            frame_batch["frames"],
            rtol=5e-4,
            atol=5e-4,
        )

    def test_training_slice_samples_boundary_frames(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )

        slices = [
            renderer.render_training_slice(
                "single_instrument_melody",
                "train",
                seed=seed,
                frame_count=256,
                peak_warmup_frames=512,
            )
            for seed in range(30)
        ]

        assert any(item["onset"].any() for item in slices)
        assert any(item["offset"].any() for item in slices)
        assert any(not item["activity"].any() for item in slices)

    def test_single_note_training_samples_all_available_families(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path).build()
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(NsynthIndexV2(tmp_path)),
            MidiScheduleStoreV2(MidiIndexV2(tmp_path)),
        )
        families = FamilyVocabularyV2()

        seen = set()
        for seed in range(96):
            item = renderer.render_training_slice(
                "single_note_all",
                "train",
                seed=seed,
                frame_count=256,
                peak_warmup_frames=512,
            )
            active = item["activity"] > 0.5
            if active.any():
                seen.add(families.families[int(item["family"][active][0])])

        assert seen == set(TinyCorpusBuilderV2.FAMILIES)
