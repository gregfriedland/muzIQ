from __future__ import annotations

import numpy as np
from conftest import TinyCorpusBuilderV2

from muziq_nn.datasets.midi import MidiIndexV2
from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    MidiScheduleStoreV2,
    NsynthNoteStoreV2,
    SourceTrackingAudioConfigV2,
    SourceTrackingRendererV2,
)
from muziq_nn.training.train import TrainingBatchBuilderV2


class TestRendererV2:
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

        full = TrainingBatchBuilderV2.slice_from_example(
            renderer.render("single_instrument_melody", "train", seed=5)
        )
        sliced = renderer.render_training_slice(
            "single_instrument_melody",
            "train",
            seed=5,
            frame_count=256,
            peak_warmup_frames=10_000,
        )

        assert sliced["frames"].shape == full["frames"].shape
        np.testing.assert_allclose(sliced["frames"], full["frames"], rtol=1e-6, atol=1e-6)
        assert (sliced["activity"] == full["activity"]).all()
        assert (sliced["family"] == full["family"]).all()
        assert (sliced["onset"] == full["onset"]).all()
        assert (sliced["offset"] == full["offset"]).all()
