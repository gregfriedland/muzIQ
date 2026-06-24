from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import TinyCorpusBuilderV2

from muziq_nn.datasets.render import SourceTrackingAudioConfigV2
from muziq_nn.models.frontend import LeafFrontendConfigV2
from muziq_nn.training.train import (
    CurriculumPlanV2,
    CurriculumStageV2,
    SourceTrackingTrainerV2,
    TrainingBatchBuilderV2,
    TrainingCliV2,
    TrainingConfigV2,
)


class TestTrainSmokeV2:
    def test_curriculum_scale_reduces_examples_per_epoch(self):
        stages = CurriculumPlanV2().scale_examples(0.1).stages

        assert [stage.train_examples_per_epoch for stage in stages] == [
            3_000,
            6_000,
            8_000,
            10_000,
            4_000,
        ]

    def test_epoch_multiplier_increases_epochs(self):
        stages = CurriculumPlanV2().scale_epochs(2).stages

        assert [stage.epochs for stage in stages] == [6, 6, 6, 6, 4]

    def test_epochs_per_stage_overrides_all_stage_epochs(self):
        stages = CurriculumPlanV2().scale_epochs(2).with_epochs_per_stage(50).stages

        assert [stage.epochs for stage in stages] == [50, 50, 50, 50, 50]

    def test_cosine_learning_rate_decay_reaches_min_fraction(self):
        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.config = TrainingConfigV2(
            learning_rate_scale=0.1,
            learning_rate_decay="cosine",
            learning_rate_min_fraction=0.1,
            learning_rate_decay_epochs=10,
        )
        stage = CurriculumStageV2("single_note_all", 30_000, 10, 1e-3)

        assert trainer._learning_rate_for_step(stage, 0, 0, 10) == pytest.approx(1e-4)
        assert trainer._learning_rate_for_step(stage, 9, 9, 10) == pytest.approx(1e-5)

    def test_midi_render_workers_override_only_midi_stage(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                render_workers=8,
                midi_render_workers=2,
            )
        )

        assert trainer._render_worker_count("simple_duo_trio") == 8
        assert trainer._render_worker_count("midi_complex") == 2

    def test_cli_parses_midi_render_workers(self):
        config = TrainingCliV2.parse(["--midi-render-workers", "4"])

        assert config.midi_render_workers == 4

    def test_stage_filter_selects_requested_stages(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                stage_filter="single_note_all,single_instrument_melody",
                device="cpu",
            )
        )
        stages = trainer._filter_stages(CurriculumPlanV2().stages)

        assert [stage.name for stage in stages] == [
            "single_note_all",
            "single_instrument_melody",
        ]

    def test_stage_filter_selects_single_note_shape_alias(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                stage_filter="single_note_shape,single_note_all",
                device="cpu",
            )
        )
        stages = trainer._filter_stages(CurriculumPlanV2().stages)

        assert [stage.name for stage in stages] == [
            "single_note_shape",
            "single_note_all",
        ]
        assert trainer._base_stage_for_training("single_note_shape") == "single_note_all"

    def test_stage_filter_rejects_removed_cached_frame_stages(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        with pytest.raises(RuntimeError, match="Cached FFT frame pools are disabled"):
            SourceTrackingTrainerV2(
                TrainingConfigV2(
                    data_root=str(tmp_path / "data"),
                    stage_filter="single_note_frames_cached,single_instrument_melody_frames_cached",
                    frame_cache_examples_per_stage=5,
                    device="cpu",
                )
            )

    def test_stage_filter_rejects_removed_cached_frame_names(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                stage_filter="single_note_frames_cached,single_instrument_melody_frames_cached",
                device="cpu",
            )
        )
        with pytest.raises(ValueError, match="Unknown training stage"):
            trainer._filter_stages(CurriculumPlanV2().stages)

    def test_cli_parses_loss_weights_and_stage_filter(self):
        config = TrainingCliV2.parse(
            [
                "--stage-filter",
                "single_note_all,single_instrument_melody",
                "--inactive-slot-weight",
                "6",
                "--count-loss-weight",
                "3",
                "--onset-loss-weight",
                "0.5",
                "--offset-loss-weight",
                "0.05",
                "--onset-focal-gamma",
                "2",
                "--offset-focal-gamma",
                "1",
                "--event-decoder-layers",
                "2",
                "--event-teacher-forcing-start",
                "0.8",
                "--event-teacher-forcing-end",
                "0",
                "--event-state-onset-threshold",
                "0.25",
                "--event-state-offset-threshold",
                "0.03",
                "--event-state-soft-events",
                "--validation-examples-per-epoch",
                "4",
                "--validation-interval-epochs",
                "2",
                "--validation-teacher-forcing-rate",
                "0.25",
                "--onset-shoulder-radius-frames",
                "5",
                "--offset-label-radius-frames",
                "8",
                "--boundary-negative-radius-frames",
                "32",
                "--onset-hard-negative-sample-prob",
                "0.25",
                "--onset-context-sample-prob",
                "1.0",
                "--positive-family-boosts",
                "guitar=2,flute=2",
                "--positive-quality-boosts",
                "fast_decay=2,bright=1.5",
                "--positive-source-boosts",
                "acoustic=1.2",
                "--positive-velocity-boosts",
                "50=1.4",
                "--early-stopping-metric",
                "onset_average_precision",
                "--early-stopping-patience",
                "10",
                "--early-stopping-min-delta",
                "0.001",
                "--learning-rate-scale",
                "0.1",
                "--learning-rate-decay",
                "cosine",
                "--learning-rate-min-fraction",
                "0.1",
                "--learning-rate-decay-epochs",
                "40",
                "--hard-boundary-negative-loss-weight",
                "0.75",
                "--hard-boundary-negative-fraction",
                "0.05",
                "--onset-pairwise-ranking-loss-weight",
                "0.2",
                "--onset-pairwise-ranking-margin",
                "0.25",
                "--onset-softmax-loss-weight",
                "0.5",
                "--onset-sequence-loss-weight",
                "0.25",
                "--onset-sequence-pairwise-ranking-loss-weight",
                "0.1",
                "--onset-sequence-block-positive-loss-weight",
                "0.3",
                "--onset-sequence-block-ranking-loss-weight",
                "0.7",
                "--onset-sequence-post-onset-ranking-loss-weight",
                "1.3",
                "--onset-sequence-post-onset-min-frames",
                "4",
                "--onset-sequence-post-onset-max-frames",
                "12",
                "--onset-nearby-pairwise-ranking-loss-weight",
                "0.9",
                "--onset-peak-to-shoulder-ranking-loss-weight",
                "1.1",
                "--onset-peak-loss-weight",
                "0.25",
                "--onset-shoulder-loss-weight",
                "0.12",
                "--onset-event-recall-loss-weight",
                "1.5",
                "--onset-false-peak-loss-weight",
                "0.4",
                "--first-pass-boundary-loss-weight",
                "0.5",
                "--free-run-onset-sequence-loss-weight",
                "0.4",
                "--free-run-onset-sequence-pairwise-ranking-loss-weight",
                "0.6",
                "--first-pass-distillation-loss-weight",
                "0.7",
                "--first-pass-sequence-distillation-loss-weight",
                "0.8",
                "--first-pass-distillation-offset-weight",
                "0.2",
                "--phase1-onset-teacher-checkpoint",
                "phase1.pt",
                "--phase1-onset-distillation-loss-weight",
                "0.11",
                "--phase1-onset-sequence-distillation-loss-weight",
                "0.22",
                "--first-pass-event-age-loss-weight",
                "0.9",
                "--first-pass-event-age-offset-weight",
                "0.3",
                "--first-pass-event-age-recent-weight",
                "500",
                "--freeze-for-first-pass-event-age",
                "--freeze-for-free-run-onset-sequence",
                "--freeze-for-free-run-event-decoder",
                "--event-state-use-predicted-age",
                "--disable-event-state-conditioning",
                "--primary-audio-only-event-decoder",
                "--event-state-noise-std",
                "0.1",
                "--event-state-dropout-prob",
                "0.2",
                "--input-novelty-features",
                "leaf",
                "--phase-jitter-samples",
                "79",
                "--feature-noise-std",
                "0.01",
                "--anneal-noise-phase-jitter-samples",
                "23",
                "--anneal-noise-feature-std",
                "0.02",
            ]
        )

        assert config.stage_filter == "single_note_all,single_instrument_melody"
        assert config.inactive_slot_weight == 6
        assert config.count_loss_weight == 3
        assert config.onset_loss_weight == 0.5
        assert config.offset_loss_weight == 0.05
        assert config.onset_focal_gamma == 2
        assert config.offset_focal_gamma == 1
        assert config.event_decoder_layers == 2
        assert config.event_teacher_forcing_start == 0.8
        assert config.event_teacher_forcing_end == 0
        assert config.event_state_onset_threshold == 0.25
        assert config.event_state_offset_threshold == 0.03
        assert config.event_state_soft_events is True
        assert config.validation_examples_per_epoch == 4
        assert config.validation_interval_epochs == 2
        assert config.validation_teacher_forcing_rate == 0.25
        assert config.onset_shoulder_radius_frames == 5
        assert config.offset_label_radius_frames == 8
        assert config.boundary_negative_radius_frames == 32
        assert config.onset_hard_negative_sample_prob == 0.25
        assert config.onset_context_sample_prob == 1.0
        assert config.positive_family_boosts == "guitar=2,flute=2"
        assert config.positive_quality_boosts == "fast_decay=2,bright=1.5"
        assert config.positive_source_boosts == "acoustic=1.2"
        assert config.positive_velocity_boosts == "50=1.4"
        assert config.early_stopping_metric == "onset_average_precision"
        assert config.early_stopping_patience == 10
        assert config.early_stopping_min_delta == 0.001
        assert config.learning_rate_scale == 0.1
        assert config.learning_rate_decay == "cosine"
        assert config.learning_rate_min_fraction == 0.1
        assert config.learning_rate_decay_epochs == 40
        assert config.hard_boundary_negative_loss_weight == 0.75
        assert config.hard_boundary_negative_fraction == 0.05
        assert config.onset_pairwise_ranking_loss_weight == 0.2
        assert config.onset_pairwise_ranking_margin == 0.25
        assert config.onset_softmax_loss_weight == 0.5
        assert config.onset_sequence_loss_weight == 0.25
        assert config.onset_sequence_pairwise_ranking_loss_weight == 0.1
        assert config.onset_sequence_block_positive_loss_weight == 0.3
        assert config.onset_sequence_block_ranking_loss_weight == 0.7
        assert config.onset_sequence_post_onset_ranking_loss_weight == 1.3
        assert config.onset_sequence_post_onset_min_frames == 4
        assert config.onset_sequence_post_onset_max_frames == 12
        assert config.onset_nearby_pairwise_ranking_loss_weight == 0.9
        assert config.onset_peak_to_shoulder_ranking_loss_weight == 1.1
        assert config.onset_peak_loss_weight == 0.25
        assert config.onset_shoulder_loss_weight == 0.12
        assert config.onset_event_recall_loss_weight == 1.5
        assert config.onset_false_peak_loss_weight == 0.4
        assert config.first_pass_boundary_loss_weight == 0.5
        assert config.free_run_onset_sequence_loss_weight == 0.4
        assert config.free_run_onset_sequence_pairwise_ranking_loss_weight == 0.6
        assert config.first_pass_distillation_loss_weight == 0.7
        assert config.first_pass_sequence_distillation_loss_weight == 0.8
        assert config.first_pass_distillation_offset_weight == 0.2
        assert config.phase1_onset_teacher_checkpoint == "phase1.pt"
        assert config.phase1_onset_distillation_loss_weight == 0.11
        assert config.phase1_onset_sequence_distillation_loss_weight == 0.22
        assert config.first_pass_event_age_loss_weight == 0.9
        assert config.first_pass_event_age_offset_weight == 0.3
        assert config.first_pass_event_age_recent_weight == 500
        assert config.freeze_for_first_pass_event_age is True
        assert config.freeze_for_free_run_onset_sequence is True
        assert config.freeze_for_free_run_event_decoder is True
        assert config.event_state_use_predicted_age is True
        assert config.event_state_conditioning is False
        assert config.primary_audio_only_event_decoder is True
        assert config.event_state_noise_std == 0.1
        assert config.event_state_dropout_prob == 0.2
        assert config.input_novelty_features == "leaf"
        assert config.phase_jitter_samples == 79
        assert config.feature_noise_std == 0.01
        assert config.anneal_noise_phase_jitter_samples == 23
        assert config.anneal_noise_feature_std == 0.02

    def test_flux_input_features_append_bandwise_positive_deltas(self):
        builder = object.__new__(TrainingBatchBuilderV2)
        builder.input_novelty_features = "flux"
        frames = torch.tensor(
            [
                [
                    [0.1, 0.3],
                    [0.4, 0.2],
                    [0.2, 0.5],
                ]
            ],
            dtype=torch.float32,
        )

        augmented = builder._augment_frame_features(frames)

        assert augmented.shape == (1, 3, 4)
        assert torch.equal(augmented[..., :2], frames)
        assert torch.allclose(
            augmented[..., 2:],
            torch.tensor([[[0.0, 0.0], [0.3, 0.0], [0.0, 0.3]]]),
        )

    def test_salience_input_features_append_dsp_onset_channels(self):
        builder = object.__new__(TrainingBatchBuilderV2)
        builder.input_novelty_features = "salience"
        frames = torch.tensor(
            [
                [
                    [0.1, 0.3],
                    [0.4, 0.2],
                    [0.2, 0.5],
                ]
            ],
            dtype=torch.float32,
        )

        augmented = builder._augment_frame_features(frames)

        assert TrainingBatchBuilderV2.feature_dim("salience") == (
            SourceTrackingAudioConfigV2.bands * 6 + 5
        )
        assert augmented.shape == (1, 3, 17)
        assert torch.equal(augmented[..., :2], frames)
        assert torch.allclose(
            augmented[..., 2:4],
            torch.tensor([[[0.0, 0.0], [0.3, 0.0], [0.0, 0.3]]]),
        )
        assert torch.allclose(
            augmented[..., 4:6],
            torch.tensor([[[0.0, 0.0], [0.0, 0.1], [0.2, 0.0]]]),
        )
        assert torch.allclose(
            augmented[..., -5:],
            torch.tensor(
                [
                    [
                        [0.2, 0.0, 0.0, 0.0, 0.75],
                        [0.3, 0.15, 0.3, 0.0, 1.0 / 3.0],
                        [0.35, 0.15, 0.0, 0.3, 5.0 / 7.0],
                    ]
                ]
            ),
        )

    def test_leaf_feature_dim_matches_frontend_descriptor(self):
        assert TrainingBatchBuilderV2.feature_dim("leaf") == LeafFrontendConfigV2().feature_dim

    def test_trainer_rejects_legacy_public_frontends(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()

        with pytest.raises(RuntimeError, match="LEAF frontend"):
            SourceTrackingTrainerV2(
                TrainingConfigV2(
                    data_root=str(tmp_path / "data"),
                    device="cpu",
                    input_novelty_features="flux",
                )
            )

    def test_freeze_for_first_pass_event_age_trains_only_age_heads(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                freeze_for_first_pass_event_age=True,
            )
        )

        trainable = trainer._configure_trainable_parameters()
        trainable_names = {
            name for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }

        assert len(trainable) == 4
        assert trainable_names == {
            "event_decoder.onset_age.weight",
            "event_decoder.onset_age.bias",
            "event_decoder.offset_age.weight",
            "event_decoder.offset_age.bias",
        }

    def test_freeze_for_free_run_onset_sequence_trains_only_onset_head(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                freeze_for_free_run_onset_sequence=True,
            )
        )

        trainable = trainer._configure_trainable_parameters()
        trainable_names = {
            name for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }

        assert len(trainable) == 2
        assert trainable_names == {
            "event_decoder.onset.weight",
            "event_decoder.onset.bias",
        }

    def test_freeze_for_free_run_event_decoder_trains_only_decoder(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                freeze_for_free_run_event_decoder=True,
            )
        )

        trainer._configure_trainable_parameters()
        trainable_names = {
            name for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }

        assert trainable_names
        assert all(name.startswith("event_decoder.") for name in trainable_names)

    def test_first_pass_outputs_are_kept_only_for_auxiliary_losses(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        base = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                first_pass_boundary_loss_weight=0.0,
            )
        )
        auxiliary = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                first_pass_boundary_loss_weight=0.5,
            )
        )
        free_run_sequence = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                free_run_onset_sequence_loss_weight=0.25,
            )
        )
        distillation = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                first_pass_distillation_loss_weight=0.25,
            )
        )
        event_age = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=1,
                first_pass_event_age_loss_weight=0.25,
            )
        )
        batch = base._build_validation_batch("single_note_all", 0, 1)

        assert "_first_pass_outputs" not in base._model_forward(
            batch,
            teacher_forcing_rate=0.0,
        )
        assert "_first_pass_outputs" in auxiliary._model_forward(
            batch,
            teacher_forcing_rate=0.0,
        )
        assert "_first_pass_outputs" in free_run_sequence._model_forward(
            batch,
            teacher_forcing_rate=0.0,
        )
        distillation_outputs = distillation._model_forward(
            batch,
            teacher_forcing_rate=0.0,
        )
        assert "_first_pass_outputs" in distillation_outputs
        assert "_first_pass_teacher_outputs" in distillation_outputs
        assert "_first_pass_outputs" in event_age._model_forward(
            batch,
            teacher_forcing_rate=0.0,
        )

    def test_boundary_radius_config_reaches_renderer(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                onset_shoulder_radius_frames=5,
                offset_label_radius_frames=8,
                boundary_negative_radius_frames=32,
                onset_hard_negative_sample_prob=0.25,
            )
        )

        assert trainer.renderer.config.onset_shoulder_radius_frames == 5
        assert trainer.renderer.config.offset_label_radius_frames == 8
        assert trainer.renderer.config.boundary_negative_radius_frames == 32
        assert trainer.renderer.config.onset_hard_negative_sample_prob == 0.25

    def test_hard_negative_sampler_does_not_change_validation_batches(
        self,
        tmp_path,
    ):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        base = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=4,
                onset_hard_negative_sample_prob=0.0,
            )
        )
        hard_negative = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                batch_size=4,
                onset_hard_negative_sample_prob=0.25,
            )
        )

        base_batch = base._build_validation_batch("single_note_all", 0, 4)
        hard_negative_batch = hard_negative._build_validation_batch(
            "single_note_all",
            0,
            4,
        )

        assert torch.equal(base_batch["onset"], hard_negative_batch["onset"])
        assert torch.equal(
            base_batch["context_onset"],
            hard_negative_batch["context_onset"],
        )

    def test_one_train_step_writes_metrics_and_checkpoint(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            smoke_examples=2,
            device="cpu",
        )

        payload = SourceTrackingTrainerV2(config).run()

        assert payload["history"]
        assert payload["history"][0]["metric_scope"] == "train_batch"
        assert payload["history"][0]["split"] == "train"
        assert payload["artifacts"]["checkpoint_path"].endswith("checkpoint.pt")
        assert payload["artifacts"]["metrics_path"].endswith("metrics.json")
        assert list((tmp_path / "runs").glob("*/metrics.json"))
        checkpoint = next((tmp_path / "runs").glob("*/checkpoint.pt"))
        checkpoint_payload = torch.load(checkpoint, map_location="cpu")
        assert checkpoint_payload["optimizer_state_dict"] is not None
        assert checkpoint_payload["checkpoint_metadata"]["optimizer_state_included"]

    def test_warm_start_checkpoint_runs_one_train_step(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        first = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs" / "first"),
            calibration_examples=1,
            batch_size=1,
            smoke_examples=1,
            device="cpu",
        )
        SourceTrackingTrainerV2(first).run()
        checkpoint = next((tmp_path / "runs" / "first").glob("*/checkpoint.pt"))
        second = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs" / "second"),
            calibration_examples=1,
            batch_size=1,
            smoke_examples=1,
            device="cpu",
            warm_start_checkpoint=str(checkpoint),
        )

        payload = SourceTrackingTrainerV2(second).run()

        assert payload["config"]["warm_start_checkpoint"] == str(checkpoint)
        assert payload["artifacts"]["checkpoint_path"].endswith("checkpoint.pt")
        assert list((tmp_path / "runs" / "second").glob("*/checkpoint.pt"))

    def test_prefetch_train_step_runs(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            smoke_examples=2,
            device="cpu",
            prefetch_batches=1,
        )

        payload = SourceTrackingTrainerV2(config).run()

        assert payload["history"]
        assert payload["config"]["prefetch_batches"] == 1

    def test_annealed_noise_train_step_runs(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            epochs_per_stage=1,
            stage_filter="single_note_all",
            smoke_examples=0,
            curriculum_scale=0.0002,
            anneal_noise_phase_jitter_samples=12,
            anneal_noise_feature_std=0.01,
            device="cpu",
        )

        payload = SourceTrackingTrainerV2(config).run()

        assert payload["history"]
        assert payload["history"][0]["stage"] == "single_note_all"
        assert payload["config"]["anneal_noise_phase_jitter_samples"] == 12
        assert payload["config"]["anneal_noise_feature_std"] == 0.01

    def test_constant_noise_train_step_runs(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            epochs_per_stage=1,
            stage_filter="single_note_all",
            smoke_examples=0,
            curriculum_scale=0.0002,
            phase_jitter_samples=12,
            feature_noise_std=0.01,
            device="cpu",
        )

        payload = SourceTrackingTrainerV2(config).run()

        assert payload["history"]
        assert payload["history"][0]["stage"] == "single_note_all"
        assert payload["config"]["phase_jitter_samples"] == 12
        assert payload["config"]["feature_noise_std"] == 0.01

    def test_validation_epoch_metrics_use_inference_style_validation(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            epochs_per_stage=1,
            stage_filter="single_note_all",
            smoke_examples=0,
            curriculum_scale=0.0002,
            validation_examples_per_epoch=2,
            validation_interval_epochs=1,
            device="cpu",
        )

        payload = SourceTrackingTrainerV2(config).run()

        validation = payload["history"][0]["validation"]
        assert validation["metric_scope"] == "validation"
        assert (
            validation["metric_definition"]
            == "pointwise_inference_style_validation_onset_shoulder_accepted_"
            "sequence_onset_censored_block_ap"
        )
        assert validation["onset_metric_label"] == "onset_nearby_mask_when_available"
        assert validation["split"] == "validation"
        assert validation["event_teacher_forcing_rate"] == 0.0
        assert validation["examples"] == 2
        assert validation["metric_threshold"] == config.metric_activity_threshold
        assert "onset_average_precision" in validation
        assert "onset_best_f1" in validation
        assert "onset_best_threshold" in validation

    def test_validation_teacher_forcing_rate_can_be_overridden(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            epochs_per_stage=1,
            stage_filter="single_note_all",
            smoke_examples=0,
            curriculum_scale=0.0002,
            validation_examples_per_epoch=2,
            validation_interval_epochs=1,
            validation_teacher_forcing_rate=0.25,
            device="cpu",
        )

        payload = SourceTrackingTrainerV2(config).run()

        validation = payload["history"][0]["validation"]
        assert validation["event_teacher_forcing_rate"] == 0.25

    def test_validation_split_can_use_train_for_diagnostics(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            epochs_per_stage=1,
            stage_filter="single_note_all",
            smoke_examples=0,
            curriculum_scale=0.0002,
            validation_examples_per_epoch=2,
            validation_interval_epochs=1,
            validation_split="train",
            device="cpu",
        )

        payload = SourceTrackingTrainerV2(config).run()

        validation = payload["history"][0]["validation"]
        assert validation["split"] == "train"

    def test_validation_seeds_are_fixed_across_epochs(self):
        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.config = TrainingConfigV2(batch_size=4, seed=123, device="cpu")

        first_epoch = trainer._validation_seeds(batch_idx=2, batch_size=3)
        later_epoch = trainer._validation_seeds(batch_idx=2, batch_size=3)

        assert first_epoch == later_epoch
        assert first_epoch == [900_000_131, 900_000_132, 900_000_133]

    def test_threshold_free_boundary_metrics(self):
        metrics = SourceTrackingTrainerV2._threshold_free_boundary_metrics(
            torch.tensor([0.9, 0.8, 0.7, 0.1]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "onset",
        )

        assert metrics["onset_average_precision"] == pytest.approx(
            (1.0 + 2.0 / 3.0) / 2.0
        )
        assert metrics["onset_best_f1"] == pytest.approx(0.8)
        assert metrics["onset_best_precision"] == pytest.approx(2.0 / 3.0)
        assert metrics["onset_best_recall"] == pytest.approx(1.0)
        assert metrics["onset_best_threshold"] == pytest.approx(0.7)

    def test_slot_invariant_boundary_metrics_ignore_source_slot_permutation(self):
        scores = torch.tensor(
            [
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.1],
            ]
        )
        targets = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

        per_slot = SourceTrackingTrainerV2._threshold_free_boundary_metrics(
            scores.reshape(-1),
            targets.reshape(-1),
            "onset",
        )
        slot_invariant = SourceTrackingTrainerV2._slot_invariant_boundary_metrics(
            scores,
            targets,
            "onset_slot_invariant",
        )

        assert per_slot["onset_average_precision"] < 1.0
        assert slot_invariant["onset_slot_invariant_average_precision"] == pytest.approx(
            1.0
        )

    def test_early_stopping_state_tracks_validation_metric_patience(self):
        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.config = TrainingConfigV2(
            early_stopping_metric="onset_average_precision",
            early_stopping_patience=2,
            early_stopping_min_delta=0.01,
            device="cpu",
        )

        assert trainer._early_stopping_state(
            {"onset_average_precision": 0.2},
            None,
            0,
        ) == (0.2, 0, True)
        assert trainer._early_stopping_state(
            {"onset_average_precision": 0.205},
            0.2,
            0,
        ) == (0.2, 1, False)
        assert trainer._should_stop_early(1) is False
        assert trainer._should_stop_early(2) is True

    def test_zero_teacher_forcing_uses_inference_style_first_pass(self):
        class SpyModel:
            def __init__(self):
                self.calls = []

            def __call__(self, frames, event_state=None):
                self.calls.append(event_state)
                batch, frames_count = frames.shape[:2]
                sources = 2
                logits = torch.full((batch, frames_count, sources), -10.0)
                return {
                    "onset_logits_sequence": logits,
                    "offset_logits_sequence": logits,
                }

        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.model = SpyModel()
        trainer.config = TrainingConfigV2()
        batch = {
            "frames": torch.zeros(1, 4, 40),
            "event_state": torch.ones(1, 4, 2, 5),
        }

        outputs = trainer._model_forward(batch, teacher_forcing_rate=0.0)

        assert outputs["onset_logits_sequence"].shape == (1, 4, 2)
        assert trainer.model.calls[0] is None
        assert trainer.model.calls[1] is not batch["event_state"]

    def test_predicted_event_age_can_override_generated_state(self):
        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.config = TrainingConfigV2(event_state_use_predicted_age=True)
        true_state = torch.zeros(1, 3, 2, 5)
        outputs = {
            "onset_logits_sequence": torch.full((1, 3, 2), -10.0),
            "offset_logits_sequence": torch.full((1, 3, 2), -10.0),
            "onset_age_logits_sequence": torch.zeros(1, 3, 2),
            "offset_age_logits_sequence": torch.full(
                (1, 3, 2),
                torch.logit(torch.tensor(0.25)),
            ),
        }

        state = trainer._scheduled_event_state(
            true_state,
            outputs,
            teacher_forcing_rate=0.0,
        )

        torch.testing.assert_close(state[..., 3], torch.full((1, 3, 2), 0.5))
        torch.testing.assert_close(state[..., 4], torch.full((1, 3, 2), 0.25))

    def test_event_state_corruption_is_training_only(self):
        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.config = TrainingConfigV2(
            event_state_noise_std=0.0,
            event_state_dropout_prob=1.0,
        )
        trainer.model = torch.nn.Linear(1, 1)
        event_state = torch.ones(1, 2, 3, 5)

        trainer.model.train()
        corrupted = trainer._corrupt_event_state(event_state)
        trainer.model.eval()
        unchanged = trainer._corrupt_event_state(event_state)

        assert torch.count_nonzero(corrupted) == 0
        torch.testing.assert_close(unchanged, event_state)

    def test_full_teacher_forcing_uses_truth_state_once(self):
        class SpyModel:
            def __init__(self):
                self.calls = []

            def __call__(self, frames, event_state=None):
                self.calls.append(event_state)
                return {
                    "onset_logits_sequence": torch.zeros(1, 4, 2),
                    "offset_logits_sequence": torch.zeros(1, 4, 2),
                }

        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.model = SpyModel()
        batch = {
            "frames": torch.zeros(1, 4, 40),
            "event_state": torch.ones(1, 4, 2, 5),
        }

        trainer._model_forward(batch, teacher_forcing_rate=1.0)

        assert trainer.model.calls == [batch["event_state"]]

    def test_disabled_event_state_conditioning_bypasses_scheduled_sampling(self):
        class SpyModel:
            def __init__(self):
                self.calls = []

            def __call__(self, frames, event_state=None):
                self.calls.append(event_state)
                batch, frames_count = frames.shape[:2]
                sources = 2
                logits = torch.zeros(batch, frames_count, sources)
                return {
                    "onset_logits_sequence": logits,
                    "offset_logits_sequence": logits,
                }

        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.model = SpyModel()
        trainer.config = TrainingConfigV2(event_state_conditioning=False)
        batch = {
            "frames": torch.zeros(1, 4, 40),
            "event_state": torch.ones(1, 4, 2, 5),
        }

        trainer._model_forward(batch, teacher_forcing_rate=1.0)

        assert trainer.model.calls == [None]

    def test_primary_audio_only_event_decoder_bypasses_scheduled_sampling(self):
        class SpyModel:
            def __init__(self):
                self.calls = []

            def __call__(self, frames, event_state=None):
                self.calls.append(event_state)
                batch, frames_count = frames.shape[:2]
                sources = 2
                logits = torch.zeros(batch, frames_count, sources)
                return {
                    "onset_logits_sequence": logits,
                    "offset_logits_sequence": logits,
                }

        trainer = object.__new__(SourceTrackingTrainerV2)
        trainer.model = SpyModel()
        trainer.config = TrainingConfigV2(primary_audio_only_event_decoder=True)
        batch = {
            "frames": torch.zeros(1, 4, 40),
            "event_state": torch.ones(1, 4, 2, 5),
        }

        trainer._model_forward(batch, teacher_forcing_rate=0.0)

        assert trainer.model.calls == [None]

    def test_onset_sequence_negative_diagnostics_splits_hard_negative_classes(self):
        scores = torch.tensor([[[0.9], [0.7], [0.1], [0.8]]])
        targets = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
        nearby = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
        timing = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])

        metrics = SourceTrackingTrainerV2._onset_sequence_negative_diagnostics(
            scores,
            targets,
            nearby,
            timing,
        )

        assert metrics["onset_sequence_positive_score_count"] == 2
        assert metrics["onset_sequence_core_positive_score_count"] == 1
        assert metrics["onset_sequence_shoulder_score_count"] == 1
        assert metrics["onset_sequence_nearby_negative_score_count"] == 1
        assert metrics["onset_sequence_far_negative_score_count"] == 2
        assert metrics["onset_sequence_false_peak_negative_score_count"] == 1
        assert metrics["onset_sequence_nearby_negative_score_max"] == pytest.approx(0.7)
        assert metrics["onset_sequence_false_peak_negative_score_max"] == pytest.approx(0.8)

    def test_censored_onset_sequence_metric_uses_one_item_per_onset_block(self):
        scores = torch.tensor([[[0.2], [0.9], [0.1], [0.8], [0.05]]])
        nearby = torch.tensor([[[1.0], [1.0], [1.0], [0.0], [0.0]]])
        delta = torch.tensor([[[-1.0], [0.0], [1.0], [0.0], [0.0]]])
        timing = torch.tensor([[[1.0], [1.0], [1.0], [0.0], [0.0]]])

        metric_scores, metric_targets = (
            SourceTrackingTrainerV2._censored_onset_sequence_metric_items(
                scores,
                nearby,
                delta,
                timing,
            )
        )

        torch.testing.assert_close(metric_scores, torch.tensor([0.9, 0.8, 0.05]))
        torch.testing.assert_close(metric_targets, torch.tensor([1.0, 0.0, 0.0]))
        metrics = SourceTrackingTrainerV2._threshold_free_boundary_metrics(
            metric_scores,
            metric_targets,
            "onset_sequence",
        )
        assert metrics["onset_sequence_average_precision"] == pytest.approx(1.0)

    def test_boundary_classification_metrics_aggregate_counts(self):
        first = {
            "onset_true_positive_count": 1.0,
            "onset_false_positive_count": 1.0,
            "onset_false_negative_count": 1.0,
            "onset_true_negative_count": 1.0,
            "onset_target_positive_count": 2.0,
            "onset_target_negative_count": 2.0,
            "onset_predicted_positive_count": 2.0,
            "offset_true_positive_count": 0.0,
            "offset_false_positive_count": 0.0,
            "offset_false_negative_count": 1.0,
            "offset_true_negative_count": 3.0,
            "offset_target_positive_count": 1.0,
            "offset_target_negative_count": 3.0,
            "offset_predicted_positive_count": 0.0,
            "source_count_accuracy": 1.0,
        }
        second = {
            **first,
            "onset_true_positive_count": 2.0,
            "onset_false_positive_count": 0.0,
            "onset_false_negative_count": 0.0,
            "onset_true_negative_count": 2.0,
            "offset_true_positive_count": 1.0,
            "offset_false_positive_count": 1.0,
            "offset_false_negative_count": 0.0,
            "offset_true_negative_count": 2.0,
            "source_count_accuracy": 0.0,
        }

        metrics = SourceTrackingTrainerV2._mean_batch_metrics([first, second])

        assert metrics["source_count_accuracy"] == 0.5
        assert metrics["onset_true_positive_count"] == 3.0
        assert metrics["onset_false_positive_count"] == 1.0
        assert metrics["onset_false_negative_count"] == 1.0
        assert metrics["onset_precision"] == 0.75
        assert metrics["onset_recall"] == 0.75
        assert metrics["onset_f1"] == 0.75
        assert metrics["onset_false_positive_rate"] == 0.25
        assert metrics["offset_precision"] == 0.5
        assert metrics["offset_recall"] == 0.5
        assert metrics["offset_f1"] == 0.5
        assert metrics["offset_false_positive_rate"] == pytest.approx(1 / 6)

    def test_process_render_pool_train_step_runs(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=2,
            batch_size=2,
            smoke_examples=2,
            device="cpu",
            render_workers=2,
        )

        payload = SourceTrackingTrainerV2(config).run()

        assert payload["history"]
        assert payload["config"]["render_workers"] == 2

    def test_partial_warm_start_loads_matching_deeper_model_tensors(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        checkpoint_path = tmp_path / "checkpoint.pt"
        base = SourceTrackingTrainerV2(
            TrainingConfigV2(data_root=str(tmp_path / "data"), device="cpu")
        )
        torch.save(
            {
                "state_dict": base.model.state_dict(),
                "checkpoint_metadata": {
                    "checkpoint_number": 12,
                    "phase": "batch_interval",
                    "stage": "simple_duo_trio",
                    "epoch": 1,
                    "batch": 25,
                    "batches_per_epoch": 313,
                    "examples_seen": 6_400,
                },
            },
            checkpoint_path,
        )

        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                device="cpu",
                warm_start_checkpoint=str(checkpoint_path),
                partial_warm_start=True,
                model_layers=4,
            )
        )

        assert trainer.warm_start_load_report is not None
        assert trainer.warm_start_load_report["mode"] == "partial"
        assert trainer.warm_start_load_report["loaded_tensors"] > 0
        assert trainer.warm_start_load_report["missing_tensors"] > 0

    def test_checkpoint_upload_writes_numbered_artifacts(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        upload_root = tmp_path / "uploads"
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            run_root=str(tmp_path / "runs"),
            calibration_examples=1,
            batch_size=1,
            smoke_examples=1,
            device="cpu",
            checkpoint_upload_uri=str(upload_root),
            checkpoint_upload_run_id="unit-run",
            checkpoint_interval_batches=1,
        )

        payload = SourceTrackingTrainerV2(config).run()

        uploaded = list(upload_root.glob("*/unit-run/checkpoints/checkpoint_*.json"))
        assert uploaded
        metadata = json.loads(sorted(uploaded)[0].read_text(encoding="utf-8"))
        assert metadata["checkpoint_number"] == 1
        assert metadata["phase"] in {"batch_interval", "epoch_done", "training_done"}
        assert "stage" in metadata
        assert metadata["optimizer_state_included"]
        assert metadata["optimizer_class"] == "AdamW"
        uploaded_checkpoint = sorted(upload_root.glob("*/unit-run/checkpoints/*.pt"))[0]
        checkpoint_payload = torch.load(uploaded_checkpoint, map_location="cpu")
        assert checkpoint_payload["optimizer_state_dict"] is not None
        assert payload["artifacts"]["checkpoint_upload_uri"].endswith("/unit-run")
        run_dir = Path(payload["artifacts"]["run_dir"])
        assert not list((run_dir / "checkpoints").glob("checkpoint_*.pt"))
        assert not list((run_dir / "checkpoints").glob("checkpoint_*.json"))
        assert (run_dir / "checkpoints" / "latest.json").exists()

    def test_resume_cursor_from_warm_start_metadata(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        checkpoint_path = tmp_path / "checkpoint.pt"
        base = SourceTrackingTrainerV2(
            TrainingConfigV2(data_root=str(tmp_path / "data"), device="cpu")
        )
        torch.save(
            {
                "state_dict": base.model.state_dict(),
                "checkpoint_metadata": {
                    "checkpoint_number": 45,
                    "phase": "batch_interval",
                    "stage": "single_instrument_melody",
                    "epoch": 2,
                    "batch": 500,
                    "batches_per_epoch": 938,
                },
            },
            checkpoint_path,
        )
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            device="cpu",
            warm_start_checkpoint=str(checkpoint_path),
            resume_from_warm_start_metadata=True,
        )
        trainer = SourceTrackingTrainerV2(config)
        stages = CurriculumPlanV2().scale_epochs(2).stages

        cursor = trainer._resume_cursor(stages)

        assert cursor is not None
        assert cursor.stage == "single_instrument_melody"
        assert cursor.epoch == 2
        assert cursor.batch == 500
        assert cursor.checkpoint_number == 45
        assert trainer._stages_from_resume(stages, cursor)[0].name == (
            "single_instrument_melody"
        )

    def test_resume_cursor_recomputes_batch_when_batch_size_changes(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        checkpoint_path = tmp_path / "checkpoint.pt"
        base = SourceTrackingTrainerV2(
            TrainingConfigV2(data_root=str(tmp_path / "data"), device="cpu")
        )
        torch.save(
            {
                "state_dict": base.model.state_dict(),
                "checkpoint_metadata": {
                    "checkpoint_number": 64,
                    "phase": "batch_interval",
                    "stage": "single_instrument_melody",
                    "epoch": 4,
                    "batch": 400,
                    "batches_per_epoch": 938,
                    "examples_seen": 205_600,
                },
            },
            checkpoint_path,
        )
        config = TrainingConfigV2(
            data_root=str(tmp_path / "data"),
            device="cpu",
            batch_size=256,
            warm_start_checkpoint=str(checkpoint_path),
            resume_from_warm_start_metadata=True,
        )
        trainer = SourceTrackingTrainerV2(config)
        stages = CurriculumPlanV2().scale_epochs(2).stages

        cursor = trainer._resume_cursor(stages)

        assert cursor is not None
        assert cursor.stage == "single_instrument_melody"
        assert cursor.epoch == 4
        assert cursor.batch == 100
        assert cursor.checkpoint_number == 64
