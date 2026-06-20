from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import TinyCorpusBuilderV2

from muziq_nn.training.train import (
    CurriculumPlanV2,
    SourceTrackingTrainerV2,
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

    def test_stage_filter_rejects_removed_cached_frame_stages(self, tmp_path):
        TinyCorpusBuilderV2(tmp_path / "data").build()
        trainer = SourceTrackingTrainerV2(
            TrainingConfigV2(
                data_root=str(tmp_path / "data"),
                stage_filter="single_note_frames_cached,single_instrument_melody_frames_cached",
                frame_cache_examples_per_stage=5,
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
                "--onset-peak-loss-weight",
                "0.25",
                "--onset-event-recall-loss-weight",
                "1.5",
                "--onset-false-peak-loss-weight",
                "0.4",
                "--anneal-noise-phase-jitter-samples",
                "79",
                "--anneal-noise-feature-std",
                "0.01",
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
        assert config.onset_peak_loss_weight == 0.25
        assert config.onset_event_recall_loss_weight == 1.5
        assert config.onset_false_peak_loss_weight == 0.4
        assert config.anneal_noise_phase_jitter_samples == 79
        assert config.anneal_noise_feature_std == 0.01

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
