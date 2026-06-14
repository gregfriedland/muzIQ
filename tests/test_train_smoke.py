from __future__ import annotations

import json

import torch
from conftest import TinyCorpusBuilderV2

from muziq_nn.training.train import CurriculumPlanV2, SourceTrackingTrainerV2, TrainingConfigV2


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
