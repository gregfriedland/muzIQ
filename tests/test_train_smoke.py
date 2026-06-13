from __future__ import annotations

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
        assert list((tmp_path / "runs").glob("*/metrics.json"))
        assert list((tmp_path / "runs").glob("*/checkpoint.pt"))
