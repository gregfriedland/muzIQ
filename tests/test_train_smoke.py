from __future__ import annotations

from conftest import TinyCorpusBuilderV2

from muziq_nn.training.train import SourceTrackingTrainerV2, TrainingConfigV2


class TestTrainSmokeV2:
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
