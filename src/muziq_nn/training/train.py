"""Curriculum trainer for on-the-fly source tracking."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from muziq_nn.datasets.midi import MidiIndexV2
from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    MidiScheduleStoreV2,
    NsynthNoteStoreV2,
    SourceTrackingAudioConfigV2,
    SourceTrackingRendererV2,
)
from muziq_nn.datasets.schema import SplitName
from muziq_nn.models.attention import (
    DualPathTransformerSourceTrackerV2,
    SourceTrackingLossV2,
    SourceTrackingModelConfigV2,
)


@dataclass(frozen=True)
class CurriculumStageV2:
    name: str
    train_examples_per_epoch: int
    epochs: int
    learning_rate: float


class CurriculumPlanV2:
    """Training curriculum and time-budget trimming policy."""

    DEFAULT_STAGES = (
        CurriculumStageV2("single_note_all", 30_000, 3, 3e-4),
        CurriculumStageV2("single_instrument_melody", 60_000, 3, 3e-4),
        CurriculumStageV2("simple_duo_trio", 80_000, 3, 1e-4),
        CurriculumStageV2("midi_complex", 100_000, 3, 1e-4),
        CurriculumStageV2("hard_case_finetune", 40_000, 2, 5e-5),
    )

    def __init__(self, stages: tuple[CurriculumStageV2, ...] | None = None):
        self.stages = stages or self.DEFAULT_STAGES

    def scale_examples(self, scale: float) -> CurriculumPlanV2:
        if scale <= 0.0:
            raise ValueError("curriculum scale must be positive")
        return CurriculumPlanV2(
            tuple(
                CurriculumStageV2(
                    stage.name,
                    max(1, int(round(stage.train_examples_per_epoch * scale))),
                    stage.epochs,
                    stage.learning_rate,
                )
                for stage in self.stages
            )
        )

    def scale_epochs(self, multiplier: float) -> CurriculumPlanV2:
        if multiplier <= 0.0:
            raise ValueError("epoch multiplier must be positive")
        return CurriculumPlanV2(
            tuple(
                CurriculumStageV2(
                    stage.name,
                    stage.train_examples_per_epoch,
                    max(1, int(round(stage.epochs * multiplier))),
                    stage.learning_rate,
                )
                for stage in self.stages
            )
        )

    def trim_for_hours(
        self, examples_per_second: float, max_hours: float
    ) -> tuple[CurriculumStageV2, ...]:
        stages = list(self.stages)
        while self._project_hours(stages, examples_per_second) > max_hours:
            if not self._trim_stage(stages, "midi_complex", 70_000):
                if not self._trim_stage(stages, "simple_duo_trio", 60_000):
                    if not self._trim_stage(stages, "hard_case_finetune", 30_000):
                        break
        return tuple(stages)

    @staticmethod
    def _project_hours(
        stages: Sequence[CurriculumStageV2], examples_per_second: float
    ) -> float:
        examples = sum(stage.train_examples_per_epoch * stage.epochs for stage in stages)
        return examples / max(examples_per_second, 1e-6) / 3600.0

    @staticmethod
    def _trim_stage(stages: list[CurriculumStageV2], name: str, target_examples: int) -> bool:
        for idx, stage in enumerate(stages):
            if stage.name == name and stage.train_examples_per_epoch > target_examples:
                stages[idx] = CurriculumStageV2(
                    stage.name,
                    target_examples,
                    stage.epochs,
                    stage.learning_rate,
                )
                return True
        return False


@dataclass(frozen=True)
class TrainingConfigV2:
    data_root: str = "data"
    run_root: str = "runs"
    curriculum: str = "all"
    curriculum_scale: float = 1.0
    max_hours: float = 8.0
    generate_on_the_fly: bool = True
    calibration_examples: int = 10_000
    batch_size: int = 16
    seed: int = 7
    device: str = "auto"
    smoke_examples: int = 0
    progress_interval_s: float = 10.0
    render_workers: int = 1
    warm_start_checkpoint: str | None = None
    epoch_multiplier: float = 1.0


class TrainingBatchBuilderV2:
    """Convert rendered frame endpoints into torch training batches."""

    def __init__(self, device: torch.device):
        self.device = device

    def build(self, examples) -> dict[str, torch.Tensor]:
        return self.build_slices([self.slice_from_example(example) for example in examples])

    def build_slices(self, slices) -> dict[str, torch.Tensor]:
        frames = []
        activity = []
        family = []
        onset = []
        offset = []
        for item in slices:
            frames.append(item["frames"])
            activity.append(item["activity"])
            family.append(np.maximum(item["family"], 0))
            onset.append(item["onset"])
            offset.append(item["offset"])
        max_len = max(frame.shape[0] for frame in frames)
        padded = np.zeros(
            (len(frames), max_len, SourceTrackingAudioConfigV2.bands), dtype=np.float32
        )
        for idx, frame in enumerate(frames):
            padded[idx, -frame.shape[0] :] = frame
        return {
            "frames": torch.from_numpy(padded).to(self.device),
            "activity": torch.from_numpy(np.asarray(activity, dtype=np.float32)).to(
                self.device
            ),
            "family": torch.from_numpy(np.asarray(family, dtype=np.int64)).to(self.device),
            "onset": torch.from_numpy(np.asarray(onset, dtype=np.float32)).to(self.device),
            "offset": torch.from_numpy(np.asarray(offset, dtype=np.float32)).to(self.device),
        }

    @classmethod
    def slice_from_example(cls, example) -> dict[str, np.ndarray]:
        frame_idx = cls._sample_active_frame(example)
        return {
            "frames": example.frames[max(0, frame_idx - 255) : frame_idx + 1],
            "activity": example.labels.active[frame_idx],
            "family": example.labels.family[frame_idx],
            "onset": example.labels.onset[frame_idx],
            "offset": example.labels.offset[frame_idx],
        }

    @staticmethod
    def _sample_active_frame(example) -> int:
        active = np.where(example.labels.active.any(axis=1))[0]
        if len(active) == 0:
            return int(example.frames.shape[0] - 1)
        return int(active[len(active) // 2])


class TrainingProgressLoggerV2:
    """Emit periodic JSON progress lines during long training runs."""

    def __init__(self, interval_s: float = 10.0):
        self.interval_s = interval_s
        self.started = time.monotonic()
        self.last_emit = 0.0

    def emit(self, event: str, payload: dict[str, object], force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_emit < self.interval_s:
            return
        self.last_emit = now
        print(
            json.dumps(
                {"event": event, "elapsed_s": round(now - self.started, 3), **payload},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )


class TrainingRenderWorkerV2:
    """Render batch slices in worker processes with one renderer per process."""

    _renderers: dict[str, SourceTrackingRendererV2] = {}

    @staticmethod
    def render_slice(task: tuple[str, str, SplitName, int]) -> dict[str, np.ndarray]:
        data_root, stage, split, seed = task
        renderer = TrainingRenderWorkerV2._renderer(data_root)
        example = renderer.render(stage, split, seed)
        return TrainingBatchBuilderV2.slice_from_example(example)

    @staticmethod
    def _renderer(data_root: str) -> SourceTrackingRendererV2:
        renderer = TrainingRenderWorkerV2._renderers.get(data_root)
        if renderer is not None:
            return renderer
        nsynth = NsynthIndexV2(Path(data_root))
        midi = MidiIndexV2(Path(data_root))
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(nsynth),
            MidiScheduleStoreV2(midi),
        )
        TrainingRenderWorkerV2._renderers[data_root] = renderer
        return renderer


class SourceTrackingTrainerV2:
    """Train the compact attention model from generated examples."""

    def __init__(self, config: TrainingConfigV2):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.renderer = self._build_renderer()
        self.model = DualPathTransformerSourceTrackerV2(SourceTrackingModelConfigV2()).to(
            self.device
        )
        if config.warm_start_checkpoint is not None:
            self._load_warm_start(Path(config.warm_start_checkpoint))
        self.loss_fn = SourceTrackingLossV2()
        self.batch_builder = TrainingBatchBuilderV2(self.device)
        self.progress = TrainingProgressLoggerV2(config.progress_interval_s)
        self.render_executor: ProcessPoolExecutor | None = None

    def run(self) -> dict[str, object]:
        self.progress.emit(
            "training_start",
            {
                "device": str(self.device),
                "curriculum_scale": self.config.curriculum_scale,
                "batch_size": self.config.batch_size,
                "render_workers": self.config.render_workers,
                "warm_start_checkpoint": self.config.warm_start_checkpoint,
            },
            force=True,
        )
        try:
            if self.config.render_workers > 1:
                self.render_executor = ProcessPoolExecutor(
                    max_workers=self.config.render_workers
                )
            plan = (
                CurriculumPlanV2()
                .scale_examples(self.config.curriculum_scale)
                .scale_epochs(self.config.epoch_multiplier)
            )
            examples_per_second = self.calibrate()
            self.progress.emit(
                "training_calibrated",
                {"examples_per_second": round(examples_per_second, 3)},
                force=True,
            )
            stages = plan.trim_for_hours(examples_per_second, self.config.max_hours)
            if self.config.smoke_examples > 0:
                stages = (
                    CurriculumStageV2("single_note_all", self.config.smoke_examples, 1, 3e-4),
                )
            self.progress.emit(
                "training_plan",
                {
                    "stages": [asdict(stage) for stage in stages],
                    "projected_hours": round(
                        CurriculumPlanV2._project_hours(stages, examples_per_second), 3
                    ),
                },
                force=True,
            )
            run_dir = self._run_dir()
            history = []
            for stage in stages:
                history.extend(self._train_stage(stage))
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "checkpoint.pt"
            artifacts = {
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "checkpoint_path": str(checkpoint_path),
            }
            payload = {
                "config": asdict(self.config),
                "examples_per_second": examples_per_second,
                "stages": [asdict(stage) for stage in stages],
                "history": history,
                "artifacts": artifacts,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            torch.save(self.model.state_dict(), checkpoint_path)
            self.progress.emit("training_done", artifacts, force=True)
            print(
                json.dumps(
                    {"artifacts": artifacts, "history": history[-5:]},
                    sort_keys=True,
                )
            )
            return payload
        finally:
            if self.render_executor is not None:
                self.render_executor.shutdown(wait=True)

    def calibrate(self) -> float:
        start = time.perf_counter()
        count = max(1, min(self.config.calibration_examples, 128))
        for idx in range(0, count, self.config.batch_size):
            seeds = [
                self.config.seed + row
                for row in range(idx, min(idx + self.config.batch_size, count))
            ]
            self._render_slices("single_note_all", "train", seeds)
        elapsed = max(time.perf_counter() - start, 1e-6)
        return count / elapsed

    def _train_stage(self, stage: CurriculumStageV2) -> list[dict[str, object]]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=stage.learning_rate)
        rows = []
        batches_per_epoch = max(
            1, int(np.ceil(stage.train_examples_per_epoch / self.config.batch_size))
        )
        self.progress.emit(
            "training_stage_start",
            {
                "stage": stage.name,
                "epochs": stage.epochs,
                "batches_per_epoch": batches_per_epoch,
                "examples_per_epoch": stage.train_examples_per_epoch,
            },
            force=True,
        )
        for epoch in range(stage.epochs):
            losses = []
            epoch_start = time.monotonic()
            for batch_idx in range(batches_per_epoch):
                seeds = [
                    self.config.seed
                    + epoch * 1_000_000
                    + batch_idx * self.config.batch_size
                    + row
                    for row in range(self.config.batch_size)
                ]
                batch = self.batch_builder.build_slices(
                    self._render_slices(stage.name, "train", seeds)
                )
                outputs = self.model(batch["frames"])
                loss = self.loss_fn(outputs, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                self.progress.emit(
                    "training_batch_progress",
                    {
                        "stage": stage.name,
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "batches_per_epoch": batches_per_epoch,
                        "last_loss": round(loss_value, 6),
                        "mean_loss": round(float(np.mean(losses)), 6),
                    },
                )
            row = {
                "stage": stage.name,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "examples": stage.train_examples_per_epoch,
                "elapsed_s": round(time.monotonic() - epoch_start, 3),
            }
            self.progress.emit("training_epoch_done", row, force=True)
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
        return rows

    def _render_slices(
        self, stage: str, split: SplitName, seeds: list[int]
    ) -> list[dict[str, np.ndarray]]:
        if self.render_executor is None:
            return [
                TrainingBatchBuilderV2.slice_from_example(
                    self.renderer.render(stage, split, seed)
                )
                for seed in seeds
            ]
        tasks = [(self.config.data_root, stage, split, seed) for seed in seeds]
        return list(self.render_executor.map(TrainingRenderWorkerV2.render_slice, tasks))

    def _build_renderer(self) -> SourceTrackingRendererV2:
        nsynth = NsynthIndexV2(Path(self.config.data_root))
        midi = MidiIndexV2(Path(self.config.data_root))
        note_store = NsynthNoteStoreV2(nsynth)
        midi_store = MidiScheduleStoreV2(midi)
        return SourceTrackingRendererV2(note_store, midi_store)

    def _load_warm_start(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location=self.device)
        state_dict = (
            payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        )
        self.model.load_state_dict(state_dict)

    def _run_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return Path(self.config.run_root) / stamp

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class TrainingCliV2:
    """Command-line interface for source-tracking training."""

    @staticmethod
    def parse(argv: Sequence[str] | None = None) -> TrainingConfigV2:
        parser = argparse.ArgumentParser(description="Train muziq-nn source tracker.")
        parser.add_argument("--data-root", default=TrainingConfigV2.data_root)
        parser.add_argument("--run-root", default=TrainingConfigV2.run_root)
        parser.add_argument("--curriculum", default=TrainingConfigV2.curriculum)
        parser.add_argument(
            "--curriculum-scale",
            type=float,
            default=TrainingConfigV2.curriculum_scale,
        )
        parser.add_argument("--max-hours", type=float, default=TrainingConfigV2.max_hours)
        parser.add_argument("--generate-on-the-fly", action="store_true")
        parser.add_argument(
            "--calibration-examples", type=int, default=TrainingConfigV2.calibration_examples
        )
        parser.add_argument("--batch-size", type=int, default=TrainingConfigV2.batch_size)
        parser.add_argument("--seed", type=int, default=TrainingConfigV2.seed)
        parser.add_argument("--device", default=TrainingConfigV2.device)
        parser.add_argument(
            "--smoke-examples", type=int, default=TrainingConfigV2.smoke_examples
        )
        parser.add_argument(
            "--progress-interval-s",
            type=float,
            default=TrainingConfigV2.progress_interval_s,
        )
        parser.add_argument(
            "--render-workers",
            type=int,
            default=TrainingConfigV2.render_workers,
        )
        parser.add_argument("--warm-start-checkpoint")
        parser.add_argument(
            "--epoch-multiplier",
            type=float,
            default=TrainingConfigV2.epoch_multiplier,
        )
        args = parser.parse_args(argv)
        values = vars(args)
        if not values["generate_on_the_fly"]:
            values["generate_on_the_fly"] = True
        return TrainingConfigV2(**values)

    @staticmethod
    def main(argv: Sequence[str] | None = None) -> int:
        SourceTrackingTrainerV2(TrainingCliV2.parse(argv)).run()
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    return TrainingCliV2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
