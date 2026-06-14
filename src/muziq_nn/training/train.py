"""Curriculum trainer for on-the-fly source tracking."""

from __future__ import annotations

import argparse
import importlib
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from muziq_nn.datasets.midi import MidiIndexV2
from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    AudioFrameExtractorV2,
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
    checkpoint_upload_uri: str | None = None
    checkpoint_upload_run_id: str | None = None
    checkpoint_interval_batches: int = 100
    resume_from_warm_start_metadata: bool = False
    resume_optimizer_state: bool = True
    training_slice_frames: int = 256
    training_slice_peak_warmup_frames: int = 512
    gpu_frame_extraction: bool = False
    pin_memory: bool = False
    prefetch_batches: int = 0
    mixed_precision: bool = False
    partial_warm_start: bool = False
    model_dim: int = 128
    model_heads: int = 4
    model_layers: int = 2
    identity_dim: int = 16


@dataclass(frozen=True)
class TrainingResumeCursorV2:
    stage: str
    epoch: int
    batch: int
    checkpoint_number: int


class TrainingBatchBuilderV2:
    """Convert rendered frame endpoints into torch training batches."""

    def __init__(
        self,
        device: torch.device,
        frame_count: int = 256,
        pin_memory: bool = False,
    ):
        self.device = device
        self.frame_count = frame_count
        self.pin_memory = pin_memory and device.type == "cuda"
        extractor = AudioFrameExtractorV2()
        self.frame_window = torch.from_numpy(extractor._window).to(device)
        self.frame_fold = torch.from_numpy(extractor._fold.T).to(device)

    def build(self, examples) -> dict[str, torch.Tensor]:
        return self.build_slices([self.slice_from_example(example) for example in examples])

    def build_slices(self, slices) -> dict[str, torch.Tensor]:
        activity = []
        family = []
        onset = []
        offset = []
        for item in slices:
            activity.append(item["activity"])
            family.append(np.maximum(item["family"], 0))
            onset.append(item["onset"])
            offset.append(item["offset"])
        return {
            "frames": self._build_frame_tensor(slices),
            "activity": self._to_device(np.asarray(activity, dtype=np.float32)),
            "family": self._to_device(np.asarray(family, dtype=np.int64)),
            "onset": self._to_device(np.asarray(onset, dtype=np.float32)),
            "offset": self._to_device(np.asarray(offset, dtype=np.float32)),
        }

    def _build_frame_tensor(self, slices) -> torch.Tensor:
        if slices and "audio_context" in slices[0]:
            return self._build_frame_tensor_from_audio(slices)
        frames = [item["frames"] for item in slices]
        max_len = max(frame.shape[0] for frame in frames)
        padded = np.zeros(
            (len(frames), max_len, SourceTrackingAudioConfigV2.bands), dtype=np.float32
        )
        for idx, frame in enumerate(frames):
            padded[idx, -frame.shape[0] :] = frame
        return self._to_device(padded)

    def _build_frame_tensor_from_audio(self, slices) -> torch.Tensor:
        audio = np.stack([item["audio_context"] for item in slices]).astype(np.float32)
        audio_tensor = self._to_device(audio)
        windows = audio_tensor.unfold(
            dimension=1,
            size=SourceTrackingAudioConfigV2.win,
            step=SourceTrackingAudioConfigV2.hop,
        )
        mag = torch.fft.rfft(windows * self.frame_window, dim=-1).abs()
        raw_bands = torch.matmul(mag.float(), self.frame_fold)
        return self._normalize_bands(raw_bands)[:, -self.frame_count :, :]

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if self.pin_memory:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.pin_memory)

    @staticmethod
    def _normalize_bands(raw_bands: torch.Tensor) -> torch.Tensor:
        frame_count = raw_bands.shape[1]
        if frame_count == 0:
            return raw_bands
        powers = torch.pow(
            torch.tensor(0.9996, dtype=raw_bands.dtype, device=raw_bands.device),
            torch.arange(frame_count, dtype=raw_bands.dtype, device=raw_bands.device),
        ).view(1, frame_count, 1)
        peak = torch.cummax(raw_bands / powers, dim=1).values * powers
        return torch.clamp(raw_bands / torch.clamp_min(peak, 1e-6), 0.0, 1.0)

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

    _renderers: dict[tuple[str, bool], SourceTrackingRendererV2] = {}

    @staticmethod
    def render_slice(
        task: tuple[str, str, SplitName, int, int, int, bool],
    ) -> dict[str, np.ndarray]:
        (
            data_root,
            stage,
            split,
            seed,
            frame_count,
            peak_warmup_frames,
            gpu_frame_extraction,
        ) = task
        renderer = TrainingRenderWorkerV2._renderer(
            data_root, include_midi=stage == "midi_complex"
        )
        if gpu_frame_extraction:
            return renderer.render_training_audio_slice(
                stage,
                split,
                seed,
                frame_count=frame_count,
                peak_warmup_frames=peak_warmup_frames,
            )
        return renderer.render_training_slice(
            stage,
            split,
            seed,
            frame_count=frame_count,
            peak_warmup_frames=peak_warmup_frames,
        )

    @staticmethod
    def _renderer(data_root: str, include_midi: bool) -> SourceTrackingRendererV2:
        cache_key = (data_root, include_midi)
        renderer = TrainingRenderWorkerV2._renderers.get(cache_key)
        if renderer is not None:
            return renderer
        nsynth = NsynthIndexV2(Path(data_root))
        midi_store = MidiScheduleStoreV2(MidiIndexV2(Path(data_root))) if include_midi else None
        renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(nsynth),
            midi_store,
        )
        TrainingRenderWorkerV2._renderers[cache_key] = renderer
        return renderer


@dataclass(frozen=True)
class CheckpointMetadataV2:
    checkpoint_number: int
    phase: str
    stage: str
    epoch: int
    batch: int
    batches_per_epoch: int
    examples_seen: int
    loss: float | None
    mean_loss: float | None
    run_id: str
    created_at_utc: str
    local_checkpoint_path: str
    optimizer_state_included: bool
    optimizer_class: str | None


@dataclass(frozen=True)
class CheckpointUploadTaskV2:
    files: tuple[tuple[Path, str], ...]
    checkpoint_relative_path: str
    event_payload: dict[str, Any]


class CheckpointArtifactUploaderV2:
    """Upload numbered checkpoint artifacts to GCS or a local test directory."""

    def __init__(self, base_uri: str | None, run_id: str, date_stamp: str):
        self.base_uri = base_uri.rstrip("/") if base_uri else None
        self.run_id = self._safe_component(run_id)
        self.date_stamp = self._safe_component(date_stamp)

    def remote_run_uri(self) -> str | None:
        if self.base_uri is None:
            return None
        return f"{self.base_uri}/{self.date_stamp}/{self.run_id}"

    def remote_uri(self, relative_path: str) -> str | None:
        if self.base_uri is None:
            return None
        return f"{self.remote_run_uri()}/{relative_path}"

    def upload(self, local_path: Path, relative_path: str) -> str | None:
        if self.base_uri is None:
            return None
        remote_uri = self.remote_uri(relative_path)
        if remote_uri is None:
            return None
        if self.base_uri.startswith("gs://"):
            self._upload_gcs(local_path, remote_uri)
        else:
            self._copy_local(local_path, remote_uri)
        return remote_uri

    def _upload_gcs(self, local_path: Path, remote_uri: str) -> None:
        if self._upload_gcs_with_python(local_path, remote_uri):
            return
        gsutil = shutil.which("gsutil")
        if gsutil is not None:
            subprocess.run([gsutil, "-q", "cp", str(local_path), remote_uri], check=True)
            return
        gcloud = shutil.which("gcloud")
        if gcloud is not None:
            subprocess.run(
                [gcloud, "storage", "cp", str(local_path), remote_uri, "--quiet"],
                check=True,
            )
            return
        raise RuntimeError(
            "checkpoint upload requested for GCS, but google-cloud-storage, gsutil, "
            "and gcloud are unavailable"
        )

    def _upload_gcs_with_python(self, local_path: Path, remote_uri: str) -> bool:
        try:
            storage = importlib.import_module("google.cloud.storage")
        except ModuleNotFoundError:
            return False
        bucket_name, blob_name = self._split_gcs_uri(remote_uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        return True

    @staticmethod
    def _copy_local(local_path: Path, remote_uri: str) -> None:
        destination = Path(remote_uri.removeprefix("file://"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)

    @staticmethod
    def _split_gcs_uri(uri: str) -> tuple[str, str]:
        path = uri.removeprefix("gs://")
        bucket, _, blob = path.partition("/")
        if not bucket or not blob:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return bucket, blob

    @staticmethod
    def _safe_component(value: str) -> str:
        safe = []
        for char in value:
            if char.isalnum() or char in ("-", "_", "."):
                safe.append(char)
            else:
                safe.append("-")
        return "".join(safe).strip("-") or "run"


class CheckpointUploadWorkerV2:
    """Copy checkpoint artifacts without blocking model training on GCS I/O."""

    def __init__(
        self,
        uploader: CheckpointArtifactUploaderV2,
        progress: TrainingProgressLoggerV2,
    ):
        self.uploader = uploader
        self.progress = progress
        self.tasks: queue.Queue[CheckpointUploadTaskV2 | None] = queue.Queue()
        self.failures: list[str] = []
        self.thread = threading.Thread(
            target=self._run,
            name="checkpoint-upload-worker-v2",
            daemon=False,
        )
        self.thread.start()

    def enqueue(self, task: CheckpointUploadTaskV2) -> None:
        self.raise_if_failed()
        self.tasks.put(task)

    def close(self) -> None:
        self.tasks.put(None)
        self.thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self.failures:
            raise RuntimeError(self.failures[0])

    def _run(self) -> None:
        while True:
            task = self.tasks.get()
            try:
                if task is None:
                    return
                uploaded = {}
                for local_path, relative_path in task.files:
                    uploaded[relative_path] = self.uploader.upload(
                        local_path, relative_path
                    )
                self.progress.emit(
                    "checkpoint_uploaded",
                    {
                        **task.event_payload,
                        "remote_checkpoint_uri": uploaded.get(
                            task.checkpoint_relative_path
                        ),
                        "upload_queue_size": self.tasks.qsize(),
                    },
                    force=True,
                )
            except Exception as exc:
                message = f"checkpoint upload failed: {exc}"
                self.failures.append(message)
                self.progress.emit(
                    "checkpoint_upload_failed",
                    {"error": message},
                    force=True,
                )
            finally:
                self.tasks.task_done()


class SourceTrackingTrainerV2:
    """Train the compact attention model from generated examples."""

    def __init__(self, config: TrainingConfigV2):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.renderer = self._build_renderer(include_midi=False)
        self.midi_renderer: SourceTrackingRendererV2 | None = None
        self.warm_start_metadata: dict[str, object] | None = None
        self.warm_start_optimizer_state: dict[str, object] | None = None
        self.optimizer_resume_stage: str | None = None
        self.warm_start_load_report: dict[str, object] | None = None
        self.model_config = SourceTrackingModelConfigV2(
            model_dim=config.model_dim,
            heads=config.model_heads,
            layers=config.model_layers,
            identity_dim=config.identity_dim,
        )
        self.model = DualPathTransformerSourceTrackerV2(self.model_config).to(self.device)
        if config.warm_start_checkpoint is not None:
            self._load_warm_start(Path(config.warm_start_checkpoint))
        self.loss_fn = SourceTrackingLossV2()
        self.batch_builder = TrainingBatchBuilderV2(
            self.device,
            frame_count=config.training_slice_frames,
            pin_memory=config.pin_memory,
        )
        self.progress = TrainingProgressLoggerV2(config.progress_interval_s)
        self.render_executor: ProcessPoolExecutor | None = None
        self.checkpoint_number = 0
        self.checkpoint_uploader: CheckpointArtifactUploaderV2 | None = None
        self.checkpoint_upload_worker: CheckpointUploadWorkerV2 | None = None
        self.current_optimizer: torch.optim.Optimizer | None = None

    def run(self) -> dict[str, object]:
        self.progress.emit(
            "training_start",
            {
                "device": str(self.device),
                "curriculum_scale": self.config.curriculum_scale,
                "batch_size": self.config.batch_size,
                "render_workers": self.config.render_workers,
                "warm_start_checkpoint": self.config.warm_start_checkpoint,
                "checkpoint_upload_uri": self.config.checkpoint_upload_uri,
                "checkpoint_interval_batches": self.config.checkpoint_interval_batches,
                "resume_from_warm_start_metadata": (
                    self.config.resume_from_warm_start_metadata
                ),
                "resume_optimizer_state": self.config.resume_optimizer_state,
                "training_slice_frames": self.config.training_slice_frames,
                "training_slice_peak_warmup_frames": (
                    self.config.training_slice_peak_warmup_frames
                ),
                "gpu_frame_extraction": self.config.gpu_frame_extraction,
                "pin_memory": self.config.pin_memory,
                "prefetch_batches": self.config.prefetch_batches,
                "mixed_precision": self.config.mixed_precision,
                "partial_warm_start": self.config.partial_warm_start,
                "model_config": asdict(self.model_config),
                "warm_start_load_report": self.warm_start_load_report,
            },
            force=True,
        )
        if self.config.mixed_precision and self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
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
            resume_cursor = self._resume_cursor(stages)
            if resume_cursor is not None:
                stages = self._stages_from_resume(stages, resume_cursor)
                self.optimizer_resume_stage = resume_cursor.stage
                self.checkpoint_number = max(
                    self.checkpoint_number, resume_cursor.checkpoint_number
                )
                self.progress.emit(
                    "training_resume",
                    {
                        "stage": resume_cursor.stage,
                        "epoch": resume_cursor.epoch,
                        "batch": resume_cursor.batch,
                        "checkpoint_number": resume_cursor.checkpoint_number,
                    },
                    force=True,
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
            run_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_uploader = CheckpointArtifactUploaderV2(
                self.config.checkpoint_upload_uri,
                self.config.checkpoint_upload_run_id or run_dir.name,
                time.strftime("%Y%m%d"),
            )
            if self.checkpoint_uploader.base_uri is not None:
                self.checkpoint_upload_worker = CheckpointUploadWorkerV2(
                    self.checkpoint_uploader, self.progress
            )
            history = []
            for stage in stages:
                first_epoch = (
                    resume_cursor.epoch
                    if resume_cursor is not None and stage.name == resume_cursor.stage
                    else 1
                )
                first_batch = (
                    resume_cursor.batch
                    if resume_cursor is not None and stage.name == resume_cursor.stage
                    else 0
                )
                history.extend(
                    self._train_stage(stage, run_dir, first_epoch, first_batch)
                )
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "checkpoint.pt"
            artifacts = {
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_upload_uri": self.checkpoint_uploader.remote_run_uri()
                if self.checkpoint_uploader is not None
                else None,
            }
            payload = {
                "config": asdict(self.config),
                "examples_per_second": examples_per_second,
                "stages": [asdict(stage) for stage in stages],
                "history": history,
                "artifacts": artifacts,
            }
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._save_final_checkpoint(checkpoint_path, history)
            self._save_checkpoint_artifact(
                run_dir,
                phase="training_done",
                stage="complete",
                epoch=0,
                batch=0,
                batches_per_epoch=0,
                examples_seen=sum(row["examples"] for row in history),
                loss=history[-1]["loss"] if history else None,
                mean_loss=history[-1]["loss"] if history else None,
                optimizer=self.current_optimizer,
                force=True,
            )
            self._close_checkpoint_upload_worker()
            self.progress.emit("training_done", artifacts, force=True)
            print(
                json.dumps(
                    {"artifacts": artifacts, "history": history[-5:]},
                    sort_keys=True,
                )
            )
            return payload
        finally:
            self._close_checkpoint_upload_worker()
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

    def _train_stage(
        self,
        stage: CurriculumStageV2,
        run_dir: Path,
        first_epoch: int = 1,
        first_batch: int = 0,
    ) -> list[dict[str, object]]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=stage.learning_rate)
        self._restore_optimizer_state_if_needed(optimizer, stage.name)
        self.current_optimizer = optimizer
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
                "first_epoch": first_epoch,
                "first_batch": first_batch,
            },
            force=True,
        )
        for epoch in range(first_epoch - 1, stage.epochs):
            losses = []
            epoch_start = time.monotonic()
            batch_start = first_batch if epoch == first_epoch - 1 else 0
            for batch_idx, batch in self._iter_training_batches(
                stage.name,
                epoch,
                batch_start,
                batches_per_epoch,
            ):
                with self._autocast_context():
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
                self._save_checkpoint_artifact(
                    run_dir,
                    phase="batch_interval",
                    stage=stage.name,
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                    batches_per_epoch=batches_per_epoch,
                    examples_seen=(epoch * stage.train_examples_per_epoch)
                    + min(
                        stage.train_examples_per_epoch,
                        (batch_idx + 1) * self.config.batch_size,
                    ),
                    loss=loss_value,
                    mean_loss=float(np.mean(losses)),
                    optimizer=optimizer,
                )
            row = {
                "stage": stage.name,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "examples": max(
                    0,
                    stage.train_examples_per_epoch - batch_start * self.config.batch_size,
                ),
                "elapsed_s": round(time.monotonic() - epoch_start, 3),
            }
            self.progress.emit("training_epoch_done", row, force=True)
            print(json.dumps(row, sort_keys=True), flush=True)
            self._save_checkpoint_artifact(
                run_dir,
                phase="epoch_done",
                stage=stage.name,
                epoch=epoch + 1,
                batch=batches_per_epoch,
                batches_per_epoch=batches_per_epoch,
                examples_seen=(epoch + 1) * stage.train_examples_per_epoch,
                loss=row["loss"],
                mean_loss=row["loss"],
                optimizer=optimizer,
                force=True,
            )
            rows.append(row)
        return rows

    def _iter_training_batches(
        self,
        stage_name: str,
        epoch: int,
        batch_start: int,
        batches_per_epoch: int,
    ):
        if self.config.prefetch_batches <= 0:
            for batch_idx in range(batch_start, batches_per_epoch):
                yield batch_idx, self._build_training_batch(stage_name, epoch, batch_idx)
            return
        batch_indices = iter(range(batch_start, batches_per_epoch))
        pending: list[tuple[int, Future[dict[str, torch.Tensor]]]] = []
        with ThreadPoolExecutor(
            max_workers=self.config.prefetch_batches,
            thread_name_prefix="training-batch-prefetch-v2",
        ) as prefetch_executor:

            def enqueue_next() -> bool:
                try:
                    next_batch_idx = next(batch_indices)
                except StopIteration:
                    return False
                pending.append(
                    (
                        next_batch_idx,
                        prefetch_executor.submit(
                            self._build_training_batch,
                            stage_name,
                            epoch,
                            next_batch_idx,
                        ),
                    )
                )
                return True

            for _ in range(self.config.prefetch_batches):
                if not enqueue_next():
                    break
            while pending:
                batch_idx, future = pending.pop(0)
                enqueue_next()
                yield batch_idx, future.result()

    def _build_training_batch(
        self,
        stage_name: str,
        epoch: int,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        seeds = self._batch_seeds(epoch, batch_idx)
        return self.batch_builder.build_slices(
            self._render_slices(stage_name, "train", seeds)
        )

    def _batch_seeds(self, epoch: int, batch_idx: int) -> list[int]:
        return [
            self.config.seed
            + epoch * 1_000_000
            + batch_idx * self.config.batch_size
            + row
            for row in range(self.config.batch_size)
        ]

    def _autocast_context(self):
        if self.config.mixed_precision and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _save_checkpoint_artifact(
        self,
        run_dir: Path,
        *,
        phase: str,
        stage: str,
        epoch: int,
        batch: int,
        batches_per_epoch: int,
        examples_seen: int,
        loss: float | None,
        mean_loss: float | None,
        optimizer: torch.optim.Optimizer | None = None,
        force: bool = False,
    ) -> None:
        if self.checkpoint_uploader is None:
            return
        if self.checkpoint_uploader.base_uri is None:
            return
        if not force:
            interval = max(0, self.config.checkpoint_interval_batches)
            if interval == 0 or batch % interval != 0:
                return
        self.checkpoint_number += 1
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename_stem = self._checkpoint_filename_stem(
            self.checkpoint_number,
            phase,
            stage,
            epoch,
            batch,
        )
        checkpoint_path = checkpoint_dir / f"{filename_stem}.pt"
        optimizer_state = self._optimizer_state_dict(optimizer)
        metadata = CheckpointMetadataV2(
            checkpoint_number=self.checkpoint_number,
            phase=phase,
            stage=stage,
            epoch=epoch,
            batch=batch,
            batches_per_epoch=batches_per_epoch,
            examples_seen=examples_seen,
            loss=loss,
            mean_loss=mean_loss,
            run_id=self.checkpoint_uploader.run_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            local_checkpoint_path=str(checkpoint_path),
            optimizer_state_included=optimizer_state is not None,
            optimizer_class=type(optimizer).__name__ if optimizer is not None else None,
        )
        metadata_path = checkpoint_dir / f"{filename_stem}.json"
        latest_path = checkpoint_dir / "latest.json"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer_state,
                "checkpoint_metadata": asdict(metadata),
            },
            checkpoint_path,
        )
        metadata_payload = asdict(metadata)
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        checkpoint_relative_path = f"checkpoints/{checkpoint_path.name}"
        metadata_relative_path = f"checkpoints/{metadata_path.name}"
        latest_relative_path = "checkpoints/latest.json"
        latest_payload = {
            **metadata_payload,
            "remote_checkpoint_uri": self.checkpoint_uploader.remote_uri(
                checkpoint_relative_path
            ),
            "remote_metadata_uri": self.checkpoint_uploader.remote_uri(
                metadata_relative_path
            ),
        }
        latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")
        event_payload = {
            "checkpoint_number": self.checkpoint_number,
            "phase": phase,
            "stage": stage,
            "epoch": epoch,
            "batch": batch,
        }
        if self.checkpoint_upload_worker is not None:
            self.checkpoint_upload_worker.enqueue(
                CheckpointUploadTaskV2(
                    files=(
                        (checkpoint_path, checkpoint_relative_path),
                        (metadata_path, metadata_relative_path),
                        (latest_path, latest_relative_path),
                    ),
                    checkpoint_relative_path=checkpoint_relative_path,
                    event_payload=event_payload,
                )
            )
        self.progress.emit(
            "checkpoint_upload_queued",
            {
                **event_payload,
                "remote_checkpoint_uri": self.checkpoint_uploader.remote_uri(
                    checkpoint_relative_path
                ),
                "upload_queue_size": self.checkpoint_upload_worker.tasks.qsize()
                if self.checkpoint_upload_worker is not None
                else 0,
            },
            force=True,
        )

    def _close_checkpoint_upload_worker(self) -> None:
        if self.checkpoint_upload_worker is None:
            return
        worker = self.checkpoint_upload_worker
        self.checkpoint_upload_worker = None
        worker.close()

    def _save_final_checkpoint(
        self, checkpoint_path: Path, history: list[dict[str, object]]
    ) -> None:
        optimizer_state = self._optimizer_state_dict(self.current_optimizer)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer_state,
                "checkpoint_metadata": {
                    "phase": "training_done",
                    "stage": "complete",
                    "examples_seen": sum(row["examples"] for row in history),
                    "loss": history[-1]["loss"] if history else None,
                    "optimizer_state_included": optimizer_state is not None,
                    "optimizer_class": type(self.current_optimizer).__name__
                    if self.current_optimizer is not None
                    else None,
                },
            },
            checkpoint_path,
        )

    @staticmethod
    def _optimizer_state_dict(
        optimizer: torch.optim.Optimizer | None,
    ) -> dict[str, object] | None:
        if optimizer is None:
            return None
        return optimizer.state_dict()

    def _restore_optimizer_state_if_needed(
        self, optimizer: torch.optim.Optimizer, stage_name: str
    ) -> None:
        if self.warm_start_optimizer_state is None:
            return
        if self.config.partial_warm_start:
            self.progress.emit(
                "optimizer_state_skipped",
                {"stage": stage_name, "reason": "partial_warm_start"},
                force=True,
            )
            self.warm_start_optimizer_state = None
            return
        if not self.config.resume_from_warm_start_metadata:
            return
        if not self.config.resume_optimizer_state:
            return
        if self.optimizer_resume_stage != stage_name:
            return
        optimizer.load_state_dict(self.warm_start_optimizer_state)
        self.progress.emit(
            "optimizer_state_restored",
            {"stage": stage_name},
            force=True,
        )
        self.warm_start_optimizer_state = None

    @staticmethod
    def _checkpoint_filename_stem(
        checkpoint_number: int,
        phase: str,
        stage: str,
        epoch: int,
        batch: int,
    ) -> str:
        safe_phase = CheckpointArtifactUploaderV2._safe_component(phase)
        safe_stage = CheckpointArtifactUploaderV2._safe_component(stage)
        return (
            f"checkpoint_{checkpoint_number:06d}_phase-{safe_phase}"
            f"_stage-{safe_stage}_epoch-{epoch:03d}_batch-{batch:06d}"
        )

    def _render_slices(
        self, stage: str, split: SplitName, seeds: list[int]
    ) -> list[dict[str, np.ndarray]]:
        if self.render_executor is None:
            renderer = self._renderer_for_stage(stage)
            return [
                self._render_training_slice(
                    renderer,
                    stage,
                    split,
                    seed,
                )
                for seed in seeds
            ]
        tasks = [
            (
                self.config.data_root,
                stage,
                split,
                seed,
                self.config.training_slice_frames,
                self.config.training_slice_peak_warmup_frames,
                self.config.gpu_frame_extraction,
            )
            for seed in seeds
        ]
        return list(self.render_executor.map(TrainingRenderWorkerV2.render_slice, tasks))

    def _render_training_slice(
        self,
        renderer: SourceTrackingRendererV2,
        stage: str,
        split: SplitName,
        seed: int,
    ) -> dict[str, np.ndarray]:
        if self.config.gpu_frame_extraction:
            return renderer.render_training_audio_slice(
                stage,
                split,
                seed,
                frame_count=self.config.training_slice_frames,
                peak_warmup_frames=self.config.training_slice_peak_warmup_frames,
            )
        return renderer.render_training_slice(
            stage,
            split,
            seed,
            frame_count=self.config.training_slice_frames,
            peak_warmup_frames=self.config.training_slice_peak_warmup_frames,
        )

    def _renderer_for_stage(self, stage: str) -> SourceTrackingRendererV2:
        if stage != "midi_complex":
            return self.renderer
        if self.midi_renderer is None:
            self.midi_renderer = self._build_renderer(include_midi=True)
        return self.midi_renderer

    def _build_renderer(self, include_midi: bool) -> SourceTrackingRendererV2:
        nsynth = NsynthIndexV2(Path(self.config.data_root))
        note_store = NsynthNoteStoreV2(nsynth)
        midi_store = (
            MidiScheduleStoreV2(MidiIndexV2(Path(self.config.data_root)))
            if include_midi
            else None
        )
        return SourceTrackingRendererV2(note_store, midi_store)

    def _resume_cursor(
        self, stages: tuple[CurriculumStageV2, ...]
    ) -> TrainingResumeCursorV2 | None:
        if not self.config.resume_from_warm_start_metadata:
            return None
        if self.warm_start_metadata is None:
            raise ValueError(
                "resume_from_warm_start_metadata requires checkpoint_metadata in the "
                "warm-start checkpoint"
            )
        metadata = self.warm_start_metadata
        stage_name = str(metadata["stage"])
        checkpoint_number = int(metadata.get("checkpoint_number", 0))
        if stage_name == "complete":
            return None
        stage_index = self._stage_index(stages, stage_name)
        if "examples_seen" in metadata:
            return self._resume_cursor_from_examples_seen(
                stages,
                stage_index,
                int(metadata["examples_seen"]),
                checkpoint_number,
            )
        epoch = int(metadata["epoch"])
        batch = int(metadata["batch"])
        phase = str(metadata.get("phase", ""))
        batches_per_epoch = int(
            metadata.get("batches_per_epoch")
            or np.ceil(stages[stage_index].train_examples_per_epoch / self.config.batch_size)
        )
        if phase == "epoch_done" or batch >= batches_per_epoch:
            epoch += 1
            batch = 0
        if epoch > stages[stage_index].epochs:
            stage_index += 1
            if stage_index >= len(stages):
                return None
            stage_name = stages[stage_index].name
            epoch = 1
            batch = 0
        return TrainingResumeCursorV2(
            stage=stage_name,
            epoch=epoch,
            batch=batch,
            checkpoint_number=checkpoint_number,
        )

    def _resume_cursor_from_examples_seen(
        self,
        stages: tuple[CurriculumStageV2, ...],
        stage_index: int,
        examples_seen: int,
        checkpoint_number: int,
    ) -> TrainingResumeCursorV2 | None:
        examples_seen = max(0, examples_seen)
        while stage_index < len(stages):
            stage = stages[stage_index]
            stage_examples = stage.train_examples_per_epoch * stage.epochs
            if examples_seen < stage_examples:
                completed_epochs, examples_in_epoch = divmod(
                    examples_seen,
                    stage.train_examples_per_epoch,
                )
                return TrainingResumeCursorV2(
                    stage=stage.name,
                    epoch=completed_epochs + 1,
                    batch=examples_in_epoch // self.config.batch_size,
                    checkpoint_number=checkpoint_number,
                )
            examples_seen -= stage_examples
            stage_index += 1
        return None

    @staticmethod
    def _stages_from_resume(
        stages: tuple[CurriculumStageV2, ...], cursor: TrainingResumeCursorV2
    ) -> tuple[CurriculumStageV2, ...]:
        return stages[SourceTrackingTrainerV2._stage_index(stages, cursor.stage) :]

    @staticmethod
    def _stage_index(stages: tuple[CurriculumStageV2, ...], stage_name: str) -> int:
        for idx, stage in enumerate(stages):
            if stage.name == stage_name:
                return idx
        raise ValueError(f"Resume stage {stage_name!r} is not in the training plan")

    def _load_warm_start(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location=self.device)
        if isinstance(payload, dict):
            metadata = payload.get("checkpoint_metadata")
            if isinstance(metadata, dict):
                self.warm_start_metadata = metadata
            optimizer_state = payload.get("optimizer_state_dict")
            if isinstance(optimizer_state, dict):
                self.warm_start_optimizer_state = optimizer_state
        state_dict = (
            payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        )
        if self.config.partial_warm_start:
            self._load_partial_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
            self.warm_start_load_report = {
                "mode": "strict",
                "loaded_tensors": len(state_dict),
            }

    def _load_partial_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        model_state = self.model.state_dict()
        matched = {
            name: value
            for name, value in state_dict.items()
            if name in model_state and model_state[name].shape == value.shape
        }
        skipped = sorted(set(state_dict) - set(matched))
        result = self.model.load_state_dict(matched, strict=False)
        self.warm_start_load_report = {
            "mode": "partial",
            "loaded_tensors": len(matched),
            "skipped_tensors": len(skipped),
            "missing_tensors": len(result.missing_keys),
            "unexpected_tensors": len(result.unexpected_keys),
        }

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
        parser.add_argument("--checkpoint-upload-uri")
        parser.add_argument("--checkpoint-upload-run-id")
        parser.add_argument(
            "--checkpoint-interval-batches",
            type=int,
            default=TrainingConfigV2.checkpoint_interval_batches,
        )
        parser.add_argument("--resume-from-warm-start-metadata", action="store_true")
        parser.add_argument(
            "--no-resume-optimizer-state",
            dest="resume_optimizer_state",
            action="store_false",
            help="Do not restore optimizer state from warm-start checkpoints.",
        )
        parser.add_argument(
            "--training-slice-frames",
            type=int,
            default=TrainingConfigV2.training_slice_frames,
        )
        parser.add_argument(
            "--training-slice-peak-warmup-frames",
            type=int,
            default=TrainingConfigV2.training_slice_peak_warmup_frames,
        )
        parser.add_argument("--gpu-frame-extraction", action="store_true")
        parser.add_argument("--pin-memory", action="store_true")
        parser.add_argument(
            "--prefetch-batches",
            type=int,
            default=TrainingConfigV2.prefetch_batches,
        )
        parser.add_argument("--mixed-precision", action="store_true")
        parser.add_argument("--partial-warm-start", action="store_true")
        parser.add_argument(
            "--model-dim",
            type=int,
            default=TrainingConfigV2.model_dim,
        )
        parser.add_argument(
            "--model-heads",
            type=int,
            default=TrainingConfigV2.model_heads,
        )
        parser.add_argument(
            "--model-layers",
            type=int,
            default=TrainingConfigV2.model_layers,
        )
        parser.add_argument(
            "--identity-dim",
            type=int,
            default=TrainingConfigV2.identity_dim,
        )
        parser.set_defaults(
            resume_optimizer_state=TrainingConfigV2.resume_optimizer_state
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
