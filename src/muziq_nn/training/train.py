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
    SourceTrackingEventStateBuilderV2,
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

    def with_epochs_per_stage(self, epochs: int | None) -> CurriculumPlanV2:
        if epochs is None:
            return self
        if epochs <= 0:
            raise ValueError("epochs per stage must be positive")
        return CurriculumPlanV2(
            tuple(
                CurriculumStageV2(
                    stage.name,
                    stage.train_examples_per_epoch,
                    epochs,
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
    stage_filter: str | None = None
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
    midi_render_workers: int | None = None
    warm_start_checkpoint: str | None = None
    epoch_multiplier: float = 1.0
    epochs_per_stage: int | None = None
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
    metric_activity_threshold: float = 0.35
    activity_pos_weight: float = 10.0
    inactive_slot_weight: float = 6.0
    family_loss_weight: float = 2.5
    count_loss_weight: float = 8.0
    onset_pos_weight: float = 10.0
    offset_pos_weight: float = 10.0
    boundary_loss_weight: float = 0.5
    onset_loss_weight: float | None = None
    offset_loss_weight: float | None = None
    onset_focal_gamma: float = 0.0
    offset_focal_gamma: float = 0.0
    boundary_timing_loss_weight: float = 0.25
    boundary_f1_loss_weight: float = 0.0
    boundary_f1_fp_weight: float = 1.0
    boundary_f1_fn_weight: float = 1.0
    hard_boundary_negative_loss_weight: float = 0.0
    hard_boundary_negative_fraction: float = 0.1
    onset_peak_loss_weight: float = 0.25
    onset_event_recall_loss_weight: float = 1.0
    onset_false_peak_loss_weight: float = 0.5
    onset_peak_radius_frames: int = 2
    onset_false_peak_fraction: float = 0.02
    model_dim: int = 128
    model_heads: int = 4
    model_layers: int = 2
    event_decoder_layers: int = 1
    event_decoder_heads: int = 4
    event_teacher_forcing_start: float = 0.9
    event_teacher_forcing_end: float = 0.25
    identity_dim: int = 16
    frame_cache_examples_per_stage: int = 0
    frame_cache_build_batch_size: int = 512
    frame_cache_dtype: str = "float16"
    frame_cache_phase_jitter_samples: int = 0
    frame_phase_noise_std: float = 0.0
    anneal_noise_phase_jitter_samples: int = 0
    anneal_noise_feature_std: float = 0.0


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
        onset_delta = []
        offset_delta = []
        onset_timing_mask = []
        offset_timing_mask = []
        context_activity = []
        context_family = []
        context_onset = []
        context_offset = []
        context_onset_delta = []
        context_offset_delta = []
        context_onset_timing_mask = []
        context_offset_timing_mask = []
        event_state = []
        for item in slices:
            activity.append(item["activity"])
            family.append(np.maximum(item["family"], 0))
            onset.append(item["onset"])
            offset.append(item["offset"])
            onset_delta.append(item["onset_delta"])
            offset_delta.append(item["offset_delta"])
            onset_timing_mask.append(item["onset_timing_mask"])
            offset_timing_mask.append(item["offset_timing_mask"])
            context_activity.append(item["context_activity"])
            context_family.append(np.maximum(item["context_family"], 0))
            context_onset.append(item["context_onset"])
            context_offset.append(item["context_offset"])
            context_onset_delta.append(item["context_onset_delta"])
            context_offset_delta.append(item["context_offset_delta"])
            context_onset_timing_mask.append(item["context_onset_timing_mask"])
            context_offset_timing_mask.append(item["context_offset_timing_mask"])
            event_state.append(item["event_state"])
        return {
            "frames": self._build_frame_tensor(slices),
            "activity": self._to_device(np.asarray(activity, dtype=np.float32)),
            "family": self._to_device(np.asarray(family, dtype=np.int64)),
            "onset": self._to_device(np.asarray(onset, dtype=np.float32)),
            "offset": self._to_device(np.asarray(offset, dtype=np.float32)),
            "onset_delta": self._to_device(np.asarray(onset_delta, dtype=np.float32)),
            "offset_delta": self._to_device(np.asarray(offset_delta, dtype=np.float32)),
            "onset_timing_mask": self._to_device(
                np.asarray(onset_timing_mask, dtype=np.float32)
            ),
            "offset_timing_mask": self._to_device(
                np.asarray(offset_timing_mask, dtype=np.float32)
            ),
            "context_activity": self._to_device(
                np.asarray(context_activity, dtype=np.float32)
            ),
            "context_family": self._to_device(np.asarray(context_family, dtype=np.int64)),
            "context_onset": self._to_device(np.asarray(context_onset, dtype=np.float32)),
            "context_offset": self._to_device(np.asarray(context_offset, dtype=np.float32)),
            "context_onset_delta": self._to_device(
                np.asarray(context_onset_delta, dtype=np.float32)
            ),
            "context_offset_delta": self._to_device(
                np.asarray(context_offset_delta, dtype=np.float32)
            ),
            "context_onset_timing_mask": self._to_device(
                np.asarray(context_onset_timing_mask, dtype=np.float32)
            ),
            "context_offset_timing_mask": self._to_device(
                np.asarray(context_offset_timing_mask, dtype=np.float32)
            ),
            "event_state": self._to_device(np.asarray(event_state, dtype=np.float32)),
        }

    def _build_frame_tensor(self, slices) -> torch.Tensor:
        if slices and "audio_context" in slices[0]:
            return self._build_frame_tensor_from_audio(slices)
        frames = [item["frames"] for item in slices]
        padded = np.zeros(
            (len(frames), self.frame_count, SourceTrackingAudioConfigV2.bands),
            dtype=np.float32,
        )
        for idx, frame in enumerate(frames):
            clipped = frame[-self.frame_count :]
            padded[idx, -clipped.shape[0] :] = clipped
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
        log_bands = torch.log1p(raw_bands * SourceTrackingAudioConfigV2.log_feature_scale)
        powers = torch.pow(
            torch.tensor(0.9996, dtype=log_bands.dtype, device=log_bands.device),
            torch.arange(frame_count, dtype=log_bands.dtype, device=log_bands.device),
        ).view(1, frame_count, 1)
        frame_peak = torch.amax(log_bands, dim=2, keepdim=True).clamp_min(1e-6)
        peak = torch.cummax(frame_peak / powers, dim=1).values * powers
        return torch.clamp(log_bands / torch.clamp_min(peak, 1e-6), 0.0, 1.0)

    @classmethod
    def slice_from_example(cls, example) -> dict[str, np.ndarray]:
        frame_idx = cls._sample_active_frame(example)
        return {
            "frames": example.frames[max(0, frame_idx - 255) : frame_idx + 1],
            "activity": example.labels.active[frame_idx],
            "family": example.labels.family[frame_idx],
            "onset": example.labels.onset[frame_idx],
            "offset": example.labels.offset[frame_idx],
            "onset_delta": example.labels.onset_delta[frame_idx],
            "offset_delta": example.labels.offset_delta[frame_idx],
            "onset_timing_mask": example.labels.onset_timing_mask[frame_idx],
            "offset_timing_mask": example.labels.offset_timing_mask[frame_idx],
        }

    @staticmethod
    def _sample_active_frame(example) -> int:
        active = np.where(example.labels.active.any(axis=1))[0]
        if len(active) == 0:
            return int(example.frames.shape[0] - 1)
        return int(active[len(active) // 2])


class CachedFramePoolV2:
    """In-memory shuffled feature/label pool for frame-cached training stages."""

    def __init__(
        self,
        examples: int,
        frame_count: int,
        frame_dtype: str,
    ):
        dtype = np.float16 if frame_dtype == "float16" else np.float32
        if frame_dtype not in {"float16", "float32"}:
            raise ValueError("frame_cache_dtype must be float16 or float32")
        self.frames = np.empty(
            (examples, frame_count, SourceTrackingAudioConfigV2.bands),
            dtype=dtype,
        )
        self.activity = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.float32,
        )
        self.family = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.int64,
        )
        self.onset = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.float32,
        )
        self.offset = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.float32,
        )
        self.onset_delta = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.float32,
        )
        self.offset_delta = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.float32,
        )
        self.onset_timing_mask = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.float32,
        )
        self.offset_timing_mask = np.empty(
            (examples, SourceTrackingAudioConfigV2.max_sources),
            dtype=np.float32,
        )
        context_shape = (
            examples,
            frame_count,
            SourceTrackingAudioConfigV2.max_sources,
        )
        self.context_activity = np.empty(context_shape, dtype=np.float32)
        self.context_family = np.empty(context_shape, dtype=np.int64)
        self.context_onset = np.empty(context_shape, dtype=np.float32)
        self.context_offset = np.empty(context_shape, dtype=np.float32)
        self.context_onset_delta = np.empty(context_shape, dtype=np.float32)
        self.context_offset_delta = np.empty(context_shape, dtype=np.float32)
        self.context_onset_timing_mask = np.empty(context_shape, dtype=np.float32)
        self.context_offset_timing_mask = np.empty(context_shape, dtype=np.float32)
        self.event_state = np.empty((*context_shape, 5), dtype=np.float32)

    def write(self, start: int, batch: dict[str, torch.Tensor]) -> None:
        stop = start + int(batch["frames"].shape[0])
        self.frames[start:stop] = batch["frames"].detach().cpu().numpy().astype(
            self.frames.dtype, copy=False
        )
        self.activity[start:stop] = batch["activity"].detach().cpu().numpy()
        self.family[start:stop] = batch["family"].detach().cpu().numpy()
        self.onset[start:stop] = batch["onset"].detach().cpu().numpy()
        self.offset[start:stop] = batch["offset"].detach().cpu().numpy()
        self.onset_delta[start:stop] = batch["onset_delta"].detach().cpu().numpy()
        self.offset_delta[start:stop] = batch["offset_delta"].detach().cpu().numpy()
        self.onset_timing_mask[start:stop] = (
            batch["onset_timing_mask"].detach().cpu().numpy()
        )
        self.offset_timing_mask[start:stop] = (
            batch["offset_timing_mask"].detach().cpu().numpy()
        )
        self.context_activity[start:stop] = (
            batch["context_activity"].detach().cpu().numpy()
        )
        self.context_family[start:stop] = batch["context_family"].detach().cpu().numpy()
        self.context_onset[start:stop] = batch["context_onset"].detach().cpu().numpy()
        self.context_offset[start:stop] = batch["context_offset"].detach().cpu().numpy()
        self.context_onset_delta[start:stop] = (
            batch["context_onset_delta"].detach().cpu().numpy()
        )
        self.context_offset_delta[start:stop] = (
            batch["context_offset_delta"].detach().cpu().numpy()
        )
        self.context_onset_timing_mask[start:stop] = (
            batch["context_onset_timing_mask"].detach().cpu().numpy()
        )
        self.context_offset_timing_mask[start:stop] = (
            batch["context_offset_timing_mask"].detach().cpu().numpy()
        )
        self.event_state[start:stop] = batch["event_state"].detach().cpu().numpy()

    def write_slices(
        self,
        start: int,
        slices: list[dict[str, np.ndarray]],
        frame_count: int,
    ) -> None:
        stop = start + len(slices)
        padded = np.zeros(
            (len(slices), frame_count, SourceTrackingAudioConfigV2.bands),
            dtype=self.frames.dtype,
        )
        for idx, item in enumerate(slices):
            clipped = item["frames"][-frame_count:]
            padded[idx, -clipped.shape[0] :] = clipped.astype(
                self.frames.dtype, copy=False
            )
        self.frames[start:stop] = padded
        self.activity[start:stop] = np.asarray(
            [item["activity"] for item in slices], dtype=np.float32
        )
        self.family[start:stop] = np.asarray(
            [np.maximum(item["family"], 0) for item in slices], dtype=np.int64
        )
        self.onset[start:stop] = np.asarray(
            [item["onset"] for item in slices], dtype=np.float32
        )
        self.offset[start:stop] = np.asarray(
            [item["offset"] for item in slices], dtype=np.float32
        )
        self.onset_delta[start:stop] = np.asarray(
            [item["onset_delta"] for item in slices], dtype=np.float32
        )
        self.offset_delta[start:stop] = np.asarray(
            [item["offset_delta"] for item in slices], dtype=np.float32
        )
        self.onset_timing_mask[start:stop] = np.asarray(
            [item["onset_timing_mask"] for item in slices], dtype=np.float32
        )
        self.offset_timing_mask[start:stop] = np.asarray(
            [item["offset_timing_mask"] for item in slices], dtype=np.float32
        )
        self.context_activity[start:stop] = np.asarray(
            [item["context_activity"] for item in slices], dtype=np.float32
        )
        self.context_family[start:stop] = np.asarray(
            [np.maximum(item["context_family"], 0) for item in slices],
            dtype=np.int64,
        )
        self.context_onset[start:stop] = np.asarray(
            [item["context_onset"] for item in slices], dtype=np.float32
        )
        self.context_offset[start:stop] = np.asarray(
            [item["context_offset"] for item in slices], dtype=np.float32
        )
        self.context_onset_delta[start:stop] = np.asarray(
            [item["context_onset_delta"] for item in slices], dtype=np.float32
        )
        self.context_offset_delta[start:stop] = np.asarray(
            [item["context_offset_delta"] for item in slices], dtype=np.float32
        )
        self.context_onset_timing_mask[start:stop] = np.asarray(
            [item["context_onset_timing_mask"] for item in slices], dtype=np.float32
        )
        self.context_offset_timing_mask[start:stop] = np.asarray(
            [item["context_offset_timing_mask"] for item in slices], dtype=np.float32
        )
        self.event_state[start:stop] = np.asarray(
            [item["event_state"] for item in slices], dtype=np.float32
        )

    def batch(
        self,
        indices: np.ndarray,
        device: torch.device,
        *,
        pin_memory: bool,
        phase_noise_std: float,
    ) -> dict[str, torch.Tensor]:
        frames = self._to_device(self.frames[indices], device, pin_memory).float()
        if phase_noise_std > 0.0:
            frames = torch.clamp(
                frames + torch.randn_like(frames) * phase_noise_std,
                0.0,
                1.0,
            )
        return {
            "frames": frames,
            "activity": self._to_device(
                self.activity[indices], device, pin_memory
            ).float(),
            "family": self._to_device(self.family[indices], device, pin_memory).long(),
            "onset": self._to_device(self.onset[indices], device, pin_memory).float(),
            "offset": self._to_device(self.offset[indices], device, pin_memory).float(),
            "onset_delta": self._to_device(
                self.onset_delta[indices], device, pin_memory
            ).float(),
            "offset_delta": self._to_device(
                self.offset_delta[indices], device, pin_memory
            ).float(),
            "onset_timing_mask": self._to_device(
                self.onset_timing_mask[indices], device, pin_memory
            ).float(),
            "offset_timing_mask": self._to_device(
                self.offset_timing_mask[indices], device, pin_memory
            ).float(),
            "context_activity": self._to_device(
                self.context_activity[indices], device, pin_memory
            ).float(),
            "context_family": self._to_device(
                self.context_family[indices], device, pin_memory
            ).long(),
            "context_onset": self._to_device(
                self.context_onset[indices], device, pin_memory
            ).float(),
            "context_offset": self._to_device(
                self.context_offset[indices], device, pin_memory
            ).float(),
            "context_onset_delta": self._to_device(
                self.context_onset_delta[indices], device, pin_memory
            ).float(),
            "context_offset_delta": self._to_device(
                self.context_offset_delta[indices], device, pin_memory
            ).float(),
            "context_onset_timing_mask": self._to_device(
                self.context_onset_timing_mask[indices], device, pin_memory
            ).float(),
            "context_offset_timing_mask": self._to_device(
                self.context_offset_timing_mask[indices], device, pin_memory
            ).float(),
            "event_state": self._to_device(
                self.event_state[indices], device, pin_memory
            ).float(),
        }

    @staticmethod
    def _to_device(
        array: np.ndarray,
        device: torch.device,
        pin_memory: bool,
    ) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if pin_memory and device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=pin_memory and device.type == "cuda")


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
        task: tuple[str, str, SplitName, int, int, int, bool, int],
    ) -> dict[str, np.ndarray]:
        (
            data_root,
            stage,
            split,
            seed,
            frame_count,
            peak_warmup_frames,
            gpu_frame_extraction,
            phase_offset_samples,
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
                phase_offset_samples=phase_offset_samples,
            )
        return renderer.render_training_slice(
            stage,
            split,
            seed,
            frame_count=frame_count,
            peak_warmup_frames=peak_warmup_frames,
            phase_offset_samples=phase_offset_samples,
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
                for local_path, _ in task.files:
                    if local_path.name != "latest.json":
                        local_path.unlink(missing_ok=True)
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

    FRAME_CACHE_STAGE_TO_BASE: dict[str, str] = {}
    ANNEALED_NOISE_STAGES = {"single_note_all", "single_instrument_melody"}
    BOUNDARY_COUNT_METRIC_SUFFIXES = (
        "_true_positive_count",
        "_false_positive_count",
        "_false_negative_count",
        "_true_negative_count",
        "_target_positive_count",
        "_target_negative_count",
        "_predicted_positive_count",
    )

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
            event_decoder_layers=config.event_decoder_layers,
            event_decoder_heads=config.event_decoder_heads,
            identity_dim=config.identity_dim,
        )
        self.model = DualPathTransformerSourceTrackerV2(self.model_config).to(self.device)
        if config.warm_start_checkpoint is not None:
            self._load_warm_start(Path(config.warm_start_checkpoint))
        self.loss_fn = SourceTrackingLossV2(
            activity_pos_weight=config.activity_pos_weight,
            inactive_slot_weight=config.inactive_slot_weight,
            family_loss_weight=config.family_loss_weight,
            count_loss_weight=config.count_loss_weight,
            onset_pos_weight=config.onset_pos_weight,
            offset_pos_weight=config.offset_pos_weight,
            boundary_loss_weight=config.boundary_loss_weight,
            onset_loss_weight=config.onset_loss_weight,
            offset_loss_weight=config.offset_loss_weight,
            onset_focal_gamma=config.onset_focal_gamma,
            offset_focal_gamma=config.offset_focal_gamma,
            boundary_timing_loss_weight=config.boundary_timing_loss_weight,
            boundary_f1_loss_weight=config.boundary_f1_loss_weight,
            boundary_f1_fp_weight=config.boundary_f1_fp_weight,
            boundary_f1_fn_weight=config.boundary_f1_fn_weight,
            hard_boundary_negative_loss_weight=(
                config.hard_boundary_negative_loss_weight
            ),
            hard_boundary_negative_fraction=config.hard_boundary_negative_fraction,
            onset_peak_loss_weight=config.onset_peak_loss_weight,
            onset_event_recall_loss_weight=config.onset_event_recall_loss_weight,
            onset_false_peak_loss_weight=config.onset_false_peak_loss_weight,
            onset_peak_radius_frames=config.onset_peak_radius_frames,
            onset_false_peak_fraction=config.onset_false_peak_fraction,
        )
        self.batch_builder = TrainingBatchBuilderV2(
            self.device,
            frame_count=config.training_slice_frames,
            pin_memory=config.pin_memory,
        )
        self.progress = TrainingProgressLoggerV2(config.progress_interval_s)
        self.render_executor: ProcessPoolExecutor | None = None
        self.render_executor_stage: str | None = None
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
                "stage_filter": self.config.stage_filter,
                "render_workers": self.config.render_workers,
                "midi_render_workers": self.config.midi_render_workers,
                "warm_start_checkpoint": self.config.warm_start_checkpoint,
                "epochs_per_stage": self.config.epochs_per_stage,
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
                "frame_cache_examples_per_stage": (
                    self.config.frame_cache_examples_per_stage
                ),
                "frame_cache_build_batch_size": self.config.frame_cache_build_batch_size,
                "frame_cache_dtype": self.config.frame_cache_dtype,
                "frame_cache_phase_jitter_samples": (
                    self.config.frame_cache_phase_jitter_samples
                ),
                "frame_phase_noise_std": self.config.frame_phase_noise_std,
                "anneal_noise_phase_jitter_samples": (
                    self.config.anneal_noise_phase_jitter_samples
                ),
                "anneal_noise_feature_std": self.config.anneal_noise_feature_std,
                "event_teacher_forcing_start": self.config.event_teacher_forcing_start,
                "event_teacher_forcing_end": self.config.event_teacher_forcing_end,
                "metric_activity_threshold": self.config.metric_activity_threshold,
                "loss_weights": {
                    "activity_pos_weight": self.config.activity_pos_weight,
                    "inactive_slot_weight": self.config.inactive_slot_weight,
                    "family_loss_weight": self.config.family_loss_weight,
                    "count_loss_weight": self.config.count_loss_weight,
                    "onset_pos_weight": self.config.onset_pos_weight,
                    "offset_pos_weight": self.config.offset_pos_weight,
                    "boundary_loss_weight": self.config.boundary_loss_weight,
                    "onset_loss_weight": self.config.onset_loss_weight,
                    "offset_loss_weight": self.config.offset_loss_weight,
                    "onset_focal_gamma": self.config.onset_focal_gamma,
                    "offset_focal_gamma": self.config.offset_focal_gamma,
                    "boundary_timing_loss_weight": (
                        self.config.boundary_timing_loss_weight
                    ),
                    "boundary_f1_loss_weight": self.config.boundary_f1_loss_weight,
                    "boundary_f1_fp_weight": self.config.boundary_f1_fp_weight,
                    "boundary_f1_fn_weight": self.config.boundary_f1_fn_weight,
                    "hard_boundary_negative_loss_weight": (
                        self.config.hard_boundary_negative_loss_weight
                    ),
                    "hard_boundary_negative_fraction": (
                        self.config.hard_boundary_negative_fraction
                    ),
                    "onset_peak_loss_weight": self.config.onset_peak_loss_weight,
                    "onset_event_recall_loss_weight": (
                        self.config.onset_event_recall_loss_weight
                    ),
                    "onset_false_peak_loss_weight": (
                        self.config.onset_false_peak_loss_weight
                    ),
                    "onset_peak_radius_frames": self.config.onset_peak_radius_frames,
                    "onset_false_peak_fraction": self.config.onset_false_peak_fraction,
                },
                "model_config": asdict(self.model_config),
                "warm_start_load_report": self.warm_start_load_report,
            },
            force=True,
        )
        if self.config.mixed_precision and self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        try:
            plan = (
                CurriculumPlanV2()
                .scale_examples(self.config.curriculum_scale)
                .scale_epochs(self.config.epoch_multiplier)
                .with_epochs_per_stage(self.config.epochs_per_stage)
            )
            examples_per_second = self.calibrate()
            self.progress.emit(
                "training_calibrated",
                {"examples_per_second": round(examples_per_second, 3)},
                force=True,
            )
            stages = plan.trim_for_hours(examples_per_second, self.config.max_hours)
            stages = self._filter_stages(stages)
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
                history.extend(self._train_stage(stage, run_dir, first_epoch, first_batch))
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
                self._shutdown_render_executor()

    def calibrate(self) -> float:
        start = time.perf_counter()
        count = max(1, min(self.config.calibration_examples, 128))
        self._start_render_executor("single_note_all")
        try:
            for idx in range(0, count, self.config.batch_size):
                seeds = [
                    self.config.seed + row
                    for row in range(idx, min(idx + self.config.batch_size, count))
                ]
                self._render_slices("single_note_all", "train", seeds)
        finally:
            self._shutdown_render_executor()
        elapsed = max(time.perf_counter() - start, 1e-6)
        return count / elapsed

    def _start_render_executor(self, stage_name: str) -> None:
        self._shutdown_render_executor()
        workers = self._render_worker_count(stage_name)
        if workers > 1:
            self.render_executor = ProcessPoolExecutor(max_workers=workers)
            self.render_executor_stage = stage_name
        self.progress.emit(
            "render_executor_started",
            {
                "stage": stage_name,
                "workers": workers,
                "include_midi": stage_name == "midi_complex",
            },
            force=True,
        )

    def _shutdown_render_executor(self) -> None:
        if self.render_executor is None:
            self.render_executor_stage = None
            return
        stage_name = self.render_executor_stage
        self._emit_memory_usage(
            "memory_usage",
            stage=stage_name,
            phase="before_render_executor_shutdown",
        )
        self.render_executor.shutdown(wait=True, cancel_futures=True)
        self.render_executor = None
        self.render_executor_stage = None
        self.progress.emit(
            "render_executor_stopped",
            {"stage": stage_name},
            force=True,
        )

    def _render_worker_count(self, stage_name: str) -> int:
        if stage_name == "midi_complex" and self.config.midi_render_workers is not None:
            return max(1, self.config.midi_render_workers)
        return max(1, self.config.render_workers)

    def _emit_memory_usage(
        self,
        event: str,
        *,
        stage: str | None,
        phase: str,
        epoch: int | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "stage": stage,
            "phase": phase,
            **self._memory_snapshot(),
        }
        if epoch is not None:
            payload["epoch"] = epoch
        self.progress.emit(event, payload, force=True)

    def _memory_snapshot(self) -> dict[str, object]:
        child_rss = [
            rss_mb
            for pid in self._render_child_pids()
            if (rss_mb := self._rss_mb_for_pid(pid)) is not None
        ]
        payload: dict[str, object] = {
            "main_rss_mb": self._rss_mb_for_pid(None),
            "render_worker_count": len(child_rss),
            "render_worker_rss_mb": round(sum(child_rss), 1) if child_rss else 0.0,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            payload.update(
                {
                    "cuda_allocated_mb": round(
                        torch.cuda.memory_allocated(self.device) / 1_000_000.0, 1
                    ),
                    "cuda_reserved_mb": round(
                        torch.cuda.memory_reserved(self.device) / 1_000_000.0, 1
                    ),
                    "cuda_max_allocated_mb": round(
                        torch.cuda.max_memory_allocated(self.device) / 1_000_000.0, 1
                    ),
                }
            )
        return payload

    def _render_child_pids(self) -> list[int]:
        if self.render_executor is None:
            return []
        processes = getattr(self.render_executor, "_processes", None)
        if not processes:
            return []
        return [process.pid for process in processes.values() if process.pid is not None]

    @staticmethod
    def _rss_mb_for_pid(pid: int | None) -> float | None:
        path = Path("/proc/self/status") if pid is None else Path(f"/proc/{pid}/status")
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return round(float(line.split()[1]) / 1024.0, 1)
        except OSError:
            return None
        return None

    def _train_stage(
        self,
        stage: CurriculumStageV2,
        run_dir: Path,
        first_epoch: int = 1,
        first_batch: int = 0,
    ) -> list[dict[str, object]]:
        frame_cache: CachedFramePoolV2 | None = None
        if self._is_frame_cache_stage(stage.name):
            self._start_render_executor(self._base_stage_for_training(stage.name))
            frame_cache = self._build_frame_cache(stage)
            self._shutdown_render_executor()
        else:
            self._start_render_executor(stage.name)
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
        self._emit_memory_usage("memory_usage", stage=stage.name, phase="stage_start")
        try:
            for epoch in range(first_epoch - 1, stage.epochs):
                losses = []
                batch_metrics = []
                epoch_start = time.monotonic()
                batch_start = first_batch if epoch == first_epoch - 1 else 0
                batch_iter = (
                    self._iter_cached_frame_batches(
                        frame_cache,
                        stage.name,
                        epoch,
                        batch_start,
                        batches_per_epoch,
                    )
                    if frame_cache is not None
                    else self._iter_training_batches(
                        stage.name,
                        epoch,
                        stage.epochs,
                        batch_start,
                        batches_per_epoch,
                    )
                )
                for batch_idx, batch, batch_timing in batch_iter:
                    compute_start = time.perf_counter()
                    with self._autocast_context():
                        teacher_forcing_rate = self._event_teacher_forcing_rate(
                            epoch,
                            stage.epochs,
                        )
                        outputs = self._model_forward(
                            batch,
                            teacher_forcing_rate=teacher_forcing_rate,
                        )
                        loss = self.loss_fn(outputs, batch)
                    batch_metrics.append(self._batch_metrics(outputs, batch))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    compute_s = time.perf_counter() - compute_start
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
                            "event_teacher_forcing_rate": round(
                                teacher_forcing_rate,
                                6,
                            ),
                        },
                    )
                    self.progress.emit(
                        "training_batch_timing",
                        {
                            "stage": stage.name,
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "batches_per_epoch": batches_per_epoch,
                            "batch_wait_s": round(batch_timing, 6),
                            "compute_s": round(compute_s, 6),
                            "cached_frames": frame_cache is not None,
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
                        stage.train_examples_per_epoch
                        - batch_start * self.config.batch_size,
                    ),
                    "elapsed_s": round(time.monotonic() - epoch_start, 3),
                    **self._mean_batch_metrics(batch_metrics),
                }
                self.progress.emit("training_epoch_done", row, force=True)
                self._emit_memory_usage(
                    "memory_usage",
                    stage=stage.name,
                    phase="epoch_done",
                    epoch=epoch + 1,
                )
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
        finally:
            self._shutdown_render_executor()
            self._emit_memory_usage("memory_usage", stage=stage.name, phase="stage_done")

    def _model_forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        teacher_forcing_rate: float,
    ) -> dict[str, torch.Tensor]:
        event_state = batch.get("event_state")
        if event_state is None:
            return self.model(batch["frames"])
        if teacher_forcing_rate >= 0.999:
            return self.model(batch["frames"], event_state=event_state)
        with torch.no_grad():
            first = self.model(batch["frames"], event_state=event_state)
            sampled_state = self._scheduled_event_state(
                event_state,
                first,
                teacher_forcing_rate,
            )
        return self.model(batch["frames"], event_state=sampled_state)

    def _event_teacher_forcing_rate(self, epoch: int, epochs: int) -> float:
        start = min(1.0, max(0.0, self.config.event_teacher_forcing_start))
        end = min(1.0, max(0.0, self.config.event_teacher_forcing_end))
        if epochs <= 1:
            return end
        progress = min(1.0, max(0.0, epoch / float(epochs - 1)))
        return start + (end - start) * progress

    @staticmethod
    def _scheduled_event_state(
        true_state: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        teacher_forcing_rate: float,
    ) -> torch.Tensor:
        onset = outputs.get("onset_logits_sequence")
        offset = outputs.get("offset_logits_sequence")
        if onset is None or offset is None:
            return true_state
        predicted = SourceTrackingEventStateBuilderV2.from_logits(
            onset,
            offset,
            event_state_dim=true_state.shape[-1],
        )
        keep_true = (
            torch.rand(
                true_state.shape[:-1],
                device=true_state.device,
                dtype=true_state.dtype,
            )
            < teacher_forcing_rate
        ).unsqueeze(-1)
        return torch.where(keep_true, true_state, predicted)

    def _iter_training_batches(
        self,
        stage_name: str,
        epoch: int,
        stage_epochs: int,
        batch_start: int,
        batches_per_epoch: int,
    ):
        if self.config.prefetch_batches <= 0:
            for batch_idx in range(batch_start, batches_per_epoch):
                start = time.perf_counter()
                batch = self._build_training_batch(
                    stage_name,
                    epoch,
                    stage_epochs,
                    batch_idx,
                )
                yield batch_idx, batch, time.perf_counter() - start
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
                            stage_epochs,
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
                start = time.perf_counter()
                yield batch_idx, future.result(), time.perf_counter() - start

    def _iter_cached_frame_batches(
        self,
        frame_cache: CachedFramePoolV2,
        stage_name: str,
        epoch: int,
        batch_start: int,
        batches_per_epoch: int,
    ):
        order = self._cached_frame_epoch_order(frame_cache, stage_name, epoch)
        for batch_idx in range(batch_start, batches_per_epoch):
            start = time.perf_counter()
            offset = batch_idx * self.config.batch_size
            indices = order[offset : offset + self.config.batch_size]
            yield (
                batch_idx,
                frame_cache.batch(
                    indices,
                    self.device,
                    pin_memory=self.config.pin_memory,
                    phase_noise_std=self.config.frame_phase_noise_std,
                ),
                time.perf_counter() - start,
            )

    def _build_training_batch(
        self,
        stage_name: str,
        epoch: int,
        stage_epochs: int,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        seeds = self._batch_seeds(epoch, batch_idx)
        phase_jitter_samples = self._annealed_phase_jitter_samples(
            stage_name,
            epoch,
            stage_epochs,
        )
        batch = self.batch_builder.build_slices(
            self._render_slices(
                stage_name,
                "train",
                seeds,
                phase_jitter_samples=phase_jitter_samples,
            )
        )
        noise_std = self._annealed_feature_noise_std(stage_name, epoch, stage_epochs)
        if noise_std > 0.0:
            batch["frames"] = torch.clamp(
                batch["frames"] + torch.randn_like(batch["frames"]) * noise_std,
                0.0,
                1.0,
            )
        return batch

    def _annealed_phase_jitter_samples(
        self,
        stage_name: str,
        epoch: int,
        stage_epochs: int,
    ) -> int:
        if stage_name not in self.ANNEALED_NOISE_STAGES:
            return 0
        maximum = max(0, int(self.config.anneal_noise_phase_jitter_samples))
        return int(round(maximum * self._stage_epoch_progress(epoch, stage_epochs)))

    def _annealed_feature_noise_std(
        self,
        stage_name: str,
        epoch: int,
        stage_epochs: int,
    ) -> float:
        if stage_name not in self.ANNEALED_NOISE_STAGES:
            return 0.0
        maximum = max(0.0, float(self.config.anneal_noise_feature_std))
        return maximum * self._stage_epoch_progress(epoch, stage_epochs)

    @staticmethod
    def _stage_epoch_progress(epoch: int, stage_epochs: int) -> float:
        if stage_epochs <= 1:
            return 1.0
        return min(1.0, max(0.0, epoch / float(stage_epochs - 1)))

    def _build_frame_cache(self, stage: CurriculumStageV2) -> CachedFramePoolV2:
        base_stage = self._base_stage_for_training(stage.name)
        examples = stage.train_examples_per_epoch
        cache = CachedFramePoolV2(
            examples,
            self.config.training_slice_frames,
            self.config.frame_cache_dtype,
        )
        build_batch_size = max(1, self.config.frame_cache_build_batch_size)
        self.progress.emit(
            "frame_cache_build_start",
            {
                "stage": stage.name,
                "base_stage": base_stage,
                "examples": examples,
                "build_batch_size": build_batch_size,
                "frame_dtype": self.config.frame_cache_dtype,
                "phase_jitter_samples": self.config.frame_cache_phase_jitter_samples,
            },
            force=True,
        )
        started = time.perf_counter()
        for start in range(0, examples, build_batch_size):
            stop = min(examples, start + build_batch_size)
            seeds = [
                self._frame_cache_seed(stage.name, row)
                for row in range(start, stop)
            ]
            slices = self._render_slices(
                base_stage,
                "train",
                seeds,
                gpu_frame_extraction=False,
                phase_jitter_samples=self.config.frame_cache_phase_jitter_samples,
            )
            cache.write_slices(start, slices, self.config.training_slice_frames)
            self.progress.emit(
                "frame_cache_build_progress",
                {
                    "stage": stage.name,
                    "cached_examples": stop,
                    "examples": examples,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                },
            )
        payload = {
            "stage": stage.name,
            "base_stage": base_stage,
            "examples": examples,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "frames_gb": round(cache.frames.nbytes / 1_000_000_000.0, 3),
        }
        self.progress.emit("frame_cache_build_done", payload, force=True)
        return cache

    def _cached_frame_epoch_order(
        self,
        frame_cache: CachedFramePoolV2,
        stage_name: str,
        epoch: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(self._frame_cache_seed(stage_name, epoch + 17_003))
        return rng.permutation(frame_cache.frames.shape[0])

    def _frame_cache_seed(self, stage_name: str, row: int) -> int:
        stage_offset = (
            31_000_000
            if stage_name == "single_note_frames_cached"
            else 43_000_000
        )
        return self.config.seed + stage_offset + row

    def _batch_seeds(self, epoch: int, batch_idx: int) -> list[int]:
        return [
            self.config.seed
            + epoch * 1_000_000
            + batch_idx * self.config.batch_size
            + row
            for row in range(self.config.batch_size)
        ]

    def _filter_stages(
        self, stages: tuple[CurriculumStageV2, ...]
    ) -> tuple[CurriculumStageV2, ...]:
        if not self.config.stage_filter:
            return stages
        requested = {
            item.strip() for item in self.config.stage_filter.split(",") if item.strip()
        }
        known = {stage.name for stage in stages} | set(self.FRAME_CACHE_STAGE_TO_BASE)
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown training stage(s): {unknown}")
        filtered_rows = []
        for stage in stages:
            if stage.name in requested:
                filtered_rows.append(stage)
            for cache_stage, base_stage in self.FRAME_CACHE_STAGE_TO_BASE.items():
                if cache_stage in requested and base_stage == stage.name:
                    filtered_rows.append(
                        CurriculumStageV2(
                            cache_stage,
                            self._frame_cache_examples_for_stage(stage),
                            stage.epochs,
                            stage.learning_rate,
                        )
                    )
        filtered = tuple(filtered_rows)
        if not filtered:
            raise ValueError("stage_filter selected no training stages")
        return filtered

    def _frame_cache_examples_for_stage(self, stage: CurriculumStageV2) -> int:
        if self.config.frame_cache_examples_per_stage > 0:
            return self.config.frame_cache_examples_per_stage
        return stage.train_examples_per_epoch

    def _is_frame_cache_stage(self, stage_name: str) -> bool:
        return stage_name in self.FRAME_CACHE_STAGE_TO_BASE

    def _base_stage_for_training(self, stage_name: str) -> str:
        return self.FRAME_CACHE_STAGE_TO_BASE.get(stage_name, stage_name)

    def _batch_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        with torch.no_grad():
            true_active = batch["activity"] > 0.5
            pred_active = (
                torch.sigmoid(outputs["activity_logits"])
                >= self.config.metric_activity_threshold
            )
            true_count = true_active.sum(dim=1)
            if "count_logits" in outputs:
                pred_count = torch.argmax(outputs["count_logits"], dim=-1)
            else:
                pred_count = pred_active.sum(dim=1)
            family_pred = torch.argmax(outputs["family_logits"], dim=-1)
            family_mask = true_active
            family_ok = family_pred[family_mask] == batch["family"][family_mask]
            onset_target = batch["onset"] > 0.5
            onset_pred = (
                torch.sigmoid(outputs["onset_logits"])
                >= self.config.metric_activity_threshold
            )
            offset_target = batch["offset"] > 0.5
            offset_pred = (
                torch.sigmoid(outputs["offset_logits"])
                >= self.config.metric_activity_threshold
            )
            onset_delta_metrics = self._timing_delta_metrics(
                outputs.get("onset_delta"),
                batch.get("onset_delta"),
                batch.get("onset_timing_mask"),
                "onset",
            )
            offset_delta_metrics = self._timing_delta_metrics(
                outputs.get("offset_delta"),
                batch.get("offset_delta"),
                batch.get("offset_timing_mask"),
                "offset",
            )
            onset_classification_metrics = self._boundary_classification_counts(
                "onset",
                onset_target,
                onset_pred,
            )
            offset_classification_metrics = self._boundary_classification_counts(
                "offset",
                offset_target,
                offset_pred,
            )
            return {
                "source_count_accuracy": float((true_count == pred_count).float().mean().cpu()),
                "mean_predicted_active_count": float(pred_count.float().mean().cpu()),
                "family_accuracy": float(family_ok.float().mean().cpu())
                if family_ok.numel()
                else 1.0,
                **onset_classification_metrics,
                **offset_classification_metrics,
                **onset_delta_metrics,
                **offset_delta_metrics,
            }

    @staticmethod
    def _timing_delta_metrics(
        prediction: torch.Tensor | None,
        target: torch.Tensor | None,
        mask: torch.Tensor | None,
        prefix: str,
    ) -> dict[str, float]:
        if prediction is None or target is None or mask is None:
            return {
                f"{prefix}_timing_mae_frames": float("nan"),
                f"{prefix}_timing_rmsd_frames": float("nan"),
                f"{prefix}_timing_points": 0.0,
            }
        active = mask > 0.5
        if not active.any():
            return {
                f"{prefix}_timing_mae_frames": float("nan"),
                f"{prefix}_timing_rmsd_frames": float("nan"),
                f"{prefix}_timing_points": 0.0,
            }
        delta = prediction[active] - target[active]
        return {
            f"{prefix}_timing_mae_frames": float(delta.abs().mean().cpu()),
            f"{prefix}_timing_rmsd_frames": float(torch.sqrt((delta * delta).mean()).cpu()),
            f"{prefix}_timing_points": float(active.float().sum().cpu()),
        }

    @staticmethod
    def _boundary_classification_counts(
        prefix: str,
        target: torch.Tensor,
        predicted: torch.Tensor,
    ) -> dict[str, float]:
        target_positive = target > 0.5
        predicted_positive = predicted > 0.5
        true_positive = target_positive & predicted_positive
        false_positive = (~target_positive) & predicted_positive
        false_negative = target_positive & (~predicted_positive)
        true_negative = (~target_positive) & (~predicted_positive)
        return {
            f"{prefix}_true_positive_count": float(true_positive.sum().cpu()),
            f"{prefix}_false_positive_count": float(false_positive.sum().cpu()),
            f"{prefix}_false_negative_count": float(false_negative.sum().cpu()),
            f"{prefix}_true_negative_count": float(true_negative.sum().cpu()),
            f"{prefix}_target_positive_count": float(target_positive.sum().cpu()),
            f"{prefix}_target_negative_count": float((~target_positive).sum().cpu()),
            f"{prefix}_predicted_positive_count": float(predicted_positive.sum().cpu()),
        }

    @staticmethod
    def _mean_batch_metrics(rows: list[dict[str, float]]) -> dict[str, float | None]:
        if not rows:
            return {}
        metrics: dict[str, float | None] = {}
        count_keys = {
            key
            for row in rows
            for key in row
            if key.endswith(SourceTrackingTrainerV2.BOUNDARY_COUNT_METRIC_SUFFIXES)
        }
        for key in rows[0]:
            if key in count_keys:
                metrics[key] = float(sum(row[key] for row in rows if key in row))
                continue
            values = [
                row[key]
                for row in rows
                if key in row and row[key] is not None and not np.isnan(row[key])
            ]
            metrics[key] = float(np.mean(values)) if values else None
        for prefix in ("onset", "offset"):
            SourceTrackingTrainerV2._add_boundary_classification_rates(metrics, prefix)
        return metrics

    @staticmethod
    def _add_boundary_classification_rates(
        metrics: dict[str, float | None],
        prefix: str,
    ) -> None:
        true_positive = metrics.get(f"{prefix}_true_positive_count")
        false_positive = metrics.get(f"{prefix}_false_positive_count")
        false_negative = metrics.get(f"{prefix}_false_negative_count")
        true_negative = metrics.get(f"{prefix}_true_negative_count")
        if (
            true_positive is None
            or false_positive is None
            or false_negative is None
            or true_negative is None
        ):
            return
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        false_positive_rate_denominator = false_positive + true_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else None
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else None
        )
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        metrics[f"{prefix}_precision"] = precision
        metrics[f"{prefix}_recall"] = recall
        metrics[f"{prefix}_f1"] = f1
        metrics[f"{prefix}_false_positive_rate"] = (
            false_positive / false_positive_rate_denominator
            if false_positive_rate_denominator > 0
            else None
        )

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
        self,
        stage: str,
        split: SplitName,
        seeds: list[int],
        *,
        gpu_frame_extraction: bool | None = None,
        phase_jitter_samples: int = 0,
    ) -> list[dict[str, np.ndarray]]:
        use_gpu_frame_extraction = (
            self.config.gpu_frame_extraction
            if gpu_frame_extraction is None
            else gpu_frame_extraction
        )
        phase_jitter_samples = max(0, int(phase_jitter_samples))
        phase_offsets = self._phase_offsets_for_seeds(seeds, phase_jitter_samples)
        if self.render_executor is None:
            renderer = self._renderer_for_stage(stage)
            return [
                self._render_training_slice(
                    renderer,
                    stage,
                    split,
                    seed,
                    gpu_frame_extraction=use_gpu_frame_extraction,
                    phase_offset_samples=phase_offset,
                )
                for seed, phase_offset in zip(seeds, phase_offsets, strict=True)
            ]
        tasks = [
            (
                self.config.data_root,
                stage,
                split,
                seed,
                self.config.training_slice_frames,
                self.config.training_slice_peak_warmup_frames,
                use_gpu_frame_extraction,
                phase_offset,
            )
            for seed, phase_offset in zip(seeds, phase_offsets, strict=True)
        ]
        return list(self.render_executor.map(TrainingRenderWorkerV2.render_slice, tasks))

    def _render_training_slice(
        self,
        renderer: SourceTrackingRendererV2,
        stage: str,
        split: SplitName,
        seed: int,
        *,
        gpu_frame_extraction: bool,
        phase_offset_samples: int,
    ) -> dict[str, np.ndarray]:
        if gpu_frame_extraction:
            return renderer.render_training_audio_slice(
                stage,
                split,
                seed,
                frame_count=self.config.training_slice_frames,
                peak_warmup_frames=self.config.training_slice_peak_warmup_frames,
                phase_offset_samples=phase_offset_samples,
            )
        return renderer.render_training_slice(
            stage,
            split,
            seed,
            frame_count=self.config.training_slice_frames,
            peak_warmup_frames=self.config.training_slice_peak_warmup_frames,
            phase_offset_samples=phase_offset_samples,
        )

    def _phase_offsets_for_seeds(
        self,
        seeds: list[int],
        phase_jitter_samples: int,
    ) -> list[int]:
        if phase_jitter_samples <= 0:
            return [0 for _ in seeds]
        upper = min(phase_jitter_samples, SourceTrackingAudioConfigV2.hop - 1)
        offsets = []
        for seed in seeds:
            rng = np.random.default_rng(seed + 71_291)
            offsets.append(int(rng.integers(0, upper + 1)))
        return offsets

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
        parser.add_argument("--stage-filter", default=TrainingConfigV2.stage_filter)
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
        parser.add_argument(
            "--midi-render-workers",
            type=int,
            default=TrainingConfigV2.midi_render_workers,
        )
        parser.add_argument("--warm-start-checkpoint")
        parser.add_argument(
            "--epoch-multiplier",
            type=float,
            default=TrainingConfigV2.epoch_multiplier,
        )
        parser.add_argument(
            "--epochs-per-stage",
            type=int,
            default=TrainingConfigV2.epochs_per_stage,
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
            "--metric-activity-threshold",
            type=float,
            default=TrainingConfigV2.metric_activity_threshold,
        )
        parser.add_argument(
            "--activity-pos-weight",
            type=float,
            default=TrainingConfigV2.activity_pos_weight,
        )
        parser.add_argument(
            "--inactive-slot-weight",
            type=float,
            default=TrainingConfigV2.inactive_slot_weight,
        )
        parser.add_argument(
            "--family-loss-weight",
            type=float,
            default=TrainingConfigV2.family_loss_weight,
        )
        parser.add_argument(
            "--count-loss-weight",
            type=float,
            default=TrainingConfigV2.count_loss_weight,
        )
        parser.add_argument(
            "--onset-pos-weight",
            type=float,
            default=TrainingConfigV2.onset_pos_weight,
        )
        parser.add_argument(
            "--offset-pos-weight",
            type=float,
            default=TrainingConfigV2.offset_pos_weight,
        )
        parser.add_argument(
            "--boundary-loss-weight",
            type=float,
            default=TrainingConfigV2.boundary_loss_weight,
        )
        parser.add_argument(
            "--onset-loss-weight",
            type=float,
            default=TrainingConfigV2.onset_loss_weight,
        )
        parser.add_argument(
            "--offset-loss-weight",
            type=float,
            default=TrainingConfigV2.offset_loss_weight,
        )
        parser.add_argument(
            "--onset-focal-gamma",
            type=float,
            default=TrainingConfigV2.onset_focal_gamma,
        )
        parser.add_argument(
            "--offset-focal-gamma",
            type=float,
            default=TrainingConfigV2.offset_focal_gamma,
        )
        parser.add_argument(
            "--boundary-timing-loss-weight",
            type=float,
            default=TrainingConfigV2.boundary_timing_loss_weight,
        )
        parser.add_argument(
            "--boundary-f1-loss-weight",
            type=float,
            default=TrainingConfigV2.boundary_f1_loss_weight,
        )
        parser.add_argument(
            "--boundary-f1-fp-weight",
            type=float,
            default=TrainingConfigV2.boundary_f1_fp_weight,
        )
        parser.add_argument(
            "--boundary-f1-fn-weight",
            type=float,
            default=TrainingConfigV2.boundary_f1_fn_weight,
        )
        parser.add_argument(
            "--hard-boundary-negative-loss-weight",
            type=float,
            default=TrainingConfigV2.hard_boundary_negative_loss_weight,
        )
        parser.add_argument(
            "--hard-boundary-negative-fraction",
            type=float,
            default=TrainingConfigV2.hard_boundary_negative_fraction,
        )
        parser.add_argument(
            "--onset-peak-loss-weight",
            type=float,
            default=TrainingConfigV2.onset_peak_loss_weight,
        )
        parser.add_argument(
            "--onset-event-recall-loss-weight",
            type=float,
            default=TrainingConfigV2.onset_event_recall_loss_weight,
        )
        parser.add_argument(
            "--onset-false-peak-loss-weight",
            type=float,
            default=TrainingConfigV2.onset_false_peak_loss_weight,
        )
        parser.add_argument(
            "--onset-peak-radius-frames",
            type=int,
            default=TrainingConfigV2.onset_peak_radius_frames,
        )
        parser.add_argument(
            "--onset-false-peak-fraction",
            type=float,
            default=TrainingConfigV2.onset_false_peak_fraction,
        )
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
            "--event-decoder-layers",
            type=int,
            default=TrainingConfigV2.event_decoder_layers,
        )
        parser.add_argument(
            "--event-decoder-heads",
            type=int,
            default=TrainingConfigV2.event_decoder_heads,
        )
        parser.add_argument(
            "--event-teacher-forcing-start",
            type=float,
            default=TrainingConfigV2.event_teacher_forcing_start,
        )
        parser.add_argument(
            "--event-teacher-forcing-end",
            type=float,
            default=TrainingConfigV2.event_teacher_forcing_end,
        )
        parser.add_argument(
            "--identity-dim",
            type=int,
            default=TrainingConfigV2.identity_dim,
        )
        parser.add_argument(
            "--frame-cache-examples-per-stage",
            type=int,
            default=TrainingConfigV2.frame_cache_examples_per_stage,
            help="Examples to pre-render for each optional frame-cached stage.",
        )
        parser.add_argument(
            "--frame-cache-build-batch-size",
            type=int,
            default=TrainingConfigV2.frame_cache_build_batch_size,
        )
        parser.add_argument(
            "--frame-cache-dtype",
            choices=("float16", "float32"),
            default=TrainingConfigV2.frame_cache_dtype,
        )
        parser.add_argument(
            "--frame-cache-phase-jitter-samples",
            type=int,
            default=TrainingConfigV2.frame_cache_phase_jitter_samples,
            help="Max random STFT start offset, in samples, used while caching frames.",
        )
        parser.add_argument(
            "--frame-phase-noise-std",
            type=float,
            default=TrainingConfigV2.frame_phase_noise_std,
            help="Gaussian feature noise std added to cached frames at train time.",
        )
        parser.add_argument(
            "--anneal-noise-phase-jitter-samples",
            type=int,
            default=TrainingConfigV2.anneal_noise_phase_jitter_samples,
            help="Max STFT phase jitter gradually added across stage 1 and 2.",
        )
        parser.add_argument(
            "--anneal-noise-feature-std",
            type=float,
            default=TrainingConfigV2.anneal_noise_feature_std,
            help="Max Gaussian feature noise gradually added across stage 1 and 2.",
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
