"""Realtime checkpoint-backed source tracking inference."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sps

from muziq_nn.datasets.render import (
    AudioFrameExtractorV2,
    FamilyVocabularyV2,
    SourceTrackingAudioConfigV2,
)
from muziq_nn.models.attention import (
    DualPathTransformerSourceTrackerV2,
    SourceTrackingEventStateBuilderV2,
    SourceTrackingModelConfigV2,
)


@dataclass(frozen=True)
class SourcePredictionV2:
    slot: int
    activity: float
    family: str
    family_index: int
    confidence: float
    onset: float
    offset: float
    onset_delta: float
    offset_delta: float
    position: float

    def to_json(self) -> dict[str, float | int | str]:
        return {
            "slot": self.slot,
            "activity": self.activity,
            "family": self.family,
            "family_index": self.family_index,
            "confidence": self.confidence,
            "onset": self.onset,
            "offset": self.offset,
            "onset_delta": self.onset_delta,
            "offset_delta": self.offset_delta,
            "position": self.position,
        }


@dataclass(frozen=True)
class SourceTrackingInferenceResultV2:
    timestamp_s: float
    frame_count: int
    sample_count: int
    source_count: int
    checkpoint_path: str
    sources: tuple[SourcePredictionV2, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "type": "prediction",
            "timestamp_s": self.timestamp_s,
            "frame_count": self.frame_count,
            "sample_count": self.sample_count,
            "source_count": self.source_count,
            "checkpoint_path": self.checkpoint_path,
            "sources": [source.to_json() for source in self.sources],
        }


class SourceTrackingCheckpointLoaderV2:
    """Load trained source-tracking checkpoints without training-side objects."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        self.checkpoint_path = str(checkpoint_path)
        self.device = self._select_device(device)
        self.payload = self._load_payload()
        self.state_dict = self._state_dict_from_payload(self.payload)
        self.has_count_head = any(name.startswith("head.count.") for name in self.state_dict)
        self.config = self._infer_config(self.state_dict)
        self.model = self._load_model()

    def _load_payload(self) -> Any:
        return torch.load(self.checkpoint_path, map_location=self.device)

    def _load_model(self) -> DualPathTransformerSourceTrackerV2:
        model = DualPathTransformerSourceTrackerV2(self.config).to(self.device)
        result = model.load_state_dict(self.state_dict, strict=False)
        missing = [
            name
            for name in result.missing_keys
            if (
                (self.has_count_head or not name.startswith("head.count."))
                and not name.startswith("head.onset_delta.")
                and not name.startswith("head.offset_delta.")
                and not name.startswith("event_decoder.")
            )
        ]
        unexpected = [
            name
            for name in result.unexpected_keys
            if not name.startswith("head.onset.") and not name.startswith("head.offset.")
        ]
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint state dict did not match model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        return model

    @staticmethod
    def _select_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _state_dict_from_payload(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"]
        if isinstance(payload, dict):
            return payload
        raise ValueError("checkpoint payload does not contain a PyTorch state dict")

    @classmethod
    def _infer_config(
        cls, state_dict: dict[str, torch.Tensor]
    ) -> SourceTrackingModelConfigV2:
        input_weight = state_dict["input.weight"]
        family_weight = state_dict["head.family.weight"]
        slot_queries = state_dict["head.slot_queries"]
        identity_weight = state_dict["head.identity.weight"]
        layer_count = cls._infer_layer_count(state_dict)
        event_decoder_layers = cls._infer_event_decoder_layer_count(state_dict)
        return SourceTrackingModelConfigV2(
            n_bands=int(input_weight.shape[1]),
            n_families=int(family_weight.shape[0]),
            max_sources=int(slot_queries.shape[0]),
            model_dim=int(input_weight.shape[0]),
            heads=4,
            layers=layer_count,
            event_decoder_layers=event_decoder_layers,
            identity_dim=int(identity_weight.shape[0]),
        )

    @staticmethod
    def _infer_layer_count(state_dict: dict[str, torch.Tensor]) -> int:
        layer_ids: set[int] = set()
        for name in state_dict:
            if not name.startswith("encoder.layers."):
                continue
            parts = name.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
        return max(layer_ids) + 1 if layer_ids else 2

    @staticmethod
    def _infer_event_decoder_layer_count(
        state_dict: dict[str, torch.Tensor],
    ) -> int:
        layer_ids: set[int] = set()
        for name in state_dict:
            if not name.startswith("event_decoder.decoder.layers."):
                continue
            parts = name.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                layer_ids.add(int(parts[3]))
        return max(layer_ids) + 1 if layer_ids else 1

    def predict_sequence(
        self,
        frames: torch.Tensor,
        *,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        frames = self.prepare_frames(frames)
        first = self.model(frames)
        onset = first.get("onset_logits_sequence")
        offset = first.get("offset_logits_sequence")
        if onset is None or offset is None:
            return first
        event_state = SourceTrackingEventStateBuilderV2.from_logits(
            onset,
            offset,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            event_state_dim=self.config.event_state_dim,
        )
        return self.model(frames, event_state=event_state)

    @staticmethod
    def event_logits(outputs: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        sequence = outputs.get(f"{prefix}_logits_sequence")
        if sequence is not None:
            return sequence[:, -1, :]
        return outputs[f"{prefix}_logits"]

    def prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] == self.config.n_bands:
            return frames
        if frames.shape[-1] != SourceTrackingAudioConfigV2.bands:
            raise ValueError(
                "frame feature width does not match checkpoint: "
                f"got {frames.shape[-1]}, expected {self.config.n_bands}"
            )
        if self.config.n_bands == SourceTrackingAudioConfigV2.bands * 2:
            return self._augment_frame_features(frames, "flux")
        salience_width = SourceTrackingAudioConfigV2.bands * 6 + 5
        if self.config.n_bands == salience_width:
            return self._augment_frame_features(frames, "salience")
        raise ValueError(
            "unsupported checkpoint input feature width: "
            f"{self.config.n_bands}"
        )

    @staticmethod
    def _augment_frame_features(frames: torch.Tensor, mode: str) -> torch.Tensor:
        previous = torch.cat([frames[:, :1, :], frames[:, :-1, :]], dim=1)
        diff = frames - previous
        flux = torch.relu(diff)
        if mode == "flux":
            return torch.cat([frames, flux], dim=-1)
        previous_diff = torch.cat([diff[:, :1, :], diff[:, :-1, :]], dim=1)
        negative_flux = torch.relu(-diff)
        absolute_flux = diff.abs()
        positive_acceleration = torch.relu(diff - previous_diff)
        local_mean = F.avg_pool1d(
            frames.reshape(-1, frames.shape[-1]).unsqueeze(1),
            kernel_size=5,
            stride=1,
            padding=2,
        ).squeeze(1).reshape_as(frames)
        local_contrast = torch.relu(frames - local_mean)
        broadband_energy = frames.mean(dim=-1, keepdim=True)
        broadband_flux = flux.mean(dim=-1, keepdim=True)
        low_band_flux = flux[..., : frames.shape[-1] // 2].mean(dim=-1, keepdim=True)
        high_band_flux = flux[..., frames.shape[-1] // 2 :].mean(dim=-1, keepdim=True)
        weights = torch.linspace(
            0.0,
            1.0,
            frames.shape[-1],
            dtype=frames.dtype,
            device=frames.device,
        ).view(1, 1, -1)
        spectral_centroid = (frames * weights).sum(dim=-1, keepdim=True) / frames.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        return torch.cat(
            [
                frames,
                flux,
                negative_flux,
                absolute_flux,
                positive_acceleration,
                local_contrast,
                broadband_energy,
                broadband_flux,
                low_band_flux,
                high_band_flux,
                spectral_centroid,
            ],
            dim=-1,
        )


class RealtimeSourceTrackerV2:
    """Maintain rolling audio context and emit source-slot predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        context_frames: int = 256,
        activity_threshold: float = 0.35,
    ):
        self.loaded = SourceTrackingCheckpointLoaderV2(checkpoint_path, device=device)
        self.context_frames = context_frames
        self.activity_threshold = activity_threshold
        self.extractor = AudioFrameExtractorV2()
        self.families = FamilyVocabularyV2()
        self.audio = np.zeros(0, dtype=np.float32)
        self.started_at = time.monotonic()
        self.max_samples = int(
            SourceTrackingAudioConfigV2.sample_rate
            * SourceTrackingAudioConfigV2.duration_s
        )

    @property
    def checkpoint_path(self) -> str:
        return self.loaded.checkpoint_path

    def append_audio(
        self, samples: np.ndarray, sample_rate: int = SourceTrackingAudioConfigV2.sample_rate
    ) -> SourceTrackingInferenceResultV2:
        mono = self._prepare_audio(samples, sample_rate)
        if len(mono):
            self.audio = np.concatenate((self.audio, mono))
            if len(self.audio) > self.max_samples:
                self.audio = self.audio[-self.max_samples :]
        return self.predict()

    def predict(self) -> SourceTrackingInferenceResultV2:
        if len(self.audio) == 0:
            return self._empty_result(0)
        frames = self.extractor.extract_context(
            self.audio,
            end_frame=int(np.ceil(len(self.audio) / SourceTrackingAudioConfigV2.hop)) - 1,
            frame_count=self.context_frames,
            peak_warmup_frames=self.context_frames,
        )
        if len(frames) == 0:
            return self._empty_result(0)
        return self._predict_frames(frames)

    def reset(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)
        self.started_at = time.monotonic()

    def _predict_frames(self, frames: np.ndarray) -> SourceTrackingInferenceResultV2:
        tensor = torch.from_numpy(frames).unsqueeze(0).float().to(self.loaded.device)
        with torch.inference_mode():
            outputs = self.loaded.predict_sequence(tensor)
        activity = torch.sigmoid(outputs["activity_logits"])[0].detach().cpu().numpy()
        family_probs = torch.softmax(outputs["family_logits"], dim=-1)[0].detach().cpu()
        onset = (
            torch.sigmoid(self.loaded.event_logits(outputs, "onset"))[0]
            .detach()
            .cpu()
            .numpy()
        )
        offset = (
            torch.sigmoid(self.loaded.event_logits(outputs, "offset"))[0]
            .detach()
            .cpu()
            .numpy()
        )
        onset_delta = outputs["onset_delta"][0].detach().cpu().numpy()
        offset_delta = outputs["offset_delta"][0].detach().cpu().numpy()
        identity = outputs["identity"][0].detach().cpu().numpy()
        predicted_count = self._predicted_count(outputs)
        sources = []
        for slot, slot_activity in enumerate(activity):
            family_index = int(torch.argmax(family_probs[slot]).item())
            family_confidence = float(family_probs[slot, family_index].item())
            sources.append(
                SourcePredictionV2(
                    slot=slot,
                    activity=float(slot_activity),
                    family=self._family_name(family_index),
                    family_index=family_index,
                    confidence=float(slot_activity * family_confidence),
                    onset=float(onset[slot]),
                    offset=float(offset[slot]),
                    onset_delta=float(onset_delta[slot]),
                    offset_delta=float(offset_delta[slot]),
                    position=self._slot_position(identity[slot]),
                )
            )
        return SourceTrackingInferenceResultV2(
            timestamp_s=time.monotonic() - self.started_at,
            frame_count=int(frames.shape[0]),
            sample_count=int(len(self.audio)),
            source_count=predicted_count
            if predicted_count is not None
            else sum(1 for source in sources if source.activity >= self.activity_threshold),
            checkpoint_path=self.loaded.checkpoint_path,
            sources=tuple(sources),
        )

    def _predicted_count(self, outputs: dict[str, torch.Tensor]) -> int | None:
        if not self.loaded.has_count_head or "count_logits" not in outputs:
            return None
        count = int(torch.argmax(outputs["count_logits"][0]).detach().cpu().item())
        return int(np.clip(count, 0, self.loaded.config.max_sources))

    def _empty_result(self, frame_count: int) -> SourceTrackingInferenceResultV2:
        return SourceTrackingInferenceResultV2(
            timestamp_s=time.monotonic() - self.started_at,
            frame_count=frame_count,
            sample_count=int(len(self.audio)),
            source_count=0,
            checkpoint_path=self.loaded.checkpoint_path,
            sources=tuple(),
        )

    def _family_name(self, family_index: int) -> str:
        if 0 <= family_index < len(self.families.families):
            return self.families.families[family_index]
        return f"family_{family_index}"

    @staticmethod
    def _slot_position(identity: np.ndarray) -> float:
        if identity.size:
            value = float(1.0 / (1.0 + np.exp(-identity[0] * 2.0)))
            return float(np.clip(value, 0.0, 1.0))
        return 0.5

    @staticmethod
    def _prepare_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SourceTrackingAudioConfigV2.sample_rate:
            gcd = int(np.gcd(sample_rate, SourceTrackingAudioConfigV2.sample_rate))
            audio = sps.resample_poly(
                audio,
                SourceTrackingAudioConfigV2.sample_rate // gcd,
                sample_rate // gcd,
            ).astype(np.float32)
        return np.clip(np.nan_to_num(audio), -1.0, 1.0)


class SourceTrackingCheckpointLocatorV2:
    """Find a usable local checkpoint for the web app."""

    ENV_VAR = "MUZIQ_NN_CHECKPOINT"

    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()

    def locate(self, explicit: str | None = None) -> Path | None:
        for value in (explicit, os.environ.get(self.ENV_VAR)):
            if value:
                path = self._resolve_path(value)
                if path.exists():
                    return path
        candidates = self._run_checkpoint_candidates()
        existing = [path for path in candidates if path.exists()]
        if not existing:
            return None
        return max(existing, key=lambda path: path.stat().st_mtime)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def _run_checkpoint_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        runs = self.root / "runs"
        if runs.exists():
            candidates.extend(runs.glob("**/*phase-training_done*.pt"))
            candidates.extend(runs.glob("**/checkpoint.pt"))
            candidates.extend(runs.glob("**/*.pt"))
        return candidates
