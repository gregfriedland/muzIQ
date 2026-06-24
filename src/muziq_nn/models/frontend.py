"""Learned audio frontend modules for source tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LeafFrontendConfigV2:
    frontend_name: str = "leaf"
    sample_rate: int = 16_000
    fine_hop_samples: int = 40
    context_hop_samples: int = 160
    kernel_size: int = 384
    filters: int = 128
    min_hz: float = 40.0
    max_hz: float = 8_000.0
    pcen_eps: float = 1e-6
    onset_shoulder_ms: float = 25.0
    normalization: str = "pcen"
    pooling: str = "mean_max"

    @property
    def pool_factor(self) -> int:
        if self.context_hop_samples % self.fine_hop_samples != 0:
            raise ValueError("context_hop_samples must be divisible by fine_hop_samples")
        return self.context_hop_samples // self.fine_hop_samples

    @property
    def feature_dim(self) -> int:
        if self.pooling == "mean_max":
            return self.filters * 2
        if self.pooling == "mean":
            return self.filters
        raise ValueError(f"Unknown LEAF pooling mode: {self.pooling!r}")

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


class LeafAudioFrontendV2(nn.Module):
    """Small LEAF-style waveform frontend with Gabor init and PCEN compression."""

    def __init__(self, config: LeafFrontendConfigV2 | None = None):
        super().__init__()
        config = config or LeafFrontendConfigV2()
        self.config = config
        real, imag = self._initial_gabor_filters(config)
        self.real_filters = nn.Parameter(real)
        self.imag_filters = nn.Parameter(imag)
        self.pcen_alpha_logit = nn.Parameter(torch.full((config.filters,), 0.4).logit())
        self.pcen_smooth_logit = nn.Parameter(torch.full((config.filters,), -3.0))
        self.pcen_delta_raw = nn.Parameter(torch.full((config.filters,), 2.0))
        self.pcen_root_logit = nn.Parameter(torch.full((config.filters,), 0.5).logit())

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def metadata(self) -> dict[str, object]:
        params = self.pcen_parameter_snapshot()
        return {
            **self.config.to_metadata(),
            "pcen_alpha_range": params["alpha_range"],
            "pcen_smooth_range": params["smooth_range"],
            "pcen_delta_range": params["delta_range"],
            "pcen_root_range": params["root_range"],
        }

    def pcen_parameter_snapshot(self) -> dict[str, tuple[float, float]]:
        with torch.no_grad():
            alpha, smooth, delta, root = self._pcen_parameters()
            return {
                "alpha_range": self._range(alpha),
                "smooth_range": self._range(smooth),
                "delta_range": self._range(delta),
                "root_range": self._range(root),
            }

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"LEAF frontend expects [batch, samples], got {audio.shape}")
        fine = self._fine_features(audio)
        pcen = self._pcen(fine)
        return self._pool_to_context(pcen, audio.shape[-1])

    def _fine_features(self, audio: torch.Tensor) -> torch.Tensor:
        padded = F.pad(audio.unsqueeze(1), (self.config.kernel_size - 1, 0))
        real = F.conv1d(
            padded,
            self.real_filters.unsqueeze(1),
            stride=self.config.fine_hop_samples,
        )
        imag = F.conv1d(
            padded,
            self.imag_filters.unsqueeze(1),
            stride=self.config.fine_hop_samples,
        )
        return torch.sqrt(real.square() + imag.square() + self.config.pcen_eps).transpose(1, 2)

    def _pcen(self, features: torch.Tensor) -> torch.Tensor:
        alpha, smooth, delta, root = self._pcen_parameters()
        state = features[:, :1, :]
        states = []
        for frame_idx in range(features.shape[1]):
            current = features[:, frame_idx : frame_idx + 1, :]
            state = (1.0 - smooth.view(1, 1, -1)) * state + smooth.view(1, 1, -1) * current
            states.append(state)
        smoother = torch.cat(states, dim=1)
        normalized = features / (self.config.pcen_eps + smoother).pow(alpha.view(1, 1, -1))
        return (normalized + delta.view(1, 1, -1)).pow(root.view(1, 1, -1)) - delta.view(
            1, 1, -1
        ).pow(root.view(1, 1, -1))

    def _pool_to_context(self, fine: torch.Tensor, sample_count: int) -> torch.Tensor:
        factor = self.config.pool_factor
        pad = (-fine.shape[1]) % factor
        if pad:
            fine = F.pad(fine, (0, 0, 0, pad))
        grouped = fine.reshape(fine.shape[0], fine.shape[1] // factor, factor, fine.shape[2])
        mean = grouped.mean(dim=2)
        if self.config.pooling == "mean":
            pooled = mean
        elif self.config.pooling == "mean_max":
            pooled = torch.cat([mean, grouped.max(dim=2).values], dim=-1)
        else:
            raise ValueError(f"Unknown LEAF pooling mode: {self.config.pooling!r}")
        target_frames = max(
            1,
            int(
                torch.ceil(
                    torch.tensor(
                        max(1, sample_count - self.config.kernel_size)
                        / float(self.config.context_hop_samples)
                    )
                ).item()
            )
            + 1,
        )
        return pooled[:, -target_frames:, :]

    def _pcen_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.pcen_alpha_logit).clamp(0.05, 0.98)
        smooth = torch.sigmoid(self.pcen_smooth_logit).clamp(0.001, 0.2)
        delta = F.softplus(self.pcen_delta_raw).clamp(0.01, 20.0)
        root = torch.sigmoid(self.pcen_root_logit).clamp(0.1, 0.98)
        return alpha, smooth, delta, root

    @staticmethod
    def _range(values: torch.Tensor) -> tuple[float, float]:
        return (float(values.min().item()), float(values.max().item()))

    @staticmethod
    def _initial_gabor_filters(
        config: LeafFrontendConfigV2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = torch.logspace(
            torch.log10(torch.tensor(config.min_hz)),
            torch.log10(torch.tensor(config.max_hz)),
            config.filters,
        )
        time = torch.arange(config.kernel_size, dtype=torch.float32)
        centered = time - float(config.kernel_size - 1)
        real = []
        imag = []
        for center in centers:
            cycles = center / float(config.sample_rate)
            sigma = max(float(config.sample_rate / (center * 8.0)), 8.0)
            envelope = torch.exp(-0.5 * (centered / sigma).square())
            phase = 2.0 * torch.pi * cycles * centered
            cos_filter = envelope * torch.cos(phase)
            sin_filter = envelope * torch.sin(phase)
            norm = torch.sqrt(
                torch.sum(cos_filter.square() + sin_filter.square())
            ).clamp_min(1e-6)
            real.append(cos_filter / norm)
            imag.append(sin_filter / norm)
        return torch.stack(real), torch.stack(imag)
