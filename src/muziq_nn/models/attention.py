"""Compact attention-only source-tracking models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SourceTrackingModelConfigV2:
    n_bands: int = 40
    n_families: int = 11
    max_sources: int = 5
    model_dim: int = 128
    heads: int = 4
    layers: int = 2
    identity_dim: int = 16


class SourceTrackingHeadV2(nn.Module):
    """Decode fixed source slots from encoded sequence summaries."""

    def __init__(self, config: SourceTrackingModelConfigV2):
        super().__init__()
        self.config = config
        self.slot_queries = nn.Parameter(
            torch.randn(config.max_sources, config.model_dim) * 0.02
        )
        self.cross_attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.activity = nn.Linear(config.model_dim, 1)
        self.family = nn.Linear(config.model_dim, config.n_families)
        self.onset = nn.Linear(config.model_dim, 1)
        self.offset = nn.Linear(config.model_dim, 1)
        self.identity = nn.Linear(config.model_dim, config.identity_dim)

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = encoded.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        slots, _ = self.cross_attention(queries, encoded, encoded, need_weights=False)
        slots = self.norm(slots)
        return {
            "activity_logits": self.activity(slots).squeeze(-1),
            "family_logits": self.family(slots),
            "onset_logits": self.onset(slots).squeeze(-1),
            "offset_logits": self.offset(slots).squeeze(-1),
            "identity": F.normalize(self.identity(slots), dim=-1),
        }


class DualPathTransformerSourceTrackerV2(nn.Module):
    """Small dual-path transformer for cached 40-band frame sequences."""

    def __init__(self, config: SourceTrackingModelConfigV2 | None = None):
        super().__init__()
        self.config = config or SourceTrackingModelConfigV2()
        c = self.config
        self.input = nn.Linear(c.n_bands, c.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.model_dim,
            nhead=c.heads,
            dim_feedforward=c.model_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.layers)
        self.head = SourceTrackingHeadV2(c)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.input(frames))
        return self.head(encoded)


class SourceTrackingLossV2(nn.Module):
    """Label-only multitask training loss."""

    def __init__(self, activity_pos_weight: float = 2.0):
        super().__init__()
        self.activity_pos_weight = activity_pos_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        activity = targets["activity"].float()
        pos_weight = torch.full(
            (outputs["activity_logits"].shape[-1],),
            self.activity_pos_weight,
            device=outputs["activity_logits"].device,
        )
        activity_loss = F.binary_cross_entropy_with_logits(
            outputs["activity_logits"],
            activity,
            pos_weight=pos_weight,
        )
        onset_loss = F.binary_cross_entropy_with_logits(
            outputs["onset_logits"], targets["onset"].float()
        )
        offset_loss = F.binary_cross_entropy_with_logits(
            outputs["offset_logits"], targets["offset"].float()
        )
        family_loss = self._family_loss(outputs["family_logits"], targets["family"], activity)
        return activity_loss + 0.5 * family_loss + 0.25 * (onset_loss + offset_loss)

    @staticmethod
    def _family_loss(
        family_logits: torch.Tensor,
        family: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        active = activity > 0.5
        if not active.any():
            return family_logits.sum() * 0.0
        return F.cross_entropy(family_logits[active], family.clamp_min(0)[active].long())
