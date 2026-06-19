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
        self.onset_delta = nn.Linear(config.model_dim, 1)
        self.offset_delta = nn.Linear(config.model_dim, 1)
        self.identity = nn.Linear(config.model_dim, config.identity_dim)
        self.count = nn.Linear(config.model_dim, config.max_sources + 1)

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
            "onset_delta": self.onset_delta(slots).squeeze(-1),
            "offset_delta": self.offset_delta(slots).squeeze(-1),
            "identity": F.normalize(self.identity(slots), dim=-1),
            "count_logits": self.count(slots.mean(dim=1)),
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

    def __init__(
        self,
        activity_pos_weight: float = 10.0,
        inactive_slot_weight: float = 6.0,
        family_loss_weight: float = 2.5,
        count_loss_weight: float = 8.0,
        onset_pos_weight: float = 10.0,
        offset_pos_weight: float = 10.0,
        boundary_loss_weight: float = 0.5,
        boundary_timing_loss_weight: float = 0.25,
    ):
        super().__init__()
        self.activity_pos_weight = activity_pos_weight
        self.inactive_slot_weight = inactive_slot_weight
        self.family_loss_weight = family_loss_weight
        self.count_loss_weight = count_loss_weight
        self.onset_pos_weight = onset_pos_weight
        self.offset_pos_weight = offset_pos_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_timing_loss_weight = boundary_timing_loss_weight

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
        activity_slot_loss = F.binary_cross_entropy_with_logits(
            outputs["activity_logits"],
            activity,
            pos_weight=pos_weight,
            reduction="none",
        )
        activity_weight = torch.where(
            activity > 0.5,
            torch.ones_like(activity),
            torch.full_like(activity, self.inactive_slot_weight),
        )
        activity_loss = (activity_slot_loss * activity_weight).mean()
        count_loss = self._count_loss(outputs, activity)
        onset_loss = F.binary_cross_entropy_with_logits(
            outputs["onset_logits"],
            targets["onset"].float(),
            pos_weight=self._pos_weight(outputs["onset_logits"], self.onset_pos_weight),
        )
        offset_loss = F.binary_cross_entropy_with_logits(
            outputs["offset_logits"],
            targets["offset"].float(),
            pos_weight=self._pos_weight(outputs["offset_logits"], self.offset_pos_weight),
        )
        zero_timing_loss = activity_loss * 0.0
        onset_timing_loss = (
            self._timing_loss(
                outputs.get("onset_delta"),
                targets.get("onset_delta"),
                targets.get("onset_timing_mask"),
            )
            if self._has_timing_tensors(outputs, targets, "onset")
            else zero_timing_loss
        )
        offset_timing_loss = (
            self._timing_loss(
                outputs.get("offset_delta"),
                targets.get("offset_delta"),
                targets.get("offset_timing_mask"),
            )
            if self._has_timing_tensors(outputs, targets, "offset")
            else zero_timing_loss
        )
        family_loss = self._family_loss(outputs["family_logits"], targets["family"], activity)
        return (
            activity_loss
            + self.count_loss_weight * count_loss
            + self.family_loss_weight * family_loss
            + self.boundary_loss_weight * (onset_loss + offset_loss)
            + self.boundary_timing_loss_weight * (
                onset_timing_loss + offset_timing_loss
            )
        )

    @staticmethod
    def _timing_loss(
        prediction: torch.Tensor | None,
        target: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        active = mask.float() > 0.5
        if not active.any():
            return prediction.sum() * 0.0
        return F.smooth_l1_loss(prediction[active], target.float()[active])

    @staticmethod
    def _has_timing_tensors(
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        prefix: str,
    ) -> bool:
        return (
            outputs.get(f"{prefix}_delta") is not None
            and targets.get(f"{prefix}_delta") is not None
            and targets.get(f"{prefix}_timing_mask") is not None
        )

    @staticmethod
    def _count_loss(
        outputs: dict[str, torch.Tensor],
        activity: torch.Tensor,
    ) -> torch.Tensor:
        target_count = activity.sum(dim=-1).long()
        if "count_logits" in outputs:
            max_count = outputs["count_logits"].shape[-1] - 1
            return F.cross_entropy(outputs["count_logits"], target_count.clamp_max(max_count))
        return F.mse_loss(
            torch.sigmoid(outputs["activity_logits"]).sum(dim=-1),
            activity.sum(dim=-1),
        )

    @staticmethod
    def _pos_weight(logits: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full((logits.shape[-1],), value, device=logits.device)

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
