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
    event_state_dim: int = 5
    event_decoder_layers: int = 1
    event_decoder_heads: int = 4
    event_state_conditioning: bool = True


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
            "onset_delta": self.onset_delta(slots).squeeze(-1),
            "offset_delta": self.offset_delta(slots).squeeze(-1),
            "identity": F.normalize(self.identity(slots), dim=-1),
            "count_logits": self.count(slots.mean(dim=1)),
        }


class SourceTrackingEventDecoderV2(nn.Module):
    """Causal onset/offset decoder with optional prior-event conditioning."""

    def __init__(self, config: SourceTrackingModelConfigV2):
        super().__init__()
        self.config = config
        self.source_embedding = nn.Parameter(
            torch.randn(config.max_sources, config.model_dim) * 0.02
        )
        self.event_state = nn.Linear(config.event_state_dim, config.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.event_decoder_heads,
            dim_feedforward=config.model_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config.event_decoder_layers)
        self.norm = nn.LayerNorm(config.model_dim)
        self.onset = nn.Linear(config.model_dim, 1)
        self.offset = nn.Linear(config.model_dim, 1)
        self.onset_age = nn.Linear(config.model_dim, 1)
        self.offset_age = nn.Linear(config.model_dim, 1)

    def forward(
        self,
        encoded: torch.Tensor,
        event_state: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch, frames, dim = encoded.shape
        sources = self.config.max_sources
        if event_state is None or not self.config.event_state_conditioning:
            event_state = encoded.new_zeros(
                batch,
                frames,
                sources,
                self.config.event_state_dim,
            )
        elif event_state.shape[1] != frames:
            event_state = event_state[:, -frames:]
        event_features = self.event_state(event_state.to(dtype=encoded.dtype))
        source_features = self.source_embedding.view(1, 1, sources, dim)
        sequence = encoded.unsqueeze(2) + event_features + source_features
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * sources, frames, dim)
        mask = torch.triu(
            torch.ones(frames, frames, device=encoded.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.decoder(sequence, mask=mask)
        decoded = self.norm(decoded).view(batch, sources, frames, dim).permute(0, 2, 1, 3)
        onset_sequence = self.onset(decoded).squeeze(-1)
        offset_sequence = self.offset(decoded).squeeze(-1)
        onset_age_sequence = self.onset_age(decoded).squeeze(-1)
        offset_age_sequence = self.offset_age(decoded).squeeze(-1)
        return {
            "onset_logits_sequence": onset_sequence,
            "offset_logits_sequence": offset_sequence,
            "onset_age_logits_sequence": onset_age_sequence,
            "offset_age_logits_sequence": offset_age_sequence,
            "onset_logits": onset_sequence[:, -1, :],
            "offset_logits": offset_sequence[:, -1, :],
        }


class SourceTrackingEventStateBuilderV2:
    """Build causal event-state features from predicted boundary logits."""

    @staticmethod
    def from_logits(
        onset_logits_sequence: torch.Tensor,
        offset_logits_sequence: torch.Tensor,
        *,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.5,
        soft_events: bool = False,
        event_state_dim: int = 5,
    ) -> torch.Tensor:
        batch, frames, sources = onset_logits_sequence.shape
        state = onset_logits_sequence.new_zeros(
            (batch, frames, sources, event_state_dim)
        )
        if event_state_dim <= 0:
            return state

        onset_probs = torch.sigmoid(onset_logits_sequence)
        offset_probs = torch.sigmoid(offset_logits_sequence)
        onset_events = onset_probs >= onset_threshold
        offset_events = offset_probs >= offset_threshold

        if frames > 1:
            state[:, 1:, :, 0] = (
                onset_probs[:, :-1, :] if soft_events else onset_events[:, :-1, :].float()
            )
            if event_state_dim > 1:
                state[:, 1:, :, 1] = (
                    offset_probs[:, :-1, :]
                    if soft_events
                    else offset_events[:, :-1, :].float()
                )

        active = onset_logits_sequence.new_zeros((batch, sources))
        last_onset = torch.full(
            (batch, sources), -1, dtype=torch.long, device=state.device
        )
        last_offset = torch.full(
            (batch, sources), -1, dtype=torch.long, device=state.device
        )
        denom = max(1, frames - 1)
        soft_last_onset_age = onset_logits_sequence.new_ones((batch, sources))
        soft_last_offset_age = onset_logits_sequence.new_ones((batch, sources))
        soft_age_step = 1.0 / float(denom)
        for frame_idx in range(frames):
            if event_state_dim > 2:
                state[:, frame_idx, :, 2] = active
            if event_state_dim > 3:
                state[:, frame_idx, :, 3] = (
                    soft_last_onset_age
                    if soft_events
                    else SourceTrackingEventStateBuilderV2._age(
                        last_onset, frame_idx, denom
                    )
                )
            if event_state_dim > 4:
                state[:, frame_idx, :, 4] = (
                    soft_last_offset_age
                    if soft_events
                    else SourceTrackingEventStateBuilderV2._age(
                        last_offset, frame_idx, denom
                    )
                )
            onset_now = onset_events[:, frame_idx, :]
            offset_now = offset_events[:, frame_idx, :]
            last_onset = torch.where(
                onset_now, torch.full_like(last_onset, frame_idx), last_onset
            )
            last_offset = torch.where(
                offset_now, torch.full_like(last_offset, frame_idx), last_offset
            )
            if soft_events:
                offset_prob = offset_probs[:, frame_idx, :]
                onset_prob = onset_probs[:, frame_idx, :]
                active = active * (1.0 - offset_prob)
                active = active + (1.0 - active) * onset_prob
                aged_onset = torch.clamp(soft_last_onset_age + soft_age_step, max=1.0)
                aged_offset = torch.clamp(soft_last_offset_age + soft_age_step, max=1.0)
                soft_last_onset_age = onset_prob * soft_age_step + (
                    1.0 - onset_prob
                ) * aged_onset
                soft_last_offset_age = offset_prob * soft_age_step + (
                    1.0 - offset_prob
                ) * aged_offset
            else:
                active = ((active > 0.5) | onset_now) & ~offset_now
                active = active.float()
        return state

    @staticmethod
    def _age(last_event: torch.Tensor, frame_idx: int, denom: int) -> torch.Tensor:
        frame = torch.full_like(last_event, frame_idx)
        age = (frame - last_event).float() / float(denom)
        return torch.where(last_event >= 0, torch.clamp(age, 0.0, 1.0), torch.ones_like(age))


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
        self.event_decoder = SourceTrackingEventDecoderV2(c)

    def forward(
        self,
        frames: torch.Tensor,
        event_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.input(frames))
        outputs = self.head(encoded)
        outputs.update(self.event_decoder(encoded, event_state))
        return outputs


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
        onset_loss_weight: float | None = None,
        offset_loss_weight: float | None = None,
        onset_focal_gamma: float = 0.0,
        offset_focal_gamma: float = 0.0,
        boundary_timing_loss_weight: float = 0.25,
        boundary_f1_loss_weight: float = 0.0,
        boundary_f1_fp_weight: float = 1.0,
        boundary_f1_fn_weight: float = 1.0,
        hard_boundary_negative_loss_weight: float = 0.0,
        hard_boundary_negative_fraction: float = 0.1,
        onset_pairwise_ranking_loss_weight: float = 0.0,
        onset_pairwise_ranking_margin: float = 0.5,
        onset_pairwise_ranking_max_positives: int = 512,
        onset_pairwise_ranking_max_negatives: int = 2048,
        onset_softmax_loss_weight: float = 0.0,
        onset_sequence_loss_weight: float = 0.0,
        onset_sequence_pairwise_ranking_loss_weight: float = 0.0,
        onset_sequence_block_positive_loss_weight: float = 0.0,
        onset_sequence_block_ranking_loss_weight: float = 0.0,
        onset_nearby_pairwise_ranking_loss_weight: float = 0.0,
        onset_peak_to_shoulder_ranking_loss_weight: float = 0.0,
        onset_shoulder_loss_weight: float = 0.15,
        onset_peak_loss_weight: float = 0.0,
        onset_event_recall_loss_weight: float = 0.0,
        onset_false_peak_loss_weight: float = 0.0,
        onset_peak_radius_frames: int = 2,
        onset_false_peak_fraction: float = 0.02,
    ):
        super().__init__()
        self.activity_pos_weight = activity_pos_weight
        self.inactive_slot_weight = inactive_slot_weight
        self.family_loss_weight = family_loss_weight
        self.count_loss_weight = count_loss_weight
        self.onset_pos_weight = onset_pos_weight
        self.offset_pos_weight = offset_pos_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.onset_loss_weight = (
            boundary_loss_weight if onset_loss_weight is None else onset_loss_weight
        )
        self.offset_loss_weight = (
            boundary_loss_weight if offset_loss_weight is None else offset_loss_weight
        )
        self.onset_focal_gamma = onset_focal_gamma
        self.offset_focal_gamma = offset_focal_gamma
        self.boundary_timing_loss_weight = boundary_timing_loss_weight
        self.boundary_f1_loss_weight = boundary_f1_loss_weight
        self.boundary_f1_fp_weight = boundary_f1_fp_weight
        self.boundary_f1_fn_weight = boundary_f1_fn_weight
        self.hard_boundary_negative_loss_weight = hard_boundary_negative_loss_weight
        self.hard_boundary_negative_fraction = hard_boundary_negative_fraction
        self.onset_pairwise_ranking_loss_weight = onset_pairwise_ranking_loss_weight
        self.onset_pairwise_ranking_margin = onset_pairwise_ranking_margin
        self.onset_pairwise_ranking_max_positives = onset_pairwise_ranking_max_positives
        self.onset_pairwise_ranking_max_negatives = onset_pairwise_ranking_max_negatives
        self.onset_softmax_loss_weight = onset_softmax_loss_weight
        self.onset_sequence_loss_weight = onset_sequence_loss_weight
        self.onset_sequence_pairwise_ranking_loss_weight = (
            onset_sequence_pairwise_ranking_loss_weight
        )
        self.onset_sequence_block_positive_loss_weight = (
            onset_sequence_block_positive_loss_weight
        )
        self.onset_sequence_block_ranking_loss_weight = (
            onset_sequence_block_ranking_loss_weight
        )
        self.onset_nearby_pairwise_ranking_loss_weight = (
            onset_nearby_pairwise_ranking_loss_weight
        )
        self.onset_peak_to_shoulder_ranking_loss_weight = (
            onset_peak_to_shoulder_ranking_loss_weight
        )
        self.onset_shoulder_loss_weight = onset_shoulder_loss_weight
        self.onset_peak_loss_weight = onset_peak_loss_weight
        self.onset_event_recall_loss_weight = onset_event_recall_loss_weight
        self.onset_false_peak_loss_weight = onset_false_peak_loss_weight
        self.onset_peak_radius_frames = onset_peak_radius_frames
        self.onset_false_peak_fraction = onset_false_peak_fraction

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
        onset_loss = self._binary_boundary_loss(
            outputs["onset_logits"],
            targets["onset"].float(),
            pos_weight=self._pos_weight(outputs["onset_logits"], self.onset_pos_weight),
            focal_gamma=self.onset_focal_gamma,
        )
        offset_loss = self._binary_boundary_loss(
            outputs["offset_logits"],
            targets["offset"].float(),
            pos_weight=self._pos_weight(outputs["offset_logits"], self.offset_pos_weight),
            focal_gamma=self.offset_focal_gamma,
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
        boundary_f1_loss = self._boundary_f1_loss(outputs, targets)
        hard_boundary_negative_loss = self._hard_boundary_negative_loss(outputs, targets)
        onset_pairwise_ranking_loss = self._onset_pairwise_ranking_loss(
            outputs,
            targets,
        )
        onset_softmax_loss = self._onset_softmax_loss(outputs, targets)
        onset_sequence_loss = self._onset_sequence_loss(outputs, targets)
        onset_sequence_pairwise_ranking_loss = (
            self._onset_sequence_pairwise_ranking_loss(outputs, targets)
        )
        onset_sequence_block_positive_loss = self._onset_sequence_block_positive_loss(
            outputs,
            targets,
        )
        onset_sequence_block_ranking_loss = self._onset_sequence_block_ranking_loss(
            outputs,
            targets,
        )
        onset_nearby_pairwise_ranking_loss = (
            self._onset_nearby_pairwise_ranking_loss(outputs, targets)
        )
        onset_peak_to_shoulder_ranking_loss = (
            self._onset_peak_to_shoulder_ranking_loss(outputs, targets)
        )
        onset_peak_loss = self._onset_peak_shape_loss(outputs, targets)
        onset_event_recall_loss = self._onset_event_recall_loss(outputs, targets)
        onset_false_peak_loss = self._onset_false_peak_loss(outputs, targets)
        family_loss = self._family_loss(outputs["family_logits"], targets["family"], activity)
        return (
            activity_loss
            + self.count_loss_weight * count_loss
            + self.family_loss_weight * family_loss
            + self.onset_loss_weight * onset_loss
            + self.offset_loss_weight * offset_loss
            + self.boundary_timing_loss_weight * (
                onset_timing_loss + offset_timing_loss
            )
            + self.boundary_f1_loss_weight * boundary_f1_loss
            + self.hard_boundary_negative_loss_weight * hard_boundary_negative_loss
            + self.onset_pairwise_ranking_loss_weight * onset_pairwise_ranking_loss
            + self.onset_softmax_loss_weight * onset_softmax_loss
            + self.onset_sequence_loss_weight * onset_sequence_loss
            + self.onset_sequence_pairwise_ranking_loss_weight
            * onset_sequence_pairwise_ranking_loss
            + self.onset_sequence_block_positive_loss_weight
            * onset_sequence_block_positive_loss
            + self.onset_sequence_block_ranking_loss_weight
            * onset_sequence_block_ranking_loss
            + self.onset_nearby_pairwise_ranking_loss_weight
            * onset_nearby_pairwise_ranking_loss
            + self.onset_peak_to_shoulder_ranking_loss_weight
            * onset_peak_to_shoulder_ranking_loss
            + self.onset_peak_loss_weight * onset_peak_loss
            + self.onset_event_recall_loss_weight * onset_event_recall_loss
            + self.onset_false_peak_loss_weight * onset_false_peak_loss
        )

    def first_pass_boundary_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        onset_loss = self._binary_boundary_loss(
            outputs["onset_logits"],
            targets["onset"].float(),
            pos_weight=self._pos_weight(outputs["onset_logits"], self.onset_pos_weight),
            focal_gamma=self.onset_focal_gamma,
        )
        offset_loss = self._binary_boundary_loss(
            outputs["offset_logits"],
            targets["offset"].float(),
            pos_weight=self._pos_weight(outputs["offset_logits"], self.offset_pos_weight),
            focal_gamma=self.offset_focal_gamma,
        )
        hard_boundary_negative_loss = self._hard_boundary_negative_loss(outputs, targets)
        onset_pairwise_ranking_loss = self._onset_pairwise_ranking_loss(
            outputs,
            targets,
        )
        onset_softmax_loss = self._onset_softmax_loss(outputs, targets)
        onset_sequence_loss = self._onset_sequence_loss(outputs, targets)
        onset_sequence_pairwise_ranking_loss = (
            self._onset_sequence_pairwise_ranking_loss(outputs, targets)
        )
        onset_sequence_block_positive_loss = self._onset_sequence_block_positive_loss(
            outputs,
            targets,
        )
        onset_sequence_block_ranking_loss = self._onset_sequence_block_ranking_loss(
            outputs,
            targets,
        )
        onset_nearby_pairwise_ranking_loss = (
            self._onset_nearby_pairwise_ranking_loss(outputs, targets)
        )
        onset_peak_to_shoulder_ranking_loss = (
            self._onset_peak_to_shoulder_ranking_loss(outputs, targets)
        )
        onset_peak_loss = self._onset_peak_shape_loss(outputs, targets)
        onset_event_recall_loss = self._onset_event_recall_loss(outputs, targets)
        onset_false_peak_loss = self._onset_false_peak_loss(outputs, targets)
        return (
            self.onset_loss_weight * onset_loss
            + self.offset_loss_weight * offset_loss
            + self.hard_boundary_negative_loss_weight * hard_boundary_negative_loss
            + self.onset_pairwise_ranking_loss_weight * onset_pairwise_ranking_loss
            + self.onset_softmax_loss_weight * onset_softmax_loss
            + self.onset_sequence_loss_weight * onset_sequence_loss
            + self.onset_sequence_pairwise_ranking_loss_weight
            * onset_sequence_pairwise_ranking_loss
            + self.onset_sequence_block_positive_loss_weight
            * onset_sequence_block_positive_loss
            + self.onset_sequence_block_ranking_loss_weight
            * onset_sequence_block_ranking_loss
            + self.onset_nearby_pairwise_ranking_loss_weight
            * onset_nearby_pairwise_ranking_loss
            + self.onset_peak_to_shoulder_ranking_loss_weight
            * onset_peak_to_shoulder_ranking_loss
            + self.onset_peak_loss_weight * onset_peak_loss
            + self.onset_event_recall_loss_weight * onset_event_recall_loss
            + self.onset_false_peak_loss_weight * onset_false_peak_loss
        )

    def onset_sequence_only_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        sequence_loss_weight: float,
        sequence_pairwise_ranking_loss_weight: float,
    ) -> torch.Tensor:
        onset_sequence_loss = self._onset_sequence_loss(outputs, targets)
        onset_sequence_pairwise_ranking_loss = (
            self._onset_sequence_pairwise_ranking_loss(outputs, targets)
        )
        return (
            sequence_loss_weight * onset_sequence_loss
            + sequence_pairwise_ranking_loss_weight
            * onset_sequence_pairwise_ranking_loss
        )

    def first_pass_distillation_loss(
        self,
        student: dict[str, torch.Tensor],
        teacher: dict[str, torch.Tensor],
        *,
        final_weight: float,
        sequence_weight: float,
        offset_weight: float,
    ) -> torch.Tensor:
        zero = student["onset_logits"].sum() * 0.0
        final_weight = max(0.0, float(final_weight))
        sequence_weight = max(0.0, float(sequence_weight))
        offset_weight = max(0.0, float(offset_weight))
        if final_weight <= 0.0 and sequence_weight <= 0.0:
            return zero

        loss = zero
        if final_weight > 0.0:
            loss = loss + final_weight * self._logit_distillation_loss(
                student["onset_logits"],
                teacher["onset_logits"],
            )
            loss = loss + final_weight * offset_weight * self._logit_distillation_loss(
                student["offset_logits"],
                teacher["offset_logits"],
            )
        if sequence_weight > 0.0:
            loss = loss + sequence_weight * self._logit_distillation_loss(
                student["onset_logits_sequence"],
                teacher["onset_logits_sequence"],
            )
            loss = (
                loss
                + sequence_weight
                * offset_weight
                * self._logit_distillation_loss(
                    student["offset_logits_sequence"],
                    teacher["offset_logits_sequence"],
                )
            )
        return loss

    @staticmethod
    def _logit_distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            student_logits,
            torch.sigmoid(teacher_logits.detach()),
        )

    def event_age_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        offset_weight: float,
        recent_weight: float = 0.0,
    ) -> torch.Tensor:
        event_state = targets.get("event_state")
        onset_age = outputs.get("onset_age_logits_sequence")
        offset_age = outputs.get("offset_age_logits_sequence")
        if event_state is None or onset_age is None or offset_age is None:
            return outputs["onset_logits"].sum() * 0.0
        onset_target = event_state[..., 3].float()
        offset_target = event_state[..., 4].float()
        onset_loss = self._event_age_loss(onset_age, onset_target, recent_weight)
        offset_loss = self._event_age_loss(offset_age, offset_target, recent_weight)
        return onset_loss + max(0.0, float(offset_weight)) * offset_loss

    @staticmethod
    def _event_age_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        recent_weight: float,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if recent_weight <= 0.0:
            return loss.mean()
        weights = 1.0 + float(recent_weight) * torch.square(1.0 - target)
        return torch.mean(loss * weights) / torch.mean(weights).clamp_min(1e-6)

    def _onset_pairwise_ranking_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs["onset_logits"].reshape(-1)
        target = targets["onset"].float().reshape(-1)
        positives = logits[target > 0.5]
        negatives = logits[target <= 0.5]
        if positives.numel() == 0 or negatives.numel() == 0:
            return logits.sum() * 0.0

        max_positives = max(1, int(self.onset_pairwise_ranking_max_positives))
        if positives.numel() > max_positives:
            positives = torch.topk(-positives, k=max_positives).values.neg()

        negative_fraction = min(1.0, max(0.0, self.hard_boundary_negative_fraction))
        if negative_fraction <= 0.0:
            return logits.sum() * 0.0
        max_negatives = max(1, int(self.onset_pairwise_ranking_max_negatives))
        negative_count = min(
            max_negatives,
            max(1, int(round(negatives.numel() * negative_fraction))),
        )
        hard_negatives = torch.topk(negatives, k=negative_count).values
        margin = max(0.0, float(self.onset_pairwise_ranking_margin))
        return F.softplus(hard_negatives[:, None] - positives[None, :] + margin).mean()

    @staticmethod
    def _onset_softmax_loss(
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs["onset_logits"]
        target = targets["onset"].float()
        single_onset = target.sum(dim=-1) == 1.0
        if not single_onset.any():
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[single_onset],
            target[single_onset].argmax(dim=-1),
        )

    def _onset_sequence_target_and_weights(
        self,
        logits: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = targets["context_onset"].float()
        weights = torch.ones_like(logits)
        nearby = targets.get("context_onset_nearby_mask")
        delta = targets.get("context_onset_delta")
        if nearby is None or delta is None or nearby.shape != logits.shape:
            return target, weights

        radius = max(1, int(self.onset_peak_radius_frames))
        accepted = nearby.float() > 0.5
        center = accepted & (delta.float().abs() < 0.5)
        peak_target = torch.clamp(
            1.0 - delta.float().abs() / float(radius + 1),
            min=0.0,
        )
        target = torch.where(accepted, peak_target, target)
        shoulder_weight = max(0.0, float(self.onset_shoulder_loss_weight))
        shoulder = accepted & ~center
        weights = torch.where(shoulder, weights.new_full((), shoulder_weight), weights)
        return target, weights

    @staticmethod
    def _onset_sequence_hard_negative_mask(
        targets: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor | None:
        target = targets.get("context_onset")
        nearby = targets.get("context_onset_nearby_mask")
        if target is None or nearby is None or nearby.shape != logits.shape:
            return None
        return (target.float() <= 0.5) & (nearby.float() <= 0.5)

    def _onset_sequence_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        target = targets.get("context_onset")
        if logits is None or target is None:
            return outputs["onset_logits"].sum() * 0.0
        censored = self._onset_sequence_censored_loss(logits, targets)
        if censored is not None:
            return censored
        soft_target, weights = self._onset_sequence_target_and_weights(
            logits,
            targets,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            soft_target,
            reduction="none",
            pos_weight=self._pos_weight(logits, self.onset_pos_weight),
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def _onset_sequence_censored_loss(
        self,
        logits: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        nearby = targets.get("context_onset_nearby_mask")
        delta = targets.get("context_onset_delta")
        if nearby is None or delta is None or nearby.shape != logits.shape:
            return None

        accepted = nearby.float() > 0.5
        hard_negative = ~accepted
        total = logits.sum() * 0.0
        normalizer = logits.new_tensor(0.0)

        if hard_negative.any():
            negative_loss = F.binary_cross_entropy_with_logits(
                logits[hard_negative],
                torch.zeros_like(logits[hard_negative]),
                reduction="sum",
            )
            total = total + negative_loss
            normalizer = normalizer + hard_negative.sum().to(logits.dtype)

        center = self._onset_center_mask(targets)
        if center.shape == logits.shape and center.any() and accepted.any():
            radius = int(torch.ceil(delta.float().abs()[accepted].max()).item())
            window_max_probability = self._temporal_max(torch.sigmoid(logits), radius)
            positive_probability = window_max_probability[center].clamp_min(1e-6)
            positive_loss = -torch.log(positive_probability).sum()
            positive_weight = max(0.0, float(self.onset_pos_weight))
            total = total + positive_weight * positive_loss
            normalizer = normalizer + center.sum().to(logits.dtype)

        if normalizer <= 0:
            return logits.sum() * 0.0
        return total / normalizer.clamp_min(1e-6)

    def _onset_sequence_pairwise_ranking_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        target = targets.get("context_onset")
        if logits is None or target is None:
            return outputs["onset_logits"].sum() * 0.0
        negatives = self._onset_sequence_hard_negative_mask(targets, logits)
        if negatives is None:
            negatives = target.float() <= 0.5
        sequence_outputs = {"onset_logits": logits[negatives | (target.float() > 0.5)]}
        sequence_targets = {"onset": target.float()[negatives | (target.float() > 0.5)]}
        return self._onset_pairwise_ranking_loss(sequence_outputs, sequence_targets)

    def _onset_sequence_block_positive_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        if logits is None:
            return outputs["onset_logits"].sum() * 0.0
        block_logits = self._onset_block_max_logits(logits, targets)
        if block_logits.numel() == 0:
            return logits.sum() * 0.0
        target = torch.ones_like(block_logits)
        return F.binary_cross_entropy_with_logits(block_logits, target)

    def _onset_sequence_block_ranking_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        if logits is None:
            return outputs["onset_logits"].sum() * 0.0
        positives = self._onset_block_max_logits(logits, targets)
        hard_negative = self._onset_sequence_hard_negative_mask(targets, logits)
        if (
            positives.numel() == 0
            or hard_negative is None
            or not bool(hard_negative.any().item())
        ):
            return logits.sum() * 0.0
        negatives = logits[hard_negative]
        max_positives = max(1, int(self.onset_pairwise_ranking_max_positives))
        if positives.numel() > max_positives:
            positives = torch.topk(-positives, k=max_positives).values.neg()
        max_negatives = max(1, int(self.onset_pairwise_ranking_max_negatives))
        if negatives.numel() > max_negatives:
            negatives = torch.topk(negatives, k=max_negatives).values
        margin = max(0.0, float(self.onset_pairwise_ranking_margin))
        return F.softplus(negatives[:, None] - positives[None, :] + margin).mean()

    def _onset_block_max_logits(
        self,
        logits: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        nearby = targets.get("context_onset_nearby_mask")
        delta = targets.get("context_onset_delta")
        if nearby is None or delta is None or nearby.shape != logits.shape:
            return logits.new_empty((0,))
        accepted = nearby.float() > 0.5
        center = self._onset_center_mask(targets)
        if center.shape != logits.shape or not center.any() or not accepted.any():
            return logits.new_empty((0,))
        radius = int(torch.ceil(delta.float().abs()[accepted].max()).item())
        masked_logits = torch.where(
            accepted,
            logits,
            logits.new_full((), -torch.inf),
        )
        return self._temporal_max(masked_logits, radius)[center]

    def _onset_nearby_pairwise_ranking_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        target = targets.get("context_onset")
        nearby = targets.get("context_onset_nearby_mask")
        if logits is None or target is None or nearby is None:
            return outputs["onset_logits"].sum() * 0.0

        flat_logits = logits.reshape(-1)
        flat_nearby = nearby.float().reshape(-1)
        positives = flat_logits[flat_nearby > 0.5]
        hard_negative = self._onset_sequence_hard_negative_mask(targets, logits)
        if hard_negative is None:
            return flat_logits.sum() * 0.0
        hard_negatives = logits[hard_negative]
        if positives.numel() == 0 or hard_negatives.numel() == 0:
            return flat_logits.sum() * 0.0

        max_positives = max(1, int(self.onset_pairwise_ranking_max_positives))
        if positives.numel() > max_positives:
            positives = torch.topk(-positives, k=max_positives).values.neg()

        max_negatives = max(1, int(self.onset_pairwise_ranking_max_negatives))
        if hard_negatives.numel() > max_negatives:
            hard_negatives = torch.topk(hard_negatives, k=max_negatives).values

        margin = max(0.0, float(self.onset_pairwise_ranking_margin))
        return F.softplus(
            hard_negatives[:, None] - positives[None, :] + margin
        ).mean()

    def _onset_peak_to_shoulder_ranking_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        nearby = targets.get("context_onset_nearby_mask")
        if logits is None or nearby is None:
            return outputs["onset_logits"].sum() * 0.0

        center = self._onset_center_mask(targets)
        if center.shape != logits.shape:
            return logits.sum() * 0.0
        shoulder = (nearby.float() > 0.5) & ~center
        peak_logits = logits[center]
        shoulder_logits = logits[shoulder]
        if peak_logits.numel() == 0 or shoulder_logits.numel() == 0:
            return logits.sum() * 0.0

        max_positives = max(1, int(self.onset_pairwise_ranking_max_positives))
        if peak_logits.numel() > max_positives:
            peak_logits = torch.topk(-peak_logits, k=max_positives).values.neg()

        max_negatives = max(1, int(self.onset_pairwise_ranking_max_negatives))
        if shoulder_logits.numel() > max_negatives:
            shoulder_logits = torch.topk(shoulder_logits, k=max_negatives).values

        margin = max(0.0, float(self.onset_pairwise_ranking_margin))
        return F.softplus(
            shoulder_logits[:, None] - peak_logits[None, :] + margin
        ).mean()

    def _onset_peak_shape_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        target = targets.get("context_onset")
        if logits is None or target is None:
            return outputs["onset_logits"].sum() * 0.0
        peak_target, weights = self._onset_sequence_target_and_weights(
            logits,
            targets,
        )
        nearby = targets.get("context_onset_nearby_mask")
        if nearby is not None and nearby.shape == logits.shape:
            weights = weights * (nearby.float() > 0.5).float()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            peak_target,
            reduction="none",
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def _onset_event_recall_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        mask = targets.get("context_onset_timing_mask")
        if logits is None or mask is None:
            return outputs["onset_logits"].sum() * 0.0
        probability = torch.sigmoid(logits)
        radius = max(0, int(self.onset_peak_radius_frames))
        center = self._onset_center_mask(targets)
        if not center.any():
            return probability.sum() * 0.0
        window_max = self._temporal_max(probability, radius)
        return (1.0 - window_max[center]).mean()

    def _onset_false_peak_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs.get("onset_logits_sequence")
        if logits is None:
            return outputs["onset_logits"].sum() * 0.0
        probability = torch.sigmoid(logits)
        local_max = probability >= self._temporal_max(probability, 1).detach()
        exclusion = targets.get("context_onset_nearby_mask")
        if exclusion is None:
            exclusion = targets.get("context_onset_timing_mask")
        if exclusion is None:
            exclusion = torch.zeros_like(probability)
        candidates = probability[(exclusion <= 0.5) & local_max]
        if candidates.numel() == 0:
            return probability.sum() * 0.0
        fraction = min(1.0, max(0.0, float(self.onset_false_peak_fraction)))
        k = max(1, int(round(candidates.numel() * fraction)))
        return torch.topk(candidates, k=k).values.mean()

    @staticmethod
    def _temporal_max(values: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return values
        batch, frames, sources = values.shape
        flat = values.permute(0, 2, 1).reshape(batch * sources, 1, frames)
        pooled = F.max_pool1d(flat, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return pooled.reshape(batch, sources, frames).permute(0, 2, 1)

    @staticmethod
    def _onset_center_mask(targets: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = targets.get("context_onset_timing_mask")
        delta = targets.get("context_onset_delta")
        if mask is None:
            return torch.zeros((), dtype=torch.bool)
        if delta is None:
            return mask.float() > 0.5
        return (mask.float() > 0.5) & (delta.float().abs() < 0.5)

    @staticmethod
    def _binary_boundary_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        pos_weight: torch.Tensor,
        focal_gamma: float,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
        gamma = max(0.0, float(focal_gamma))
        if gamma <= 0.0:
            return bce.mean()
        probability = torch.sigmoid(logits)
        p_t = torch.where(target > 0.5, probability, 1.0 - probability)
        return (((1.0 - p_t).clamp_min(0.0) ** gamma) * bce).mean()

    def _boundary_f1_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        onset = self._soft_tversky_loss(
            outputs["onset_logits"],
            targets["onset"].float(),
        )
        offset = self._soft_tversky_loss(
            outputs["offset_logits"],
            targets["offset"].float(),
        )
        return onset + offset

    def _hard_boundary_negative_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        onset = self._hard_negative_loss(
            outputs["onset_logits"],
            targets["onset"].float(),
        )
        offset = self._hard_negative_loss(
            outputs["offset_logits"],
            targets["offset"].float(),
        )
        return onset + offset

    def _hard_negative_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        negative_logits = logits[target < 0.5]
        if negative_logits.numel() == 0:
            return logits.sum() * 0.0
        fraction = min(1.0, max(0.0, self.hard_boundary_negative_fraction))
        if fraction <= 0.0:
            return logits.sum() * 0.0
        k = max(1, int(round(negative_logits.numel() * fraction)))
        hard_logits = torch.topk(negative_logits, k=k).values
        return F.softplus(hard_logits).mean()

    def _soft_tversky_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        axes = tuple(range(1, probability.ndim))
        true_positive = (probability * target).sum(dim=axes)
        false_positive = (probability * (1.0 - target)).sum(dim=axes)
        false_negative = ((1.0 - probability) * target).sum(dim=axes)
        score = (true_positive + 1e-6) / (
            true_positive
            + self.boundary_f1_fp_weight * false_positive
            + self.boundary_f1_fn_weight * false_negative
            + 1e-6
        )
        return (1.0 - score).mean()

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
