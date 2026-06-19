import torch

from muziq_nn.models.attention import SourceTrackingLossV2


def test_boundary_positive_examples_are_weighted():
    outputs = {
        "activity_logits": torch.zeros(1, 2),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.tensor([[1.0, 0.0]]),
        "offset": torch.tensor([[0.0, 1.0]]),
    }

    unweighted = SourceTrackingLossV2(
        onset_pos_weight=1.0,
        offset_pos_weight=1.0,
        boundary_loss_weight=1.0,
    )(outputs, targets)
    weighted = SourceTrackingLossV2(
        onset_pos_weight=80.0,
        offset_pos_weight=80.0,
        boundary_loss_weight=1.0,
    )(outputs, targets)

    assert weighted > unweighted + 40


def test_inactive_slot_false_positives_are_weighted():
    outputs = {
        "activity_logits": torch.tensor([[0.0, 3.0]]),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
    }
    targets = {
        "activity": torch.tensor([[1.0, 0.0]]),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
    }

    light_penalty = SourceTrackingLossV2(
        inactive_slot_weight=1.0,
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
    )(outputs, targets)
    heavy_penalty = SourceTrackingLossV2(
        inactive_slot_weight=8.0,
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
    )(outputs, targets)

    assert heavy_penalty > light_penalty * 2


def test_count_loss_penalizes_extra_active_slots():
    targets = {
        "activity": torch.tensor([[1.0, 0.0]]),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
    }
    correct_count = {
        "activity_logits": torch.tensor([[4.0, -4.0]]),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
    }
    extra_count = {
        "activity_logits": torch.tensor([[4.0, 4.0]]),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
    }
    loss = SourceTrackingLossV2(
        inactive_slot_weight=1.0,
        count_loss_weight=5.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
    )

    assert loss(extra_count, targets) > loss(correct_count, targets) * 10


def test_count_head_penalizes_wrong_source_count():
    targets = {
        "activity": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "family": torch.zeros(2, 2),
        "onset": torch.zeros(2, 2),
        "offset": torch.zeros(2, 2),
    }
    correct = {
        "activity_logits": torch.zeros(2, 2),
        "family_logits": torch.zeros(2, 2, 11),
        "onset_logits": torch.zeros(2, 2),
        "offset_logits": torch.zeros(2, 2),
        "count_logits": torch.tensor([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]),
    }
    wrong = {
        **correct,
        "count_logits": torch.tensor([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=5.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
    )

    assert loss(wrong, targets) > loss(correct, targets) + 10
