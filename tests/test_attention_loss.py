import pytest
import torch

from muziq_nn.models.attention import SourceTrackingEventStateBuilderV2, SourceTrackingLossV2


def test_event_state_builder_can_use_soft_event_channels():
    onset_logits = torch.logit(torch.tensor([[[0.2], [0.8], [0.1]]]))
    offset_logits = torch.logit(torch.tensor([[[0.1], [0.3], [0.7]]]))

    hard = SourceTrackingEventStateBuilderV2.from_logits(
        onset_logits,
        offset_logits,
        onset_threshold=0.5,
        offset_threshold=0.5,
        event_state_dim=5,
    )
    soft = SourceTrackingEventStateBuilderV2.from_logits(
        onset_logits,
        offset_logits,
        onset_threshold=0.5,
        offset_threshold=0.5,
        soft_events=True,
        event_state_dim=5,
    )

    assert hard[0, 1, 0, 0] == 0.0
    assert soft[0, 1, 0, 0] == pytest.approx(0.2)
    assert 0.0 < soft[0, 2, 0, 2] < 1.0
    assert hard[0, 2, 0, 3] == pytest.approx(0.5)
    assert soft[0, 2, 0, 3] == pytest.approx(0.6)


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


def test_boundary_losses_can_weight_onset_more_than_offset():
    outputs = {
        "activity_logits": torch.full((1, 2), -20.0),
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

    offset_heavy = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.5,
        offset_loss_weight=0.5,
    )(outputs, targets)
    offset_light = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.5,
        offset_loss_weight=0.05,
    )(outputs, targets)

    assert offset_light < offset_heavy


def test_event_age_loss_can_weight_recent_events():
    loss_fn = SourceTrackingLossV2()
    outputs = {
        "onset_logits": torch.zeros(1, 1),
        "onset_age_logits_sequence": torch.full((1, 2, 1), 4.0),
        "offset_age_logits_sequence": torch.full((1, 2, 1), 4.0),
    }
    event_state = torch.zeros(1, 2, 1, 5)
    event_state[0, :, 0, 3] = torch.tensor([0.0, 1.0])
    event_state[0, :, 0, 4] = torch.tensor([0.0, 1.0])
    targets = {"event_state": event_state}

    unweighted = loss_fn.event_age_loss(
        outputs,
        targets,
        offset_weight=0.0,
        recent_weight=0.0,
    )
    weighted = loss_fn.event_age_loss(
        outputs,
        targets,
        offset_weight=0.0,
        recent_weight=100.0,
    )

    assert weighted > unweighted


def test_onset_focal_loss_downweights_easy_examples():
    outputs = {
        "activity_logits": torch.full((1, 2), -20.0),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.tensor([[4.0, -4.0]]),
        "offset_logits": torch.zeros(1, 2),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.tensor([[1.0, 0.0]]),
        "offset": torch.zeros(1, 2),
    }

    plain = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        onset_pos_weight=1.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=1.0,
        offset_loss_weight=0.0,
        onset_focal_gamma=0.0,
    )(outputs, targets)
    focal = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        onset_pos_weight=1.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=1.0,
        offset_loss_weight=0.0,
        onset_focal_gamma=2.0,
    )(outputs, targets)

    assert focal < plain * 0.01


def test_onset_sequence_pairwise_ranking_loss_contributes():
    outputs = {
        "activity_logits": torch.full((1, 2), -20.0),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
        "onset_logits_sequence": torch.tensor([[[0.0, 2.0], [-1.0, -2.0]]]),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
        "context_onset": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    }

    plain = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_sequence_pairwise_ranking_loss_weight=0.0,
    )(outputs, targets)
    ranked = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_sequence_pairwise_ranking_loss_weight=1.0,
    )(outputs, targets)

    assert ranked > plain


def test_onset_sequence_only_loss_uses_explicit_weights():
    outputs = {
        "activity_logits": torch.full((1, 2), -20.0),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
        "onset_logits_sequence": torch.tensor([[[0.0, 2.0], [-1.0, -2.0]]]),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
        "context_onset": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
    )

    off = loss.onset_sequence_only_loss(
        outputs,
        targets,
        sequence_loss_weight=0.0,
        sequence_pairwise_ranking_loss_weight=0.0,
    )
    on = loss.onset_sequence_only_loss(
        outputs,
        targets,
        sequence_loss_weight=0.5,
        sequence_pairwise_ranking_loss_weight=1.0,
    )

    assert off == 0.0
    assert on > off


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


def test_boundary_f1_loss_penalizes_false_positive_onsets():
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.tensor([[1.0, 0.0]]),
        "offset": torch.zeros(1, 2),
    }
    clean = {
        "activity_logits": torch.zeros(1, 2),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.tensor([[4.0, -4.0]]),
        "offset_logits": torch.full((1, 2), -4.0),
    }
    false_positive = {
        **clean,
        "onset_logits": torch.tensor([[4.0, 4.0]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        boundary_f1_loss_weight=2.0,
    )

    assert loss(false_positive, targets) > loss(clean, targets)


def test_boundary_f1_loss_penalizes_missed_onsets():
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.tensor([[1.0, 0.0]]),
        "offset": torch.zeros(1, 2),
    }
    detected = {
        "activity_logits": torch.zeros(1, 2),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.tensor([[4.0, -4.0]]),
        "offset_logits": torch.full((1, 2), -4.0),
    }
    missed = {
        **detected,
        "onset_logits": torch.tensor([[-4.0, -4.0]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        boundary_f1_loss_weight=2.0,
    )

    assert loss(missed, targets) > loss(detected, targets)


def test_hard_boundary_negative_loss_focuses_on_worst_false_positive_onsets():
    targets = {
        "activity": torch.zeros(1, 4),
        "family": torch.zeros(1, 4),
        "onset": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "offset": torch.zeros(1, 4),
    }
    clean = {
        "activity_logits": torch.zeros(1, 4),
        "family_logits": torch.zeros(1, 4, 11),
        "onset_logits": torch.tensor([[4.0, -4.0, -4.0, -4.0]]),
        "offset_logits": torch.full((1, 4), -4.0),
    }
    false_positive = {
        **clean,
        "onset_logits": torch.tensor([[4.0, -4.0, 5.0, -4.0]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        hard_boundary_negative_loss_weight=2.0,
        hard_boundary_negative_fraction=0.34,
    )

    assert loss(false_positive, targets) > loss(clean, targets) + 5


def test_onset_pairwise_ranking_loss_penalizes_hard_negative_above_positive():
    targets = {
        "activity": torch.zeros(1, 4),
        "family": torch.zeros(1, 4),
        "onset": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "offset": torch.zeros(1, 4),
    }
    ranked = {
        "activity_logits": torch.zeros(1, 4),
        "family_logits": torch.zeros(1, 4, 11),
        "onset_logits": torch.tensor([[4.0, -4.0, -4.0, -4.0]]),
        "offset_logits": torch.full((1, 4), -4.0),
    }
    misranked = {
        **ranked,
        "onset_logits": torch.tensor([[0.0, -4.0, 4.0, -4.0]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_pairwise_ranking_loss_weight=1.0,
        hard_boundary_negative_fraction=0.34,
    )

    assert loss(misranked, targets) > loss(ranked, targets) + 3


def test_onset_pairwise_ranking_loss_noops_without_positive_onsets():
    outputs = {
        "activity_logits": torch.zeros(1, 2),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.tensor([[4.0, -4.0]]),
        "offset_logits": torch.zeros(1, 2),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_pairwise_ranking_loss_weight=1.0,
    )

    assert torch.isfinite(loss(outputs, targets))


def test_onset_softmax_loss_ranks_single_onset_slot():
    targets = {
        "activity": torch.zeros(2, 3),
        "family": torch.zeros(2, 3),
        "onset": torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        "offset": torch.zeros(2, 3),
    }
    ranked = {
        "activity_logits": torch.full((2, 3), -20.0),
        "family_logits": torch.zeros(2, 3, 11),
        "onset_logits": torch.tensor([[-1.0, 3.0, -2.0], [0.0, 0.0, 0.0]]),
        "offset_logits": torch.full((2, 3), -20.0),
    }
    misranked = {
        **ranked,
        "onset_logits": torch.tensor([[3.0, -1.0, -2.0], [0.0, 0.0, 0.0]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_softmax_loss_weight=1.0,
    )

    assert loss(misranked, targets) > loss(ranked, targets) + 3


def test_onset_sequence_loss_supervises_context_frames():
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
        "context_onset": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    }
    ranked = {
        "activity_logits": torch.full((1, 2), -20.0),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.full((1, 2), -20.0),
        "offset_logits": torch.full((1, 2), -20.0),
        "onset_logits_sequence": torch.tensor([[[3.0, -3.0], [-3.0, -3.0]]]),
    }
    misranked = {
        **ranked,
        "onset_logits_sequence": torch.tensor([[[-3.0, 3.0], [3.0, 3.0]]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_sequence_loss_weight=1.0,
    )

    assert loss(misranked, targets) > loss(ranked, targets) + 3


def test_first_pass_boundary_loss_excludes_activity_family_and_count():
    targets = {
        "activity": torch.tensor([[0.0, 0.0]]),
        "family": torch.zeros(1, 2),
        "onset": torch.tensor([[1.0, 0.0]]),
        "offset": torch.zeros(1, 2),
        "context_onset": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    }
    outputs = {
        "activity_logits": torch.tensor([[20.0, 20.0]]),
        "family_logits": torch.full((1, 2, 11), 20.0),
        "count_logits": torch.tensor([[20.0, -20.0, -20.0]]),
        "onset_logits": torch.tensor([[4.0, -4.0]]),
        "offset_logits": torch.full((1, 2), -4.0),
        "onset_logits_sequence": torch.tensor([[[4.0, -4.0], [-4.0, -4.0]]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=100.0,
        family_loss_weight=100.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=1.0,
        offset_loss_weight=0.0,
        onset_sequence_loss_weight=1.0,
    )

    auxiliary = loss.first_pass_boundary_loss(outputs, targets)

    assert loss(outputs, targets) > auxiliary + 50
    assert auxiliary > 0
