import pytest
import torch

from muziq_nn.models.attention import (
    DualPathTransformerSourceTrackerV2,
    SourceTrackingEventStateBuilderV2,
    SourceTrackingLossV2,
    SourceTrackingModelConfigV2,
)


def test_model_can_ignore_event_state_conditioning():
    torch.manual_seed(1)
    model = DualPathTransformerSourceTrackerV2(
        SourceTrackingModelConfigV2(
            n_bands=4,
            max_sources=2,
            model_dim=16,
            heads=4,
            layers=1,
            event_decoder_layers=1,
            event_decoder_heads=4,
            event_state_conditioning=False,
        )
    )
    model.eval()
    frames = torch.randn(2, 5, 4)
    zeros = torch.zeros(2, 5, 2, 5)
    ones = torch.ones(2, 5, 2, 5)

    with torch.no_grad():
        without_state = model(frames)
        with_zero_state = model(frames, event_state=zeros)
        with_one_state = model(frames, event_state=ones)

    torch.testing.assert_close(
        without_state["onset_logits_sequence"],
        with_zero_state["onset_logits_sequence"],
    )
    torch.testing.assert_close(
        without_state["onset_logits_sequence"],
        with_one_state["onset_logits_sequence"],
    )


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


def test_onset_nearby_pairwise_ranking_loss_compares_shoulders_to_hard_negatives():
    outputs = {
        "activity_logits": torch.full((1, 2), -20.0),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
        "onset_logits_sequence": torch.tensor([[[1.0, 3.0], [4.0, -4.0]]]),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
        "context_onset": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        "context_onset_nearby_mask": torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
    }

    plain = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_nearby_pairwise_ranking_loss_weight=0.0,
    )(outputs, targets)
    ranked = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_nearby_pairwise_ranking_loss_weight=1.0,
        onset_pairwise_ranking_margin=0.5,
    )(outputs, targets)

    assert ranked > plain + 2


def test_onset_sequence_pairwise_ranking_loss_excludes_shoulder_negatives():
    outputs = {
        "activity_logits": torch.full((1, 2), -20.0),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
        "onset_logits_sequence": torch.tensor([[[2.0], [8.0], [-4.0]]]),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
        "context_onset": torch.tensor([[[1.0], [0.0], [0.0]]]),
        "context_onset_nearby_mask": torch.tensor([[[1.0], [1.0], [0.0]]]),
    }

    ranked = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_sequence_pairwise_ranking_loss_weight=1.0,
        onset_pairwise_ranking_margin=0.5,
    )(outputs, targets)

    assert ranked < 1.0


def test_onset_sequence_block_positive_loss_pushes_accepted_block_max_high():
    targets = {
        "activity": torch.zeros(1, 1),
        "family": torch.zeros(1, 1),
        "onset": torch.zeros(1, 1),
        "offset": torch.zeros(1, 1),
        "context_onset": torch.tensor([[[1.0], [1.0], [1.0], [0.0]]]),
        "context_onset_delta": torch.tensor([[[-1.0], [0.0], [1.0], [0.0]]]),
        "context_onset_timing_mask": torch.tensor([[[1.0], [1.0], [1.0], [0.0]]]),
        "context_onset_nearby_mask": torch.tensor([[[1.0], [1.0], [1.0], [0.0]]]),
    }
    common = {
        "activity_logits": torch.full((1, 1), -20.0),
        "family_logits": torch.zeros(1, 1, 11),
        "onset_logits": torch.zeros(1, 1),
        "offset_logits": torch.zeros(1, 1),
    }
    weak = {**common, "onset_logits_sequence": torch.tensor([[[-3.0], [-2.0], [-3.0], [-6.0]]])}
    strong = {**common, "onset_logits_sequence": torch.tensor([[[3.0], [5.0], [3.0], [-6.0]]])}
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_sequence_block_positive_loss_weight=1.0,
    )

    assert loss(weak, targets) > loss(strong, targets) + 1.0


def test_onset_sequence_block_ranking_loss_ranks_block_above_far_negatives():
    targets = {
        "activity": torch.zeros(1, 1),
        "family": torch.zeros(1, 1),
        "onset": torch.zeros(1, 1),
        "offset": torch.zeros(1, 1),
        "context_onset": torch.tensor([[[1.0], [1.0], [1.0], [0.0], [0.0]]]),
        "context_onset_delta": torch.tensor([[[-1.0], [0.0], [1.0], [0.0], [0.0]]]),
        "context_onset_timing_mask": torch.tensor([[[1.0], [1.0], [1.0], [0.0], [0.0]]]),
        "context_onset_nearby_mask": torch.tensor([[[1.0], [1.0], [1.0], [0.0], [0.0]]]),
    }
    common = {
        "activity_logits": torch.full((1, 1), -20.0),
        "family_logits": torch.zeros(1, 1, 11),
        "onset_logits": torch.zeros(1, 1),
        "offset_logits": torch.zeros(1, 1),
    }
    bad = {**common, "onset_logits_sequence": torch.tensor([[[0.0], [1.0], [0.0], [4.0], [3.0]]])}
    good = {**common, "onset_logits_sequence": torch.tensor([[[4.0], [5.0], [4.0], [0.0], [-1.0]]])}
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_sequence_block_ranking_loss_weight=1.0,
        onset_pairwise_ranking_margin=0.5,
    )

    assert loss(bad, targets) > loss(good, targets) + 2.0


def test_onset_peak_to_shoulder_ranking_loss_compares_exact_onset_to_shoulders():
    outputs = {
        "activity_logits": torch.full((1, 2), -20.0),
        "family_logits": torch.zeros(1, 2, 11),
        "onset_logits": torch.zeros(1, 2),
        "offset_logits": torch.zeros(1, 2),
        "onset_logits_sequence": torch.tensor([[[1.0, 3.0], [4.0, -4.0]]]),
    }
    targets = {
        "activity": torch.zeros(1, 2),
        "family": torch.zeros(1, 2),
        "onset": torch.zeros(1, 2),
        "offset": torch.zeros(1, 2),
        "context_onset": torch.tensor([[[1.0, 1.0], [1.0, 0.0]]]),
        "context_onset_delta": torch.tensor([[[0.0, -1.0], [1.0, 0.0]]]),
        "context_onset_timing_mask": torch.tensor([[[1.0, 1.0], [1.0, 0.0]]]),
        "context_onset_nearby_mask": torch.tensor([[[1.0, 1.0], [1.0, 0.0]]]),
    }

    plain = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_peak_to_shoulder_ranking_loss_weight=0.0,
    )(outputs, targets)
    ranked = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_peak_to_shoulder_ranking_loss_weight=1.0,
        onset_pairwise_ranking_margin=0.5,
    )(outputs, targets)

    assert ranked > plain + 2


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


def test_onset_sequence_loss_is_censored_within_accepted_onset_block():
    targets = {
        "activity": torch.zeros(1, 1),
        "family": torch.zeros(1, 1),
        "onset": torch.zeros(1, 1),
        "offset": torch.zeros(1, 1),
        "context_onset": torch.tensor([[[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]]]),
        "context_onset_nearby_mask": torch.tensor(
            [[[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]]]
        ),
        "context_onset_timing_mask": torch.tensor(
            [[[1.0], [1.0], [1.0], [1.0], [1.0], [0.0]]]
        ),
        "context_onset_delta": torch.tensor([[[-2.0], [-1.0], [0.0], [1.0], [2.0], [0.0]]]),
    }
    base_outputs = {
        "activity_logits": torch.full((1, 1), -20.0),
        "family_logits": torch.zeros(1, 1, 11),
        "onset_logits": torch.full((1, 1), -20.0),
        "offset_logits": torch.full((1, 1), -20.0),
    }
    one_inside_hit = {
        **base_outputs,
        "onset_logits_sequence": torch.tensor(
            [[[-8.0], [-8.0], [-8.0], [8.0], [-8.0], [-8.0]]]
        ),
    }
    several_inside_hits = {
        **base_outputs,
        "onset_logits_sequence": torch.tensor([[[-8.0], [8.0], [-8.0], [8.0], [8.0], [-8.0]]]),
    }
    missed_block = {
        **base_outputs,
        "onset_logits_sequence": torch.full((1, 6, 1), -8.0),
    }
    false_positive_outside = {
        **base_outputs,
        "onset_logits_sequence": torch.tensor([[[-8.0], [-8.0], [-8.0], [8.0], [-8.0], [8.0]]]),
    }
    loss = SourceTrackingLossV2(
        count_loss_weight=0.0,
        family_loss_weight=0.0,
        boundary_loss_weight=0.0,
        onset_loss_weight=0.0,
        offset_loss_weight=0.0,
        onset_sequence_loss_weight=1.0,
        onset_pos_weight=1.0,
    )

    one_hit_loss = loss(one_inside_hit, targets)

    torch.testing.assert_close(one_hit_loss, loss(several_inside_hits, targets))
    assert loss(missed_block, targets) > one_hit_loss + 1.0
    assert loss(false_positive_outside, targets) > one_hit_loss + 1.0


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
