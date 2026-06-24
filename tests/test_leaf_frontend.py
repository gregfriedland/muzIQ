from __future__ import annotations

import torch

from muziq_nn.models.frontend import LeafAudioFrontendV2, LeafFrontendConfigV2


class TestLeafAudioFrontendV2:
    def test_output_shape_uses_context_hop(self):
        config = LeafFrontendConfigV2(filters=96)
        frontend = LeafAudioFrontendV2(config)
        target_frames = 17
        sample_count = config.kernel_size + (target_frames - 1) * config.context_hop_samples
        audio = torch.zeros(2, sample_count)

        features = frontend(audio)

        assert features.shape == (2, target_frames, config.feature_dim)

    def test_metadata_describes_frontend_contract(self):
        config = LeafFrontendConfigV2(
            fine_hop_samples=40,
            context_hop_samples=160,
            onset_shoulder_ms=25.0,
        )
        metadata = LeafAudioFrontendV2(config).metadata()

        assert metadata["frontend_name"] == "leaf"
        assert metadata["sample_rate"] == 16_000
        assert metadata["fine_hop_samples"] == 40
        assert metadata["context_hop_samples"] == 160
        assert metadata["onset_shoulder_ms"] == 25.0

    def test_rejects_precomputed_frame_tensors(self):
        frontend = LeafAudioFrontendV2()

        try:
            frontend(torch.zeros(1, 8, 128))
        except ValueError as exc:
            assert "expects [batch, samples]" in str(exc)
        else:
            raise AssertionError("LEAF frontend accepted a precomputed frame tensor")
