import asyncio
import os
from pathlib import Path

import numpy as np
import torch

from muziq_nn.models.attention import (
    DualPathTransformerSourceTrackerV2,
    SourceTrackingModelConfigV2,
)
from muziq_nn.webapp import app as web_app
from muziq_nn.webapp.app import MacAudioDeviceProbeV2, SourceTrackingWebAppV2, create_app
from muziq_nn.webapp.inference import (
    RealtimeSourceTrackerV2,
    SourceTrackingCheckpointLoaderV2,
    SourceTrackingCheckpointLocatorV2,
)


class TestRealtimeSourceTrackerV2:
    @staticmethod
    def _write_checkpoint(checkpoint_path: Path) -> SourceTrackingModelConfigV2:
        config = SourceTrackingModelConfigV2(model_dim=32, heads=4, layers=1)
        model = DualPathTransformerSourceTrackerV2(config)
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        return config

    def test_loads_checkpoint_and_predicts_sources(self, tmp_path: Path):
        checkpoint_path = tmp_path / "checkpoint.pt"
        config = self._write_checkpoint(checkpoint_path)

        tracker = RealtimeSourceTrackerV2(
            checkpoint_path,
            device="cpu",
            context_frames=64,
        )
        seconds = 1.0
        sample_rate = 16_000
        t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        audio = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)

        prediction = tracker.append_audio(audio, sample_rate)

        assert prediction.frame_count > 0
        assert prediction.sample_count == len(audio)
        assert len(prediction.sources) == config.max_sources
        assert prediction.sources[0].family

    def test_loads_checkpoint_without_count_head(self, tmp_path: Path):
        checkpoint_path = tmp_path / "checkpoint.pt"
        config = SourceTrackingModelConfigV2(model_dim=32, heads=4, layers=1)
        model = DualPathTransformerSourceTrackerV2(config)
        state_dict = {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith("head.count.")
        }
        torch.save({"state_dict": state_dict}, checkpoint_path)

        tracker = RealtimeSourceTrackerV2(
            checkpoint_path,
            device="cpu",
            context_frames=64,
        )
        audio = np.zeros(16_000, dtype=np.float32)

        prediction = tracker.append_audio(audio, 16_000)

        assert tracker.loaded.has_count_head is False
        assert len(prediction.sources) == config.max_sources
        assert isinstance(prediction.source_count, int)

    def test_status_route_reports_explicit_checkpoint(self, tmp_path: Path):
        checkpoint_path = tmp_path / "checkpoint.pt"
        self._write_checkpoint(checkpoint_path)
        app = create_app(tmp_path)
        paths = {getattr(route, "path", "") for route in app.routes}

        payload = asyncio.run(
            SourceTrackingWebAppV2(tmp_path).status(checkpoint=str(checkpoint_path))
        )

        assert "/api/status" in paths
        assert "/api/audio-devices" in paths
        assert "/ws/infer" in paths
        assert "/ws/capture" in paths
        assert payload["checkpoint_found"] is True
        assert payload["checkpoint_path"] == str(checkpoint_path)
        assert payload["sample_rate"] == 16_000

    def test_explicit_checkpoint_wins_over_newer_run_checkpoint(self, tmp_path: Path):
        explicit_path = tmp_path / "explicit.pt"
        run_path = tmp_path / "runs" / "newer" / "checkpoint.pt"
        run_path.parent.mkdir(parents=True)
        self._write_checkpoint(explicit_path)
        self._write_checkpoint(run_path)
        os.utime(explicit_path, (1, 1))
        os.utime(run_path, (2, 2))

        located = SourceTrackingCheckpointLocatorV2(tmp_path).locate(str(explicit_path))

        assert located == explicit_path

    def test_slot_position_uses_identity_only(self):
        assert RealtimeSourceTrackerV2._slot_position(np.array([], dtype=np.float32)) == 0.5
        assert RealtimeSourceTrackerV2._slot_position(np.array([0.0], dtype=np.float32)) == 0.5
        assert RealtimeSourceTrackerV2._slot_position(np.array([2.0], dtype=np.float32)) > 0.9

    def test_event_logits_prefers_latest_sequence_step(self):
        outputs = {
            "onset_logits": torch.tensor([[99.0, 99.0]]),
            "onset_logits_sequence": torch.tensor(
                [[[1.0, 2.0], [3.0, 4.0]]]
            ),
        }

        logits = SourceTrackingCheckpointLoaderV2.event_logits(outputs, "onset")

        assert torch.equal(logits, torch.tensor([[3.0, 4.0]]))


class TestMacAudioDeviceProbeV2:
    def test_parses_avfoundation_audio_devices(self, monkeypatch):
        class Result:
            stderr = "\n".join(
                [
                    "[AVFoundation indev @ 0x0] AVFoundation video devices:",
                    "[AVFoundation indev @ 0x0] [0] FaceTime HD Camera",
                    "[AVFoundation indev @ 0x0] AVFoundation audio devices:",
                    "[AVFoundation indev @ 0x0] [0] BlackHole 2ch",
                    "[AVFoundation indev @ 0x0] [1] MacBook Pro Microphone",
                    "Error opening input file .",
                ]
            )

        def fake_run(*_args, **_kwargs):
            return Result()

        monkeypatch.setattr(web_app.sys, "platform", "darwin")
        monkeypatch.setattr(web_app.subprocess, "run", fake_run)

        devices = MacAudioDeviceProbeV2().list_avfoundation_audio_devices()

        assert devices == ["BlackHole 2ch", "MacBook Pro Microphone"]
