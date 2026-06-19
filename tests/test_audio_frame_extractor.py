from __future__ import annotations

import numpy as np

from muziq_nn.datasets.render import AudioFrameExtractorV2, SourceTrackingAudioConfigV2


class TestAudioFrameExtractorV2:
    def test_vectorized_extract_matches_legacy_loop(self):
        rng = np.random.default_rng(7)
        audio = rng.normal(
            0.0,
            0.2,
            SourceTrackingAudioConfigV2.sample_rate // 2 + 17,
        ).astype(np.float32)
        extractor = AudioFrameExtractorV2()

        actual = extractor.extract(audio)
        expected = self._legacy_extract(extractor, audio)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_context_extract_matches_full_slice_with_complete_warmup(self):
        rng = np.random.default_rng(11)
        audio = rng.normal(
            0.0,
            0.2,
            SourceTrackingAudioConfigV2.sample_rate * 2 + 17,
        ).astype(np.float32)
        extractor = AudioFrameExtractorV2()

        full = extractor.extract(audio)
        actual = extractor.extract_context(
            audio,
            end_frame=120,
            frame_count=16,
            peak_warmup_frames=1_000,
        )

        np.testing.assert_allclose(actual, full[105:121], rtol=1e-6, atol=1e-6)

    def test_phase_offset_changes_context_without_changing_shape(self):
        rng = np.random.default_rng(13)
        audio = rng.normal(
            0.0,
            0.2,
            SourceTrackingAudioConfigV2.sample_rate * 2 + 17,
        ).astype(np.float32)
        extractor = AudioFrameExtractorV2()

        baseline = extractor.extract_context(
            audio,
            end_frame=120,
            frame_count=16,
            peak_warmup_frames=64,
        )
        shifted = extractor.extract_context(
            audio,
            end_frame=120,
            frame_count=16,
            peak_warmup_frames=64,
            phase_offset_samples=SourceTrackingAudioConfigV2.hop // 2,
        )

        assert shifted.shape == baseline.shape
        assert float(np.mean(np.abs(shifted - baseline))) > 0.0

    @staticmethod
    def _legacy_extract(
        extractor: AudioFrameExtractorV2, audio: np.ndarray
    ) -> np.ndarray:
        config = extractor.config
        n_frames = int(np.ceil(len(audio) / config.hop))
        ring = np.zeros(config.win, dtype=np.float32)
        frames = np.zeros((n_frames, config.bands), dtype=np.float32)
        peak = 1e-6
        for frame_idx in range(n_frames):
            start = frame_idx * config.hop
            hop_audio = np.zeros(config.hop, dtype=np.float32)
            chunk = audio[start : start + config.hop]
            hop_audio[: len(chunk)] = chunk
            ring = np.roll(ring, -config.hop)
            ring[-config.hop :] = hop_audio
            mag = np.abs(np.fft.rfft(ring * extractor._window))
            bands = extractor._fold @ mag
            log_bands = np.log1p(bands * config.log_feature_scale).astype(np.float32)
            peak = max(float(np.max(log_bands)), peak * 0.9996)
            frames[frame_idx] = np.clip(log_bands / max(peak, 1e-6), 0.0, 1.0)
        return frames
