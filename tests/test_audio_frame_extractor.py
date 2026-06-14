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

    @staticmethod
    def _legacy_extract(
        extractor: AudioFrameExtractorV2, audio: np.ndarray
    ) -> np.ndarray:
        config = extractor.config
        n_frames = int(np.ceil(len(audio) / config.hop))
        ring = np.zeros(config.win, dtype=np.float32)
        frames = np.zeros((n_frames, config.bands), dtype=np.float32)
        peak = np.full(config.bands, 1e-6, dtype=np.float32)
        for frame_idx in range(n_frames):
            start = frame_idx * config.hop
            hop_audio = np.zeros(config.hop, dtype=np.float32)
            chunk = audio[start : start + config.hop]
            hop_audio[: len(chunk)] = chunk
            ring = np.roll(ring, -config.hop)
            ring[-config.hop :] = hop_audio
            mag = np.abs(np.fft.rfft(ring * extractor._window))
            bands = extractor._fold @ mag
            peak = np.maximum(bands, peak * 0.9996)
            frames[frame_idx] = np.clip(bands / np.maximum(peak, 1e-6), 0.0, 1.0)
        return frames
