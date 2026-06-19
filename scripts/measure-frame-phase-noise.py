#!/usr/bin/env python
"""Measure 40-band feature sensitivity to random STFT start phase offsets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from muziq_nn.datasets.midi import MidiIndexV2
from muziq_nn.datasets.nsynth import NsynthIndexV2
from muziq_nn.datasets.render import (
    MidiScheduleStoreV2,
    NsynthNoteStoreV2,
    SourceTrackingAudioConfigV2,
    SourceTrackingRendererV2,
)


@dataclass(frozen=True)
class FramePhaseNoiseConfigV2:
    data_root: str
    split: str
    stages: tuple[str, ...]
    examples_per_stage: int
    phase_variants: int
    max_phase_offset_samples: int
    seed: int
    output: str | None


class FramePhaseNoiseMeasurementV2:
    """Run deterministic feature-difference measurements for phase offsets."""

    def __init__(self, config: FramePhaseNoiseConfigV2):
        self.config = config
        data_root = Path(config.data_root)
        nsynth = NsynthIndexV2(data_root)
        midi = MidiIndexV2(data_root)
        self.renderer = SourceTrackingRendererV2(
            NsynthNoteStoreV2(nsynth),
            MidiScheduleStoreV2(midi),
        )

    def run(self) -> dict[str, object]:
        result = {
            "config": {
                **self.config.__dict__,
                "stages": list(self.config.stages),
            },
            "stages": {
                stage: self._measure_stage(stage) for stage in self.config.stages
            },
        }
        if self.config.output is not None:
            Path(self.config.output).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.output).write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return result

    def _measure_stage(self, stage: str) -> dict[str, float | int]:
        diffs = []
        rng = np.random.default_rng(self.config.seed + self._stage_offset(stage))
        max_offset = min(
            max(0, self.config.max_phase_offset_samples),
            SourceTrackingAudioConfigV2.hop - 1,
        )
        for row in range(self.config.examples_per_stage):
            seed = self.config.seed + row
            baseline = self._frames(stage, seed, phase_offset_samples=0)
            for _ in range(self.config.phase_variants):
                offset = int(rng.integers(0, max_offset + 1))
                shifted = self._frames(stage, seed, phase_offset_samples=offset)
                diffs.append(np.abs(shifted - baseline).reshape(-1))
        if not diffs:
            values = np.zeros(1, dtype=np.float32)
        else:
            values = np.concatenate(diffs).astype(np.float32, copy=False)
        return {
            "examples": self.config.examples_per_stage,
            "phase_variants": self.config.phase_variants,
            "max_phase_offset_samples": max_offset,
            "mean_abs_delta": float(np.mean(values)),
            "rms_delta": float(np.sqrt(np.mean(np.square(values)))),
            "p50_abs_delta": float(np.quantile(values, 0.50)),
            "p95_abs_delta": float(np.quantile(values, 0.95)),
            "p99_abs_delta": float(np.quantile(values, 0.99)),
            "max_abs_delta": float(np.max(values)),
        }

    def _frames(
        self,
        stage: str,
        seed: int,
        phase_offset_samples: int,
    ) -> np.ndarray:
        rendered = self.renderer.render_training_slice(
            stage,
            self.config.split,
            seed,
            frame_count=256,
            peak_warmup_frames=512,
            phase_offset_samples=phase_offset_samples,
        )
        return rendered["frames"]

    @staticmethod
    def _stage_offset(stage: str) -> int:
        return sum(ord(char) for char in stage) * 1009


class FramePhaseNoiseCliV2:
    """CLI wrapper for phase-noise measurement."""

    @staticmethod
    def parse(argv: Sequence[str] | None = None) -> FramePhaseNoiseConfigV2:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--data-root", required=True)
        parser.add_argument("--split", default="train")
        parser.add_argument(
            "--stages",
            default="single_note_all,single_instrument_melody",
        )
        parser.add_argument("--examples-per-stage", type=int, default=128)
        parser.add_argument("--phase-variants", type=int, default=4)
        parser.add_argument(
            "--max-phase-offset-samples",
            type=int,
            default=SourceTrackingAudioConfigV2.hop - 1,
        )
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--output")
        args = parser.parse_args(argv)
        return FramePhaseNoiseConfigV2(
            data_root=args.data_root,
            split=args.split,
            stages=tuple(
                stage.strip() for stage in args.stages.split(",") if stage.strip()
            ),
            examples_per_stage=args.examples_per_stage,
            phase_variants=args.phase_variants,
            max_phase_offset_samples=args.max_phase_offset_samples,
            seed=args.seed,
            output=args.output,
        )

    @staticmethod
    def main(argv: Sequence[str] | None = None) -> int:
        payload = FramePhaseNoiseMeasurementV2(
            FramePhaseNoiseCliV2.parse(argv)
        ).run()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(FramePhaseNoiseCliV2.main(sys.argv[1:]))
