"""FastAPI app for realtime source-tracking visualization."""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from muziq_nn.datasets.render import SourceTrackingAudioConfigV2
from muziq_nn.webapp.inference import (
    RealtimeSourceTrackerV2,
    SourceTrackingCheckpointLocatorV2,
)


class MacAudioDeviceProbeV2:
    """Report local CoreAudio devices visible through FFmpeg AVFoundation."""

    audio_device_pattern = re.compile(r"\]\s+\[(?P<index>\d+)\]\s+(?P<name>.+)$")

    def list_avfoundation_audio_devices(self) -> list[str]:
        if sys.platform != "darwin":
            return []
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-f",
                    "avfoundation",
                    "-list_devices",
                    "true",
                    "-i",
                    "",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []
        devices: list[str] = []
        in_audio_section = False
        for line in result.stderr.splitlines():
            if "AVFoundation audio devices:" in line:
                in_audio_section = True
                continue
            if not in_audio_section:
                continue
            match = self.audio_device_pattern.search(line)
            if match:
                devices.append(match.group("name").strip())
            elif "Error opening" in line:
                break
        return devices


class SourceTrackingWebAppV2:
    """Build the HTTP and WebSocket surface for the visualization app."""

    def __init__(self, repo_root: str | Path = "."):
        self.repo_root = Path(repo_root).resolve()
        self.static_dir = Path(__file__).with_name("static")
        self.checkpoints = SourceTrackingCheckpointLocatorV2(self.repo_root)
        self.audio_probe = MacAudioDeviceProbeV2()
        self.app = FastAPI(title="muziq-nn source tracker")
        self._mount()

    def _mount(self) -> None:
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")
        self.app.add_api_route("/", self.index, methods=["GET"])
        self.app.add_api_route("/api/status", self.status, methods=["GET"])
        self.app.add_api_route("/api/audio-devices", self.audio_devices, methods=["GET"])
        self.app.add_api_websocket_route("/ws/infer", self.infer)
        self.app.add_api_websocket_route("/ws/capture", self.capture)

    async def index(self) -> FileResponse:
        return FileResponse(self.static_dir / "index.html")

    async def status(self, checkpoint: str | None = None) -> dict[str, object]:
        checkpoint_path = self.checkpoints.locate(checkpoint)
        return {
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_found": checkpoint_path is not None,
            "sample_rate": SourceTrackingAudioConfigV2.sample_rate,
            "max_sources": SourceTrackingAudioConfigV2.max_sources,
            "bands": SourceTrackingAudioConfigV2.bands,
            "window_seconds": SourceTrackingAudioConfigV2.duration_s,
        }

    async def audio_devices(self) -> dict[str, object]:
        devices = self.audio_probe.list_avfoundation_audio_devices()
        return {
            "platform": sys.platform,
            "avfoundation_audio_devices": devices,
            "blackhole_present": any("blackhole" in device.lower() for device in devices),
        }

    async def infer(
        self,
        websocket: WebSocket,
        checkpoint: str | None = Query(default=None),
        device: str = Query(default="auto"),
        threshold: float = Query(default=0.35),
    ) -> None:
        await websocket.accept()
        checkpoint_path = self.checkpoints.locate(checkpoint)
        if checkpoint_path is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "No checkpoint found. Set MUZIQ_NN_CHECKPOINT or pass one.",
                }
            )
            await websocket.close()
            return
        tracker = RealtimeSourceTrackerV2(
            checkpoint_path,
            device=device,
            activity_threshold=threshold,
        )
        await websocket.send_json(
            {
                "type": "ready",
                "checkpoint_path": str(checkpoint_path),
                "sample_rate": SourceTrackingAudioConfigV2.sample_rate,
                "max_sources": SourceTrackingAudioConfigV2.max_sources,
            }
        )
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                if "text" in message and message["text"] == "reset":
                    tracker.reset()
                    await websocket.send_json({"type": "reset"})
                    continue
                data = message.get("bytes")
                if data is None:
                    continue
                samples = np.frombuffer(data, dtype="<f4")
                start = time.perf_counter()
                prediction = tracker.append_audio(
                    samples,
                    sample_rate=SourceTrackingAudioConfigV2.sample_rate,
                )
                payload = prediction.to_json()
                payload["latency_ms"] = (time.perf_counter() - start) * 1000
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            return

    async def capture(
        self,
        websocket: WebSocket,
        checkpoint: str | None = Query(default=None),
        capture_device: str = Query(default="BlackHole 2ch"),
        device: str = Query(default="auto"),
        threshold: float = Query(default=0.35),
    ) -> None:
        await websocket.accept()
        checkpoint_path = self.checkpoints.locate(checkpoint)
        if checkpoint_path is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "No checkpoint found. Set MUZIQ_NN_CHECKPOINT or pass one.",
                }
            )
            await websocket.close()
            return
        tracker = RealtimeSourceTrackerV2(
            checkpoint_path,
            device=device,
            activity_threshold=threshold,
        )
        await websocket.send_json(
            {
                "type": "ready",
                "checkpoint_path": str(checkpoint_path),
                "sample_rate": SourceTrackingAudioConfigV2.sample_rate,
                "max_sources": SourceTrackingAudioConfigV2.max_sources,
                "capture_device": capture_device,
            }
        )
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-i",
            f":{capture_device}",
            "-f",
            "f32le",
            "-ar",
            str(SourceTrackingAudioConfigV2.sample_rate),
            "-ac",
            "1",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        chunk_bytes = 1600 * 4
        try:
            while True:
                if process.stdout is None:
                    return
                data = await process.stdout.read(chunk_bytes)
                if not data:
                    stderr = b""
                    if process.stderr is not None:
                        stderr = await process.stderr.read()
                    message = stderr.decode(errors="replace") or "Audio capture ended."
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": message,
                        }
                    )
                    return
                samples = np.frombuffer(data, dtype="<f4")
                start = time.perf_counter()
                prediction = tracker.append_audio(
                    samples,
                    sample_rate=SourceTrackingAudioConfigV2.sample_rate,
                ).to_json()
                prediction["latency_ms"] = (time.perf_counter() - start) * 1000
                prediction["input_rms"] = self._rms(samples)
                await websocket.send_json(prediction)
        except WebSocketDisconnect:
            return
        finally:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        if not len(samples):
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def create_app(repo_root: str | Path = ".") -> FastAPI:
    return SourceTrackingWebAppV2(repo_root).app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(args.repo_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
