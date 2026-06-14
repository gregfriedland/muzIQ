"""FastAPI app for realtime source-tracking visualization."""

from __future__ import annotations

import argparse
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


class SourceTrackingWebAppV2:
    """Build the HTTP and WebSocket surface for the visualization app."""

    def __init__(self, repo_root: str | Path = "."):
        self.repo_root = Path(repo_root).resolve()
        self.static_dir = Path(__file__).with_name("static")
        self.checkpoints = SourceTrackingCheckpointLocatorV2(self.repo_root)
        self.app = FastAPI(title="muziq-nn source tracker")
        self._mount()

    def _mount(self) -> None:
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")
        self.app.add_api_route("/", self.index, methods=["GET"])
        self.app.add_api_route("/api/status", self.status, methods=["GET"])
        self.app.add_api_websocket_route("/ws/infer", self.infer)

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
                prediction = tracker.append_audio(
                    samples,
                    sample_rate=SourceTrackingAudioConfigV2.sample_rate,
                )
                await websocket.send_json(prediction.to_json())
        except WebSocketDisconnect:
            return


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
