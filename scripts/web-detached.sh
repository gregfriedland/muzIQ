#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - "$@" <<'PY'
from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


def latest_checkpoint(repo_root: Path) -> Path | None:
    candidates = list((repo_root / "runs" / "k8s_downloads").glob("*/checkpoint.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


repo_root = Path.cwd()
host = os.environ.get("MUZIQ_WEB_HOST", "127.0.0.1")
port = int(os.environ.get("MUZIQ_WEB_PORT", "8765"))
checkpoint_arg = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
checkpoint = checkpoint_arg or (
    Path(os.environ["MUZIQ_NN_CHECKPOINT"]).expanduser()
    if os.environ.get("MUZIQ_NN_CHECKPOINT")
    else latest_checkpoint(repo_root)
)

if checkpoint is None or not checkpoint.exists():
    raise SystemExit(
        "checkpoint not found; pass a checkpoint path or set MUZIQ_NN_CHECKPOINT"
    )
if port_is_open(host, port):
    raise SystemExit(f"{host}:{port} is already in use")

run_dir = repo_root / "runs" / "web"
run_dir.mkdir(parents=True, exist_ok=True)
log_path = Path(os.environ.get("MUZIQ_WEB_LOG", run_dir / f"preview-{port}.log"))
pid_path = Path(os.environ.get("MUZIQ_WEB_PID", run_dir / f"preview-{port}.pid"))

env = os.environ.copy()
env["MUZIQ_NN_CHECKPOINT"] = str(checkpoint.resolve())
log = log_path.open("ab", buffering=0)
proc = subprocess.Popen(
    ["uv", "run", "muziq-web", "--host", host, "--port", str(port)],
    cwd=repo_root,
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
print(f"started muziq-web at http://{host}:{port}/")
print(f"pid: {proc.pid}")
print(f"log: {log_path}")
print(f"checkpoint: {checkpoint.resolve()}")
PY
