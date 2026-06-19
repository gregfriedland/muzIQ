#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if command -v uv >/dev/null 2>&1; then
  exec uv run python -m muziq_nn.datasets.prepare "$@"
fi

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m muziq_nn.datasets.prepare "$@"
