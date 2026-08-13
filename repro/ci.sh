#!/usr/bin/env bash
# Fixed run command for every experiment node. Identical across the tree.
#
# Bootstraps uv if the image lacks it (HF cpu-upgrade ships python:3.12 without
# uv), materialises the locked environment into a single repo-level .venv with
# `uv sync --frozen`, then runs the publication-gate wrapper around `python -m repro.run`.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "[ci] bootstrapping uv via the image's pip"
    python -m pip install --quiet --upgrade uv
fi

echo "[ci] uv $(uv --version)"
echo "[ci] syncing locked environment (uv.lock)"
uv sync --frozen --no-progress

echo "[ci] running publication gate"
uv run python repro/src/run_publication_gate.py
