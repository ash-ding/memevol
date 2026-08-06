#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_venv.sh — build a baseline's isolated environment with uv.
#
# Each baseline is a STANDALONE uv project (its own pyproject.toml +
# .python-version + committed uv.lock). `uv sync` creates .venv/ from the
# lockfile — reproducible, isolated, CPython 3.12.
#
# This script is a thin convenience wrapper over `uv sync` that also handles
# hipporag2's external editable install. You can equally just run uv directly:
#     cd baselines/harness/cc && uv sync
#
# Usage:
#   baselines/setup_venv.sh <name>
#     <name>   one of: alma cc hipporag2 amem lightmem simplemem zep
#
# Examples:
#   baselines/setup_venv.sh cc
#   HIPPORAG_SRC=/path/to/HippoRAG baselines/setup_venv.sh hipporag2
#
# Creates:  baselines/<evolve|harness>/<name>/.venv/  (gitignored)
# Run a baseline afterwards (from its dir, uv picks its .venv automatically):
#   cd baselines/harness/cc && uv run python run.py --config config.example.yaml
# ---------------------------------------------------------------------------
set -euo pipefail

NAME="${1:-}"
if [ -z "$NAME" ]; then
  echo "usage: baselines/setup_venv.sh <alma|cc|hipporag2|amem|lightmem|simplemem|zep>" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found — install it: https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

case "$NAME" in
  alma)                                     DIR="$ROOT/baselines/evolve/$NAME" ;;
  cc|hipporag2|amem|lightmem|simplemem|zep) DIR="$ROOT/baselines/harness/$NAME" ;;
  *) echo "unknown baseline: '$NAME' (expected alma|cc|hipporag2|amem|lightmem|simplemem|zep)" >&2; exit 2 ;;
esac

if [ ! -f "$DIR/pyproject.toml" ]; then
  echo "no pyproject.toml at $DIR" >&2
  exit 2
fi

echo "[setup_venv] $NAME  →  uv sync in $DIR"
( cd "$DIR" && uv sync )

# hipporag2 depends on the external HippoRAG package (NOT vendored, NOT a
# dependency in pyproject.toml — it is an editable install from a sibling
# checkout: memo.py does `from hipporag import HippoRAG`).
if [ "$NAME" = "hipporag2" ]; then
  HIPPORAG_SRC="${HIPPORAG_SRC:-/export/scratch_large/ding/code/HippoRAG}"
  if [ -d "$HIPPORAG_SRC" ]; then
    echo "[setup_venv] hipporag2: uv pip install -e $HIPPORAG_SRC"
    ( cd "$DIR" && uv pip install -e "$HIPPORAG_SRC" )
  else
    echo "[setup_venv] WARNING: HippoRAG source not found at '$HIPPORAG_SRC'." >&2
    echo "             hipporag2 will NOT run until you install it, e.g.:" >&2
    echo "               HIPPORAG_SRC=/path/to/HippoRAG baselines/setup_venv.sh hipporag2" >&2
    echo "             or: (cd $DIR && uv pip install -e /path/to/HippoRAG)" >&2
  fi
fi

echo "[setup_venv] done → $DIR/.venv"
echo "[setup_venv] run:  (cd ${DIR#$ROOT/} && uv run python run.py --config config.example.yaml)"
