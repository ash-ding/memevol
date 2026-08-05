#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_venv.sh — create a per-baseline virtualenv from that baseline's OWN
# self-contained requirements.txt.
#
# As of the per-baseline-venvs refactor, EACH baseline has its own venv so its
# (often heavy / mutually-incompatible) dependencies stay isolated and can be
# pinned to that method's paper-faithful versions. The shared `baselines/venv/`
# is DEV/TEST ONLY (runs the shared contract tests + common/ + datasets/); it
# never runs a real baseline.
#
# Usage:
#   baselines/setup_venv.sh <name> [python]
#     <name>   one of: alma cc hipporag2 amem lightmem simplemem zep
#     [python] interpreter to build the venv with (default: python3.12 —
#              falkordblite/zep require 3.12+; all baselines target 3.12)
#
# Examples:
#   baselines/setup_venv.sh cc
#   baselines/setup_venv.sh zep python3.12
#   HIPPORAG_SRC=/path/to/HippoRAG baselines/setup_venv.sh hipporag2
#
# Creates:  baselines/<evolve|harness>/<name>/venv/  (gitignored)
# Run a baseline afterwards (its run.py self-inserts the repo root on sys.path,
# so no PYTHONPATH needed):
#   baselines/harness/cc/venv/bin/python baselines/harness/cc/run.py --config ...
# ---------------------------------------------------------------------------
set -euo pipefail

NAME="${1:-}"
PY="${2:-python3.12}"

if [ -z "$NAME" ]; then
  echo "usage: baselines/setup_venv.sh <alma|cc|hipporag2|amem|lightmem|simplemem|zep> [python]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (baselines/..)

case "$NAME" in
  alma)                                  DIR="$ROOT/baselines/evolve/$NAME" ;;
  cc|hipporag2|amem|lightmem|simplemem|zep) DIR="$ROOT/baselines/harness/$NAME" ;;
  *) echo "unknown baseline: '$NAME' (expected alma|cc|hipporag2|amem|lightmem|simplemem|zep)" >&2; exit 2 ;;
esac

REQ="$DIR/requirements.txt"
VENV="$DIR/venv"

if [ ! -f "$REQ" ]; then
  echo "no requirements.txt at $REQ" >&2
  exit 2
fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python interpreter '$PY' not found — pass one explicitly, e.g. setup_venv.sh $NAME python3.12" >&2
  exit 2
fi

echo "[setup_venv] $NAME  →  $VENV   (python: $PY)"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$REQ"

# hipporag2 depends on the external HippoRAG package, which is NOT vendored and
# NOT a pip line in its requirements.txt — it is an editable install from a
# sibling checkout (memo.py: `from hipporag import HippoRAG`).
if [ "$NAME" = "hipporag2" ]; then
  HIPPORAG_SRC="${HIPPORAG_SRC:-/export/scratch_large/ding/code/HippoRAG}"
  if [ -d "$HIPPORAG_SRC" ]; then
    echo "[setup_venv] hipporag2: pip install -e $HIPPORAG_SRC"
    "$VENV/bin/pip" install -e "$HIPPORAG_SRC"
  else
    echo "[setup_venv] WARNING: HippoRAG source not found at '$HIPPORAG_SRC'." >&2
    echo "             hipporag2 will NOT run until you install it, e.g.:" >&2
    echo "               HIPPORAG_SRC=/path/to/HippoRAG baselines/setup_venv.sh hipporag2" >&2
    echo "             or: $VENV/bin/pip install -e /path/to/HippoRAG" >&2
  fi
fi

echo "[setup_venv] done → $VENV"
echo "[setup_venv] run:  $VENV/bin/python ${DIR#$ROOT/}/run.py --config ${DIR#$ROOT/}/config.example.yaml"
