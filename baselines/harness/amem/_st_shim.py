"""Import sentence-transformers without tripping over memevol's `datasets/`.

ST 5.x imports HuggingFace `datasets` at package-import time. With the project
root on sys.path (always true inside this repo), memevol's benchmark package
`datasets/` shadows the HF library and ST's import dies with
`ImportError: cannot import name 'Dataset'`.

Strategy: temporarily drop project-root entries from sys.path and pop any
already-imported memevol `datasets*` modules, import ST (HF datasets then
resolves from site-packages), then restore everything. ST keeps working via
its already-bound references, and `datasets` goes back to meaning the memevol
package for the rest of the process.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_sentence_transformers() -> None:
    if "sentence_transformers" in sys.modules:
        return
    saved_path = sys.path[:]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _PROJECT_ROOT]
    saved_mods = {}
    for name in [m for m in list(sys.modules) if m == "datasets" or m.startswith("datasets.")]:
        saved_mods[name] = sys.modules.pop(name)
    try:
        import sentence_transformers  # noqa: F401
    finally:
        sys.path[:] = saved_path
        for name in [m for m in list(sys.modules) if m == "datasets" or m.startswith("datasets.")]:
            del sys.modules[name]
        sys.modules.update(saved_mods)
