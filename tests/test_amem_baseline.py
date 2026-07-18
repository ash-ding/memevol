"""Tests for the A-mem harness baseline (shim, init→note-unit mapping, hooks).
Zero-dependency runner (no pytest in the venvs) — baselines venv ONLY (heavy
imports: torch/sentence-transformers/litellm; the root venv will fail):

    baselines/venv/bin/python tests/test_amem_baseline.py
"""
import asyncio, json, sys, traceback
from pathlib import Path
from types import SimpleNamespace
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_st_shim_coexists_with_memevol_datasets():
    # Run in the hostile environment (project root on sys.path): the shim must
    # let ST import while `datasets` keeps meaning memevol's benchmark package.
    from baselines.harness.amem._st_shim import ensure_sentence_transformers
    ensure_sentence_transformers()
    ensure_sentence_transformers()   # idempotent
    import sentence_transformers
    from datasets.locomo.env import extract_sessions   # memevol's datasets
    assert hasattr(sentence_transformers, "SentenceTransformer")
    assert callable(extract_sessions)


# -------------------- runner --------------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}"); traceback.print_exc(); failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
