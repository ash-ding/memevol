"""Tests for the hipporag2 harness baseline (baselines/harness/hipporag2/memo.py
— retrieval MemoClass). Zero-dependency runner; run in hipporag2's OWN uv
project (baselines/harness/hipporag2/.venv/), not the repo-root dev env:

    uv run --project baselines/harness/hipporag2 python tests/test_hipporag2_baseline.py
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------- hipporag2 (retrieval MemoClass) --------------------

def test_hipporag_memo_passage_conversion():
    from baselines.harness.hipporag2.memo import _init_to_passages
    # dynamicmem
    ap = _init_to_passages({"app_logs": [{"timestamp": "T", "app_name": "A",
          "api_name": "x", "request": {}, "response": {}, "metadata": {"domain": "d"}}]})
    assert len(ap) == 1 and "App: A" in ap[0]
    # locomo
    lp = _init_to_passages({"conversation": {"speaker_a": "Amy", "speaker_b": "Bob",
          "session_1": [{"speaker": "Amy", "dia_id": "D1:1", "text": "hi"}],
          "session_1_date_time": "2023/01/01"}})
    assert any("hi" in p for p in lp)
    # longmemeval
    sp = _init_to_passages({"sessions": [{"session_id": "s1", "date": "2023/05/20",
          "messages": [{"role": "user", "content": "hello"}]}]})
    assert any("hello" in p for p in sp)


def test_hipporag_memo_retrieve_returns_passages(monkeypatch=None):
    """retrieve_memory_for_query returns retrieved passages as the context dict; the
    shared QA agent (not HippoRAG's own reader) answers."""
    import asyncio
    from baselines.harness.hipporag2.memo import HippoRAGMemo

    class _FakeHippo:
        def __init__(self, **kw): pass
        def index(self, docs): self._docs = docs
        def retrieve(self, queries, num_to_retrieve=5):
            class _S: docs = ["passage about hi"]
            return [_S()]
    memo = HippoRAGMemo(config=dict(embedding="e", llm_model="m", judge_model="j",
                          _hippo_factory=lambda **kw: _FakeHippo()))
    class _Rec:  # minimal recorder
        user_id = "u1"
        init = {"conversation": {"speaker_a": "A", "speaker_b": "B",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}],
                "session_1_date_time": "d"}}
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memo.build_memory_from_data(_Rec()))
    _Rec.init = {"conversation": _Rec.init["conversation"], "query": "what did A say?"}
    out = loop.run_until_complete(memo.retrieve_memory_for_query(_Rec()))
    assert "passages" in out and out["passages"]


def test_internal_llm_key_is_separate_from_the_shared_qa_model():
    """hipporag2 used to build its graph with the frame's `llm_model` — the
    SHARED QA-agent model — which is why it was the only baseline building
    memory with gpt-5-mini while the other six used gpt-4o-mini. It now has its
    own key, falling back to the old behaviour when null."""
    from baselines.harness.hipporag2.memo import HippoRAGMemo

    seen = {}

    class _FakeBaseConfig:
        def __init__(self, **kw): seen.update(kw)

    def _build(cfg):
        seen.clear()
        memo = HippoRAGMemo(config=cfg)
        # Stand in for the `from hipporag import ...` block without needing the
        # (externally-installed) package.
        import types
        fake = types.ModuleType("hipporag")
        fake_utils = types.ModuleType("hipporag.utils")
        fake_cfg = types.ModuleType("hipporag.utils.config_utils")
        fake_cfg.BaseConfig = _FakeBaseConfig
        fake.HippoRAG = lambda global_config=None: object()
        sys.modules.update({"hipporag": fake, "hipporag.utils": fake_utils,
                            "hipporag.utils.config_utils": fake_cfg})
        try:
            memo._ensure_hippo()
        finally:
            for m in ("hipporag", "hipporag.utils", "hipporag.utils.config_utils"):
                sys.modules.pop(m, None)
        return dict(seen)

    base = dict(embedding="text-embedding-3-small", llm_model="gpt-5-mini",
                embedding_batch_size=None, embedding_dtype=None)

    # explicit key wins
    assert _build({**base, "hipporag2_llm_model": "gpt-4o-mini"})["llm_name"] == "gpt-4o-mini"
    # null / absent falls back to the frame model (historical behaviour)
    assert _build({**base, "hipporag2_llm_model": None})["llm_name"] == "gpt-5-mini"
    assert _build(base)["llm_name"] == "gpt-5-mini"


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
