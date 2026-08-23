"""Tests for the A-mem harness baseline (imports, init→note-unit mapping, hooks).
Zero-dependency runner (no pytest) — amem's OWN uv project ONLY (heavy
imports: torch/sentence-transformers/litellm; the repo-root project will fail):

    uv run --project baselines/harness/amem python tests/test_amem_baseline.py
"""
import asyncio, json, sys, traceback
from pathlib import Path
from types import SimpleNamespace
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_sentence_transformers_coexists_with_memevol_benchmarks():
    # Regression guard for the name clash that once needed a sys.modules shim:
    # sentence-transformers eagerly imports HF `datasets`, which collided with
    # memevol's own top-level `datasets/` package. That package was renamed to
    # `benchmarks/` (2026-08-07), so both must now import plainly, side by side.
    import sentence_transformers
    from benchmarks.locomo.env import extract_sessions
    assert hasattr(sentence_transformers, "SentenceTransformer")
    assert callable(extract_sessions)


# ---- fakes (no network) ----

class _FakeSystem:
    def __init__(self, ret="MEMSTR"):
        self.calls, self.ret = [], ret
        self.query = self.k = None
    def add_note(self, content, time=None, **kw):
        self.calls.append((content, time))
    def find_related_memories_raw(self, query, k=5):
        self.query, self.k = query, k
        return self.ret


class _FakeLLMController:
    """Stands in for memory_layer.LLMController (memo uses .llm.get_completion)."""
    class _Inner:
        def get_completion(self, prompt, response_format=None, **kw):
            return json.dumps({"keywords": "alpha, beta"})
    def __init__(self):
        self.llm = self._Inner()


def _memo_with_fakes(ret="MEMSTR"):
    from baselines.harness.amem.memo import AMemMemo
    m = AMemMemo()
    m._system = _FakeSystem(ret=ret)          # pre-set → _ensure_system no-ops
    m._retriever_llm = _FakeLLMController()
    return m


# ---- _init_to_note_units ----

def test_units_locomo_verbatim_format():
    from baselines.harness.amem.memo import _init_to_note_units
    conv = {
        "speaker_a": "Alice", "speaker_b": "Bob",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "Alice", "text": "hi"},
                      {"speaker": "Bob", "text": "hello"}],
        "session_2_date_time": "2 pm on 9 May, 2023",
        "session_2": [{"speaker": "Alice", "text": "bye"}],
    }
    units = _init_to_note_units({"conversation": conv})
    # VERBATIM official A-mem LoCoMo unit — "says :" with NO space before "says".
    assert units == [
        ("Speaker Alicesays : hi", "1:56 pm on 8 May, 2023"),
        ("Speaker Bobsays : hello", "1:56 pm on 8 May, 2023"),
        ("Speaker Alicesays : bye", "2 pm on 9 May, 2023"),
    ]


def test_units_longmemeval_per_message():
    from baselines.harness.amem.memo import _init_to_note_units
    init = {"sessions": [{"date": "2023/05/20 (Sat) 02:21",
                          "messages": [{"role": "user", "content": "hey"},
                                       {"role": "assistant", "content": "hi!"}]}]}
    assert _init_to_note_units(init) == [
        ("user: hey", "2023/05/20 (Sat) 02:21"),
        ("assistant: hi!", "2023/05/20 (Sat) 02:21"),
    ]


def test_units_dynamicmem_uses_hipporag2_passage_text():
    from baselines.harness.amem.memo import _init_to_note_units
    from baselines.harness.hipporag2.memo import app_log_to_passage
    entry = {"timestamp": "2024-01-01T00:00:00", "app_name": "cal",
             "api_name": "add", "request": {"a": 1}, "response": {"ok": True},
             "metadata": {"domain": "productivity"}}
    assert _init_to_note_units({"app_logs": [entry]}) == [
        (app_log_to_passage(entry), "2024-01-01T00:00:00")]


def test_units_unknown_init_raises():
    from baselines.harness.amem.memo import _init_to_note_units
    try:
        _init_to_note_units({"bogus": 1})
    except KeyError:
        return
    raise AssertionError("expected KeyError for unrecognized init keys")


# ---- hooks (fakes, no network) ----

def test_build_adds_one_note_per_unit_in_order():
    m = _memo_with_fakes()
    rec = SimpleNamespace(init={"sessions": [
        {"date": "d1", "messages": [{"role": "user", "content": "a"},
                                    {"role": "assistant", "content": "b"}]}]})
    asyncio.run(m.build_memory_from_data(rec))
    assert m._system.calls == [("user: a", "d1"), ("assistant: b", "d1")]


def test_retrieve_rewrites_query_and_wraps_memory_string():
    m = _memo_with_fakes(ret="MEMSTR")
    out = asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "who?"})))
    assert out == {"memories": "MEMSTR"}
    assert m._system.query == "alpha, beta"   # keywords used, NOT the raw question
    assert m._system.k == 10                  # upstream default retrieve_k


def test_retrieve_respects_config_retrieve_k():
    m = _memo_with_fakes()
    m.config = {"retrieve_k": 4}              # per-instance config (constructor-injected in prod)
    asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "q"})))
    assert m._system.k == 4


def test_retrieve_empty_store_returns_empty_dict():
    # upstream find_related_memories_raw returns [] when the store is empty
    m = _memo_with_fakes(ret=[])
    out = asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "q"})))
    assert out == {}


# ---- configurable embedder (issue #26) ----

def test_embedder_key_reaches_the_system():
    """A-mem's embedder IS a constructor parameter (AgenticMemorySystem's
    `model_name`), so the config key needs no extra plumbing: the name flows to
    SimpleEmbeddingRetriever, which calls the patched SentenceTransformer
    factory, which dispatches local-vs-API on that same name."""
    from baselines.harness.amem import memo as amem_memo

    built = {}

    class _FakeAgenticMemorySystem:
        def __init__(self, model_name=None, llm_backend=None, llm_model=None):
            built.update(model_name=model_name, llm_model=llm_model)

    real_system, real_llm = amem_memo.AgenticMemorySystem, amem_memo.LLMController
    amem_memo.AgenticMemorySystem = _FakeAgenticMemorySystem
    amem_memo.LLMController = lambda **kw: object()
    try:
        m = amem_memo.AMemMemo(config={"amem_llm_model": "gpt-5-mini",
                                       "amem_embedding_model": "text-embedding-3-small"})
        m._ensure_system()
        assert built == {"model_name": "text-embedding-3-small", "llm_model": "gpt-5-mini"}

        # absent key keeps A-mem's published embedder
        built.clear()
        m2 = amem_memo.AMemMemo(config={"amem_llm_model": "gpt-4o-mini"})
        m2._ensure_system()
        assert built["model_name"] == "all-MiniLM-L6-v2"
    finally:
        amem_memo.AgenticMemorySystem, amem_memo.LLMController = real_system, real_llm


# NOTE: the `hf_datasets_active()` tests that used to live here were deleted with
# the shim itself. They asserted that A-mem ops ran inside a context manager that
# swapped memevol's `datasets` package for HF's; the `benchmarks/` rename removed
# the clash and PR #20 removed the machinery. What they covered beyond that
# assertion — build's note ordering and retrieve's return shape — is already
# covered by test_build_adds_one_note_per_unit_in_order and
# test_retrieve_rewrites_query_and_wraps_memory_string above.


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
