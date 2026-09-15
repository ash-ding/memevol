"""Tests for the SimpleMem harness baseline (init→Dialogue mapping, config wiring, hooks).
Zero-dependency runner (no pytest) — simplemem's OWN uv project ONLY (heavy
imports: torch/sentence-transformers/lancedb; the repo-root project will fail):

    uv run --project baselines/harness/simplemem python tests/test_simplemem_baseline.py

Network/LLM calls are NOT exercised: SimpleMemSystem is replaced by a fake.
"""
import asyncio, os, shutil, sys, tempfile, traceback
from pathlib import Path
from types import SimpleNamespace
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolved(arm="faithful", **memo):
    """The complete config eval_harness would hand this memo (plus `memo:` overrides)."""
    from baselines.harness import eval_harness as eh
    from baselines.harness.simplemem import memo as memo_module
    unified_models = ({"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"}
                      if arm == "unified" else None)
    return eh.resolve_memo_config(memo_module.CONFIG_DEFAULTS, memo_module.UNIFIED_MODEL_KEYS,
                                  arm=arm, unified_models=unified_models, overrides=memo)


# ---- _init_to_dialogues ----

def test_locomo_dialogues_carry_speaker_and_session_time_in_order():
    from baselines.harness.simplemem.memo import _init_to_dialogues
    conv = {
        "session_2_date_time": "2 pm on 9 May, 2023",
        "session_2": [{"speaker": "Bob", "text": "later"}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "Alice", "text": "hi"}],
    }
    dialogues, next_id = _init_to_dialogues({"conversation": conv}, start_id=1)
    assert [(d.dialogue_id, d.speaker, d.content, d.timestamp) for d in dialogues] == [
        (1, "Alice", "hi", "1:56 pm on 8 May, 2023"),
        (2, "Bob", "later", "2 pm on 9 May, 2023"),
    ]
    assert next_id == 3


def test_dialogue_ids_continue_across_build_calls():
    # DynamicMem hands BUILD one checkpoint delta at a time; ids must not restart.
    from baselines.harness.simplemem.memo import _init_to_dialogues
    first, nid = _init_to_dialogues({"app_logs": [{"app_name": "a"}, {"app_name": "b"}]}, 1)
    second, nid = _init_to_dialogues({"app_logs": [{"app_name": "c"}]}, nid)
    assert [d.dialogue_id for d in first + second] == [1, 2, 3] and nid == 4


def test_longmemeval_messages_keep_role_and_empty_date_becomes_none():
    from baselines.harness.simplemem.memo import _init_to_dialogues
    init = {"sessions": [{"messages": [{"role": "user", "content": "hey"},
                                       {"role": "assistant", "content": "hi"}]}]}
    dialogues, _ = _init_to_dialogues(init, 1)
    assert [(d.speaker, d.content, d.timestamp) for d in dialogues] == [
        ("user", "hey", None), ("assistant", "hi", None)]


def test_dynamicmem_uses_shared_passage_text():
    from baselines.harness.simplemem.memo import _init_to_dialogues
    from baselines.harness.hipporag2.memo import app_log_to_passage
    entry = {"timestamp": "2024-01-01T00:00:00", "app_name": "cal", "api_name": "add"}
    dialogues, _ = _init_to_dialogues({"app_logs": [entry]}, 1)
    assert dialogues[0].speaker == "cal"
    assert dialogues[0].content == app_log_to_passage(entry)
    assert dialogues[0].timestamp == "2024-01-01T00:00:00"


def test_unknown_init_raises():
    from baselines.harness.simplemem.memo import _init_to_dialogues
    try:
        _init_to_dialogues({"bogus": 1}, 1)
    except KeyError:
        return
    raise AssertionError("expected KeyError for unrecognized init keys")


# ---- _entry_to_passage ----

def test_entry_to_passage_renders_only_present_fields():
    from baselines.harness.simplemem.memo import MemoryEntry, _entry_to_passage
    full = MemoryEntry(lossless_restatement="Alice met Bob.", timestamp="2025-11-15T14:30:00",
                       location="Shanghai", persons=["Alice", "Bob"], entities=["XYZ"],
                       topic="meeting")
    assert _entry_to_passage(full) == (
        "Content: Alice met Bob.\nTime: 2025-11-15T14:30:00\nLocation: Shanghai\n"
        "Persons: Alice, Bob\nRelated Entities: XYZ\nTopic: meeting")
    assert _entry_to_passage(MemoryEntry(lossless_restatement="bare")) == "Content: bare"


# ---- _ensure_system wiring (fake SimpleMemSystem) ----

def test_ensure_system_pins_settings_explicitly_and_builds_a_fresh_store():
    from baselines.harness.simplemem import memo as sm
    built = {}

    class _FakeSystem:
        def __init__(self, **kw):
            built.update(kw)

    tmp = Path(tempfile.mkdtemp())
    real_system, real_outputs = sm.SimpleMemSystem, sm.OUTPUTS_DIR
    env_before = dict(os.environ)
    sm.SimpleMemSystem, sm.OUTPUTS_DIR = _FakeSystem, tmp
    os.environ["WINDOW_SIZE"] = "99"          # ambient value that must NOT win
    try:
        m = sm.SimpleMemMemo(config=_resolved(window_size=7))
        m._ensure_system()
        settings = sm._simplemem_settings
        assert settings.WINDOW_SIZE == 7, "the memo config, not the environment"
        assert settings.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
        assert settings.LLM_MODEL == "gpt-4.1-mini"
        assert settings.USE_STREAMING is False and settings.USE_JSON_FORMAT is False
        assert built["model"] == "gpt-4.1-mini" and built["api_key"] is None
        assert built["clear_db"] is True
        assert built["db_path"] == str(tmp / m._instance_id)
        assert built["enable_parallel_processing"] is True and built["max_parallel_workers"] == 16
        os.environ.pop("WINDOW_SIZE")
        assert dict(os.environ) == env_before, "the memo wrote to the environment"
    finally:
        sm.SimpleMemSystem, sm.OUTPUTS_DIR = real_system, real_outputs
        os.environ.clear(); os.environ.update(env_before)
        shutil.rmtree(tmp, ignore_errors=True)


# ---- hooks (fake system, no network) ----

class _FakeRetriever:
    def __init__(self, entries):
        self.entries, self.queries = entries, []
    def retrieve(self, query):
        self.queries.append(query)
        return self.entries


class _FakeSystem:
    def __init__(self, entries=()):
        self.events = []
        self.hybrid_retriever = _FakeRetriever(list(entries))
    def add_dialogues(self, dialogues):
        self.events.append(("add", [d.dialogue_id for d in dialogues]))
    def finalize(self):
        self.events.append("finalize")


def _memo_with_fake(entries=()):
    from baselines.harness.simplemem.memo import SimpleMemMemo
    m = SimpleMemMemo(config=_resolved())
    m._system = _FakeSystem(entries)   # pre-set → _ensure_system no-ops
    return m


def test_build_adds_then_finalizes_and_ids_advance():
    m = _memo_with_fake()
    logs = lambda *names: SimpleNamespace(init={"app_logs": [{"app_name": n} for n in names]})
    asyncio.run(m.build_memory_from_data(logs("a", "b")))
    asyncio.run(m.build_memory_from_data(logs("c")))
    assert m._system.events == [("add", [1, 2]), "finalize", ("add", [3]), "finalize"]


def test_retrieve_formats_entries_as_passages():
    from baselines.harness.simplemem.memo import MemoryEntry
    m = _memo_with_fake([MemoryEntry(lossless_restatement="fact", topic="t")])
    out = asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "who?"})))
    assert out == {"passages": ["Content: fact\nTopic: t"]}
    assert m._system.hybrid_retriever.queries == ["who?"]


def test_retrieve_empty_returns_empty_dict():
    m = _memo_with_fake()
    assert asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "q"}))) == {}


# ---- config + contract ----

def test_config_defaults_and_unified_models():
    from baselines.harness.simplemem.memo import CONFIG_DEFAULTS
    assert CONFIG_DEFAULTS["window_size"] == 20   # the paper's W, not the code's 40
    unified = _resolved("unified")
    assert unified["simplemem_llm_model"] == "gpt-5-mini"
    assert unified["embedding_model"] == "text-embedding-3-small"


def test_memo_implements_the_three_hook_contract():
    from common.memo_class import MemoClass
    from baselines.harness.simplemem.memo import SimpleMemMemo
    assert issubclass(SimpleMemMemo, MemoClass)
    for hook in ("build_memory_from_data", "retrieve_memory_for_query"):
        assert callable(getattr(SimpleMemMemo, hook, None)), hook
    # use_memory_to_answer must NOT be overridden: the shared QA agent answers.
    assert "use_memory_to_answer" not in vars(SimpleMemMemo)


def test_hooks_run_off_the_event_loop():
    """The vendored system is synchronous; the hooks hand it to a worker thread,
    so several users' calls overlap instead of queueing behind one another."""
    import asyncio, time
    from baselines.harness.simplemem.memo import SimpleMemMemo

    def slow(result):
        def call(recorder):
            time.sleep(0.4)
            return result
        return call

    memos = []
    for _ in range(3):
        m = SimpleMemMemo(config=_resolved())
        m._build, m._retrieve = slow(None), slow({})   # stand-ins for the synchronous bodies
        memos.append(m)
    rec = SimpleNamespace(init={"query": "q"})

    async def all_users():
        t0 = time.perf_counter()
        await asyncio.gather(*(m.build_memory_from_data(rec) for m in memos))
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = await asyncio.gather(*(m.retrieve_memory_for_query(rec) for m in memos))
        return build, time.perf_counter() - t0, out

    build, retrieve, out = asyncio.run(all_users())
    assert build < 0.9 and retrieve < 0.9, (build, retrieve)   # serial would take 1.2s each
    assert out == [{}, {}, {}]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
