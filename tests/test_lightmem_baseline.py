"""Tests for the LightMem harness baseline (init→turn mapping, config wiring, hooks).
Zero-dependency runner (no pytest) — lightmem's OWN uv project ONLY (heavy
imports: torch/sentence-transformers/llmlingua/qdrant; the repo-root project will fail):

    uv run --project baselines/harness/lightmem python tests/test_lightmem_baseline.py

Network/LLM calls are NOT exercised: the LightMemory system is replaced by a fake.
"""
import asyncio, sys, traceback
from pathlib import Path
from types import SimpleNamespace
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolved(arm="faithful", **memo):
    """The complete config eval_harness would hand this memo (plus `memo:` overrides)."""
    from baselines.harness import eval_harness as eh
    from baselines.harness.lightmem import memo as memo_module
    unified_models = ({"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"}
                      if arm == "unified" else None)
    return eh.resolve_memo_config(memo_module.CONFIG_DEFAULTS, memo_module.UNIFIED_MODEL_KEYS,
                                  arm=arm, unified_models=unified_models, overrides=memo)


# ---- _init_to_turns ----

def test_locomo_turns_carry_speaker_ids_and_parsed_session_time():
    from baselines.harness.lightmem.memo import _init_to_turns
    conv = {
        "speaker_a": "Alice", "speaker_b": "Bob",
        "session_2_date_time": "2:00 pm on 9 May, 2023",
        "session_2": [{"speaker": "Bob", "text": "later"}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "Alice", "text": "hi"}],
    }
    turns = _init_to_turns({"conversation": conv})
    # one [user, assistant] pair per utterance, sessions in numeric order
    assert [t[0]["content"] for t in turns] == ["hi", "later"], turns
    user, assistant = turns[0]
    assert user == {"role": "user", "content": "hi", "time_stamp": "2023-05-08 13:56:00",
                    "speaker_id": "speaker_a", "speaker_name": "Alice"}, user
    assert assistant["role"] == "assistant" and assistant["content"] == ""
    assert turns[1][0]["speaker_id"] == "speaker_b"


def test_locomo_blip_caption_folded_into_content():
    from baselines.harness.lightmem.memo import _init_to_turns
    conv = {"session_1_date_time": "d", "session_1": [
        {"speaker": "A", "text": "look", "blip_caption": "a red bicycle"}]}
    turns = _init_to_turns({"conversation": conv})
    assert turns[0][0]["content"] == "look (image description: a red bicycle)", turns


def test_parse_locomo_timestamp_falls_back_to_input():
    from baselines.harness.lightmem.memo import parse_locomo_timestamp
    assert parse_locomo_timestamp("(7:00 pm on 20 May, 2023)") == "2023-05-20 19:00:00"
    assert parse_locomo_timestamp("not a date") == "not a date"


def test_longmemeval_pairs_user_assistant_and_drops_leading_assistant():
    from baselines.harness.lightmem.memo import _init_to_turns
    init = {"sessions": [{"date": "2023/05/20", "messages": [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "user", "content": "q3"},   # malformed pair
    ]}]}
    turns = _init_to_turns(init)
    assert turns == [[{"role": "user", "content": "q1", "time_stamp": "2023/05/20"},
                      {"role": "assistant", "content": "a1", "time_stamp": "2023/05/20"}]], turns


def test_dynamicmem_uses_shared_passage_text():
    from baselines.harness.lightmem.memo import _init_to_turns
    from baselines.harness.hipporag2.memo import app_log_to_passage
    entry = {"timestamp": "2024-01-01T00:00:00", "app_name": "cal", "api_name": "add"}
    turns = _init_to_turns({"app_logs": [entry]})
    assert turns[0][0]["content"] == app_log_to_passage(entry)
    assert turns[0][0]["time_stamp"] == "2024-01-01T00:00:00"
    assert turns[0][0]["speaker_name"] == "cal"


def test_unknown_init_raises():
    from baselines.harness.lightmem.memo import _init_to_turns
    try:
        _init_to_turns({"bogus": 1})
    except KeyError:
        return
    raise AssertionError("expected KeyError for unrecognized init keys")


# ---- _build_config ----

def test_topic_segment_requires_pre_compress():
    from baselines.harness.lightmem.memo import LightMemMemo
    m = LightMemMemo(config=_resolved(pre_compress=False, topic_segment=True))
    try:
        m._build_config()
    except ValueError as e:
        assert "pre_compress" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_faithful_config_uses_huggingface_embedder_and_sized_qdrant():
    from baselines.harness.lightmem.memo import LightMemMemo
    m = LightMemMemo(config=_resolved())
    cfg = m._build_config()
    assert cfg["text_embedder"]["model_name"] == "huggingface"
    assert cfg["text_embedder"]["configs"]["model"] == "all-MiniLM-L6-v2"
    assert cfg["embedding_retriever"]["configs"]["embedding_model_dims"] == 384
    assert cfg["embedding_retriever"]["configs"]["collection_name"] == m._instance_id
    assert cfg["memory_manager"]["configs"]["model"] == "gpt-4o-mini"
    assert cfg["pre_compressor"]["configs"]["compress_config"]["rate"] == 0.6


def test_unified_config_switches_to_vendored_openai_embedder_with_matching_dims():
    from baselines.harness.lightmem.memo import LightMemMemo
    m = LightMemMemo(config=_resolved("unified"))
    cfg = m._build_config()
    assert cfg["text_embedder"]["model_name"] == "openai"
    assert cfg["text_embedder"]["configs"]["embedding_dims"] == 1536
    assert cfg["embedding_retriever"]["configs"]["embedding_model_dims"] == 1536
    assert cfg["memory_manager"]["configs"]["model"] == "gpt-5-mini"


def test_pre_compress_off_drops_compressor_and_segmenter():
    from baselines.harness.lightmem.memo import LightMemMemo
    cfg = LightMemMemo(config=_resolved(pre_compress=False, topic_segment=False))._build_config()
    assert cfg["pre_compressor"] is None and cfg["topic_segmenter"] is None


# ---- hooks (fake system, no network) ----

class _FakeLightMemory:
    def __init__(self, retrieved=("m1", "m2")):
        self.added, self.retrieved, self.calls = [], list(retrieved), []
    def add_memory(self, messages, force_segment=False, force_extract=False):
        self.added.append((messages[0]["content"], force_segment, force_extract))
    def construct_update_queue_all_entries(self):
        self.calls.append("queue")
    def offline_update_all_entries(self, score_threshold):
        self.calls.append(("update", score_threshold))
    def retrieve(self, query, limit):
        self.calls.append(("retrieve", query, limit))
        return self.retrieved


def _memo_with_fake(config=None, retrieved=("m1", "m2")):
    from baselines.harness.lightmem.memo import LightMemMemo
    m = LightMemMemo(config=_resolved(**(config or {})))
    m._system = _FakeLightMemory(retrieved)   # pre-set → _ensure_system no-ops
    return m


def _sessions(*pairs):
    msgs = []
    for q, a in pairs:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return {"sessions": [{"date": "d", "messages": msgs}]}


def test_build_flushes_only_on_last_turn_then_runs_offline_update():
    m = _memo_with_fake()
    asyncio.run(m.build_memory_from_data(SimpleNamespace(init=_sessions(("q1", "a1"), ("q2", "a2")))))
    assert m._system.added == [("q1", False, False), ("q2", True, True)], m._system.added
    assert m._system.calls == ["queue", ("update", 0.9)], m._system.calls


def test_build_skips_offline_update_when_disabled():
    m = _memo_with_fake(config={"offline_update": False})
    asyncio.run(m.build_memory_from_data(SimpleNamespace(init=_sessions(("q", "a")))))
    assert m._system.calls == []


def test_retrieve_passes_limit_and_wraps_passages():
    m = _memo_with_fake(config={"retrieve_limit": 7})
    out = asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "who?"})))
    assert out == {"passages": ["m1", "m2"]}
    assert m._system.calls == [("retrieve", "who?", 7)]


def test_retrieve_empty_returns_empty_dict():
    m = _memo_with_fake(retrieved=())
    assert asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "q"}))) == {}


# ---- config + contract ----

def test_config_defaults_and_unified_models():
    unified = _resolved("unified")
    assert unified["embedding_model"] == "text-embedding-3-small" and unified["embedding_dims"] == 1536
    assert unified["lightmem_llm_model"] == "gpt-5-mini"
    assert _resolved()["embedding_dims"] == 384


def test_memo_implements_the_two_hook_contract():
    from common.memo_class import MemoClass
    from baselines.harness.lightmem.memo import LightMemMemo
    assert issubclass(LightMemMemo, MemoClass)
    for hook in ("build_memory_from_data", "retrieve_memory_for_query"):
        assert callable(getattr(LightMemMemo, hook, None)), hook
    # Answering is not a hook: the shared QA agent answers for every memo.
    assert not hasattr(LightMemMemo, "use_memory_to_answer")


def test_system_construction_holds_the_model_load_lock():
    """Users' hooks run on worker threads and construction loads Hugging Face
    models (LLMlingua-2, the embedder); two loads at once corrupt each other
    ("Cannot copy out of meta tensor"), so construction takes turns."""
    import threading
    import time
    from baselines.harness import concurrency
    from baselines.harness.lightmem import memo as lm

    active, peak, held = [0], [0], []
    guard = threading.Lock()

    def fake_from_config(config):
        held.append(concurrency.model_load_lock._is_owned())
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.1)
        with guard:
            active[0] -= 1
        return object()

    real = lm.LightMemory.from_config
    lm.LightMemory.from_config = staticmethod(fake_from_config)
    try:
        memos = [lm.LightMemMemo(config=_resolved()) for _ in range(3)]
        threads = [threading.Thread(target=m._ensure_system) for m in memos]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        lm.LightMemory.from_config = real
    assert held == [True, True, True], held
    assert peak[0] == 1, peak[0]


def test_hooks_run_off_the_event_loop():
    """The vendored system is synchronous; the hooks hand it to a worker thread,
    so several users' calls overlap instead of queueing behind one another."""
    import asyncio, time
    from baselines.harness.lightmem.memo import LightMemMemo

    def slow(result):
        def call(recorder):
            time.sleep(0.4)
            return result
        return call

    memos = []
    for _ in range(3):
        m = LightMemMemo(config=_resolved())
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
