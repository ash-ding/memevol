"""Tests for the MemoryOS harness baseline (imports, init→dialogue-page mapping,
vendored-package identity, hooks). Zero-dependency runner (no pytest in the
venvs) — memoryos's OWN venv (heavy imports: torch/sentence-transformers/faiss):

    uv run --project baselines/harness/memoryos python tests/test_memoryos_baseline.py

Network/LLM calls are NOT exercised here; those need a live key and are covered
by an actual run.
"""
import sys, traceback
from types import SimpleNamespace
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolved(arm="faithful", **memo):
    """The complete config eval_harness would hand this memo (plus `memo:` overrides)."""
    from baselines.harness import eval_harness as eh
    from baselines.harness.memoryos import memo as memo_module
    unified_models = ({"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"}
                      if arm == "unified" else None)
    return eh.resolve_memo_config(memo_module.CONFIG_DEFAULTS, memo_module.UNIFIED_MODEL_KEYS,
                                  arm=arm, unified_models=unified_models, overrides=memo)


def test_sentence_transformers_coexists_with_memevol_benchmarks():
    # Regression guard for the name clash that once needed a sys.modules shim:
    # sentence-transformers eagerly imports HF `datasets`, which collided with
    # memevol's own top-level `datasets/` package. That package was renamed to
    # `benchmarks/` (2026-08-07), so both must now import plainly, side by side.
    import sentence_transformers      # noqa: F401
    from benchmarks.locomo.env import extract_sessions
    assert callable(extract_sessions)


def test_vendored_package_is_the_paper_not_memos():
    # On PyPI, `memoryos` is MemTensor's MemOS (module `memos`) — a DIFFERENT
    # system. Benchmarking it here would silently mis-attribute the numbers.
    from baselines.harness.memoryos import memo as m
    src = Path(m.__file__).resolve().parent / "src" / "memoryos"
    assert (src / "mid_term.py").exists(), src
    assert (src / "long_term.py").exists(), src
    assert "memos" not in str(m.Memoryos.__module__), m.Memoryos.__module__


def test_heat_formula_matches_the_paper():
    # Heat = a*N_visit + b*L_interaction + c*R_recency with a=b=c=1 (paper §4.1).
    from mid_term import compute_segment_heat, HEAT_ALPHA, HEAT_BETA, HEAT_GAMMA
    assert (HEAT_ALPHA, HEAT_BETA, HEAT_GAMMA) == (1.0, 1.0, 1), \
        (HEAT_ALPHA, HEAT_BETA, HEAT_GAMMA)
    session = {"N_visit": 3, "L_interaction": 4, "last_visit_time": None}
    # No last_visit_time -> R_recency defaults to 1.0 -> 3 + 4 + 1 = 8.
    assert compute_segment_heat(session) == 8.0, compute_segment_heat(session)


def test_locomo_turns_pair_into_dialogue_pages():
    # MemoryOS's page unit is a (user_input, agent_response) PAIR; emitting one
    # page per turn would leave every agent_response empty and starve the
    # updater's prompts.
    from baselines.harness.memoryos.memo import _pairs_from_init
    conv = {"session_1_date_time": "1:00 pm on 1 May, 2023", "session_1": [
        {"speaker": "Ann", "text": "one"}, {"speaker": "Bob", "text": "two"},
        {"speaker": "Ann", "text": "three"}]}
    pages = _pairs_from_init({"conversation": conv})
    assert len(pages) == 2, pages
    assert "one" in pages[0][0] and "two" in pages[0][1], pages[0]
    assert pages[1][1] == "", pages[1]          # trailing odd turn
    assert pages[0][2] == "1:00 pm on 1 May, 2023"


def test_locomo_sessions_ordered_numerically():
    from baselines.harness.memoryos.memo import _pairs_from_init
    conv = {}
    for i in (1, 2, 10):
        conv[f"session_{i}_date_time"] = f"day {i}"
        conv[f"session_{i}"] = [{"speaker": "A", "text": f"turn {i}"}]
    pages = _pairs_from_init({"conversation": conv})
    assert [p[0].split("turn ")[1] for p in pages] == ["1", "2", "10"], pages


def test_locomo_image_caption_is_kept():
    from baselines.harness.memoryos.memo import _pairs_from_init
    conv = {"session_1_date_time": "d", "session_1": [
        {"speaker": "A", "text": "", "blip_caption": "a red bicycle"}]}
    pages = _pairs_from_init({"conversation": conv})
    assert "red bicycle" in pages[0][0], pages


def test_longmemeval_pairs_user_then_assistant():
    from baselines.harness.memoryos.memo import _pairs_from_init
    init = {"sessions": [{"date": "2023-05-01", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]}
    pages = _pairs_from_init(init)
    assert pages == [("hi", "hello", "2023-05-01")], pages


def test_dynamicmem_uses_shared_passage_text():
    from baselines.harness.memoryos.memo import _pairs_from_init
    from baselines.harness.hipporag2.memo import app_log_to_passage
    entry = {"app_log_id": "1", "timestamp": "2023-01-01", "app_name": "Mail",
             "api_name": "send", "request": {"to": "x"}}
    pages = _pairs_from_init({"app_logs": [entry]})
    assert pages[0][0] == app_log_to_passage(entry)
    assert pages[0][1] == ""


def test_config_defaults_match_constructor():
    # Every knob the memo passes must be a real Memoryos parameter — the vendored
    # 0.1.0 has no `embedding_model_name` (that belongs to a later build), and a
    # stale key would blow up only at the first user.
    import inspect
    from baselines.harness.memoryos.memo import CONFIG_DEFAULTS, Memoryos
    params = set(inspect.signature(Memoryos.__init__).parameters)
    for key in ("short_term_capacity", "mid_term_capacity", "mid_term_heat_threshold",
                "mid_term_similarity_threshold", "retrieval_queue_capacity",
                "long_term_knowledge_capacity"):
        assert key in CONFIG_DEFAULTS, f"missing from CONFIG_DEFAULTS: {key}"
        assert key in params, f"not a Memoryos parameter: {key}"
    assert "embedding_model_name" not in params, "vendored build unexpectedly gained this knob"


def test_embedder_key_is_applied_by_seeding_the_vendored_model_cache():
    """MemoryOS has NO embedder constructor argument (asserted just above): its
    vendored `get_embedding()` carries `all-MiniLM-L6-v2` as a DEFAULT ARGUMENT,
    so the name the vendored code requests is never the configured one. The key
    is applied by seeding MemoryOS's OWN model cache under that requested name,
    which needs no global constructor patch — one dict entry instead."""
    from baselines.harness.model_config import APIEmbedder
    from baselines.harness.memoryos import memo as memoryos_memo

    assert "memoryos_embedding_model" in memoryos_memo.CONFIG_DEFAULTS

    cache = memoryos_memo._mos_utils._model_cache
    key = memoryos_memo._VENDORED_EMBEDDER_KEY
    real, saved = memoryos_memo.Memoryos, cache.pop(key, None)
    memoryos_memo.Memoryos = lambda **kw: object()
    try:
        m = memoryos_memo.MemoryOSMemo(config=_resolved("unified"))
        m._ensure_system()
        # Seeded under the name the VENDORED code asks for, not the configured one.
        assert isinstance(cache[key], APIEmbedder)
        assert cache[key].model_name == "text-embedding-3-small"
    finally:
        memoryos_memo.Memoryos = real
        cache.pop(key, None)
        if saved is not None:
            cache[key] = saved


def test_seeding_is_idempotent_so_users_share_one_embedder():
    """`_ensure_system` runs per user; the seed must not rebuild the embedder."""
    from baselines.harness.memoryos import memo as memoryos_memo

    cache = memoryos_memo._mos_utils._model_cache
    key = memoryos_memo._VENDORED_EMBEDDER_KEY
    saved = cache.pop(key, None)
    try:
        memoryos_memo._seed_embedder("text-embedding-3-small")
        first = cache[key]
        memoryos_memo._seed_embedder("text-embedding-3-small")
        assert cache[key] is first
    finally:
        cache.pop(key, None)
        if saved is not None:
            cache[key] = saved


def test_get_embedding_is_safe_to_call_from_many_threads():
    """The hooks run on worker threads, and vendored `get_embedding` shares a
    process-global cache whose eviction (past 10000 entries) `del`s a snapshot
    of keys one by one. The memo swaps in ONE locked function for every module
    that imported it, so concurrent calls right at the eviction boundary never
    raise."""
    import threading
    from baselines.harness.concurrency import quiet_stdout
    from baselines.harness.memoryos import memo as mm

    f = mm._mos_utils.get_embedding
    assert getattr(f, "_serialized", False)
    assert mm._mos_long_term.get_embedding is f and mm._mos_mid_term.get_embedding is f

    class _Fake:
        def encode(self, texts, **kw):
            return [[0.0]]

    u, key = mm._mos_utils, mm._VENDORED_EMBEDDER_KEY
    saved_model, saved_cache = u._model_cache.get(key), dict(u._embedding_cache)
    u._model_cache[key] = _Fake()
    u._embedding_cache.clear()
    u._embedding_cache.update({f"pre::{i}": [0.0] for i in range(10000)})
    errors = []

    def worker(n):
        try:
            for i in range(400):
                f(f"text {n} {i}")
        except Exception as e:   # noqa: BLE001
            errors.append(repr(e))

    try:
        with quiet_stdout():
            threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert not errors, errors[:3]
    finally:
        u._embedding_cache.clear()
        u._embedding_cache.update(saved_cache)
        if saved_model is None:
            u._model_cache.pop(key, None)
        else:
            u._model_cache[key] = saved_model


def test_memo_implements_the_two_hook_contract():
    from common.memo_class import MemoClass
    from baselines.harness.memoryos.memo import MemoryOSMemo
    assert issubclass(MemoryOSMemo, MemoClass)
    for hook in ("build_memory_from_data", "retrieve_memory_for_query"):
        assert callable(getattr(MemoryOSMemo, hook, None)), hook
    # MemoryOS ships its own answerer (get_response); it stays unused — the
    # shared QA agent answers, so the comparison is about memory, not about
    # each method's generator.
    assert not hasattr(MemoryOSMemo, "use_memory_to_answer")


def test_hooks_run_off_the_event_loop():
    """The vendored system is synchronous; the hooks hand it to a worker thread,
    so several users' calls overlap instead of queueing behind one another."""
    import asyncio, time
    from baselines.harness.memoryos.memo import MemoryOSMemo

    def slow(result):
        def call(recorder):
            time.sleep(0.4)
            return result
        return call

    memos = []
    for _ in range(3):
        m = MemoryOSMemo(config=_resolved())
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
