"""Tests for the Zep (Graphiti) harness baseline (init→episode mapping, context
rendering, hooks).
Zero-dependency runner (no pytest) — zep's OWN uv project ONLY (heavy imports:
graphiti_core/falkordblite/sentence-transformers; the repo-root project will fail):

    uv run --project baselines/harness/zep python tests/test_zep_baseline.py

Network/LLM calls and the embedded FalkorDB are NOT exercised: Graphiti is
replaced by a fake.
"""
import asyncio, sys, tempfile, traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolved(arm="faithful", **memo):
    """The complete config eval_harness would hand this memo (plus `memo:` overrides)."""
    from baselines.harness import eval_harness as eh
    from baselines.harness.zep import memo as memo_module
    unified_models = ({"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"}
                      if arm == "unified" else None)
    return eh.resolve_memo_config(memo_module.CONFIG_DEFAULTS, memo_module.UNIFIED_MODEL_KEYS,
                                  arm=arm, unified_models=unified_models, overrides=memo)


def test_memo_module_imports():
    # Regression guard: a leftover reference to the deleted `_st_shim` module
    # once made this module raise NameError at import, so zep could not run at all.
    import baselines.harness.zep.memo as zm
    assert zm.ZepMemo.__name__ == "ZepMemo"


# ---- _parse_dt ----

def test_parse_dt_handles_every_benchmark_format():
    from baselines.harness.zep.memo import _parse_dt
    utc = timezone.utc
    # LongMemEval's weekday-in-parens needs fuzzy parsing
    assert _parse_dt("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21, tzinfo=utc)
    assert _parse_dt("7:00 pm on 20 May, 2023") == datetime(2023, 5, 20, 19, 0, tzinfo=utc)
    assert _parse_dt("2024-01-01T08:30:00") == datetime(2024, 1, 1, 8, 30, tzinfo=utc)
    naive = datetime(2024, 1, 1)
    assert _parse_dt(naive).tzinfo is not None


def test_parse_dt_missing_value_falls_back_to_now():
    from baselines.harness.zep.memo import _parse_dt
    before = datetime.now(timezone.utc)
    got = _parse_dt("")
    assert got.tzinfo is not None and got >= before


# ---- _init_to_episodes ----

def test_locomo_one_message_episode_per_turn():
    from baselines.harness.zep.memo import EpisodeType, _init_to_episodes
    conv = {
        "session_2_date_time": "2:00 pm on 9 May, 2023",
        "session_2": [{"speaker": "Bob", "text": "later", "dia_id": "D2:1"}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "Alice", "text": "hi", "dia_id": "D1:1"}],
    }
    eps = _init_to_episodes({"conversation": conv})
    assert [(e["name"], e["body"]) for e in eps] == [("D1:1", "Alice: hi"), ("D2:1", "Bob: later")]
    assert eps[0]["source"] is EpisodeType.message
    assert eps[0]["reference_time"] == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)


def test_longmemeval_one_episode_per_message():
    from baselines.harness.zep.memo import EpisodeType, _init_to_episodes
    init = {"sessions": [{"session_id": "session_001", "date": "2023/05/20 (Sat) 02:21",
                          "messages": [{"role": "user", "content": "hey"},
                                       {"role": "assistant", "content": "hi"}]}]}
    eps = _init_to_episodes(init)
    assert [(e["name"], e["body"]) for e in eps] == [
        ("session_001_0", "user: hey"), ("session_001_1", "assistant: hi")]
    assert all(e["source"] is EpisodeType.message for e in eps)


def test_dynamicmem_text_episode_uses_shared_passage_text():
    from baselines.harness.zep.memo import EpisodeType, _init_to_episodes
    from baselines.harness.hipporag2.memo import app_log_to_passage
    entry = {"app_log_id": 42, "timestamp": "2024-01-01T00:00:00", "app_name": "cal"}
    eps = _init_to_episodes({"app_logs": [entry]})
    assert eps[0]["name"] == "42" and eps[0]["body"] == app_log_to_passage(entry)
    assert eps[0]["source"] is EpisodeType.text


def test_unknown_init_raises():
    from baselines.harness.zep.memo import _init_to_episodes
    try:
        _init_to_episodes({"bogus": 1})
    except KeyError:
        return
    raise AssertionError("expected KeyError for unrecognized init keys")


# ---- _format_context ----

def test_format_context_renders_facts_with_date_ranges_and_entities():
    from baselines.harness.zep.memo import _format_context
    edges = [
        SimpleNamespace(fact="Alice owns a dog", valid_at=datetime(2023, 5, 8), invalid_at=None),
        SimpleNamespace(fact="Bob likes tea", valid_at=None, invalid_at=None),
    ]
    nodes = [SimpleNamespace(name="Alice", summary="a dog owner"), SimpleNamespace(name="Bob", summary="")]
    ctx = _format_context(edges, nodes)
    assert "<FACTS>\n  - Alice owns a dog (Date range: 2023-05-08 - present)\n  - Bob likes tea\n</FACTS>" in ctx
    assert "<ENTITIES>\n  Alice: a dog owner\n  Bob\n</ENTITIES>" in ctx


def test_format_context_empty_is_empty_string():
    from baselines.harness.zep.memo import _format_context
    assert _format_context([], []) == ""


def test_db_root_defaults_to_system_temp():
    from baselines.harness.zep.memo import _db_root
    assert _db_root({"db_root": None}) == Path(tempfile.gettempdir()) / "zep_falkordb"
    assert _db_root({"db_root": "/data"}) == Path("/data") / "zep_falkordb"


# ---- hooks (fake Graphiti, no network) ----

class _FakeGraphiti:
    def __init__(self, edges=(), nodes=()):
        self.episodes, self.searches = [], []
        self._results = SimpleNamespace(edges=list(edges), nodes=list(nodes))
    async def add_episode(self, **kw):
        self.episodes.append(kw)
    async def search_(self, query, config, group_ids):
        self.searches.append((query, config.limit, group_ids))
        return self._results


def _memo_with_fake(config=None, **results):
    from baselines.harness.zep.memo import ZepMemo
    m = ZepMemo(config=_resolved(**(config or {})))
    m._graphiti = _FakeGraphiti(**results)   # pre-set → _ensure no-ops
    return m


def test_build_adds_episodes_in_order_scoped_to_the_instance_group():
    m = _memo_with_fake()
    init = {"sessions": [{"session_id": "s", "date": "2023-05-20",
                          "messages": [{"role": "user", "content": "a"},
                                       {"role": "assistant", "content": "b"}]}]}
    asyncio.run(m.build_memory_from_data(SimpleNamespace(init=init)))
    assert [e["episode_body"] for e in m._graphiti.episodes] == ["user: a", "assistant: b"]
    assert {e["group_id"] for e in m._graphiti.episodes} == {m._gid}


def test_retrieve_limits_to_retrieve_k_and_returns_inline_block():
    edges = [SimpleNamespace(fact=f"f{i}", valid_at=None, invalid_at=None) for i in range(5)]
    m = _memo_with_fake(config={"retrieve_k": 3}, edges=edges)
    out = asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "who?"})))
    assert m._graphiti.searches == [("who?", 3, [m._gid])]
    block = out["inline_memory_blocks"][0]
    assert "  - f2" in block and "  - f3" not in block   # results truncated to k


def test_retrieve_empty_query_or_results_returns_empty_dict():
    m = _memo_with_fake()
    assert asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": ""}))) == {}
    assert m._graphiti.searches == []
    assert asyncio.run(m.retrieve_memory_for_query(SimpleNamespace(init={"query": "q"}))) == {}


# ---- config + contract ----

def test_config_defaults_and_unified_models():
    from baselines.harness.zep.memo import CONFIG_DEFAULTS
    assert CONFIG_DEFAULTS["graph_llm_model"] == "gpt-4o-mini-2024-07-18"
    unified = _resolved("unified")
    assert unified["graph_llm_model"] == "gpt-5-mini"
    assert unified["embedder_model"] == "text-embedding-3-small"
    assert unified["reranker"] == "bge"   # the cross-encoder stays local in both arms


def test_graphiti_telemetry_is_switched_off_explicitly():
    import baselines.harness.zep.memo as zm
    assert zm._graphiti_telemetry.is_telemetry_enabled() is False


def test_ensure_passes_embedding_width_concurrency_and_models_explicitly():
    """The API embedder is told its width (Graphiti otherwise truncates to the
    EMBEDDING_DIM env default, 1024), concurrency is `max_coroutines`, and the
    small model is the graph model — nothing is left to the environment."""
    import redislite.async_falkordb_client as rl
    import graphiti_core.embedder.openai as g_openai
    import baselines.harness.zep.memo as zm
    seen = {}

    class _FakeDB:
        def __init__(self, dbfilename): seen["db"] = dbfilename

    class _FakeEmbedder:
        def __init__(self, config): seen["embedder_config"] = config

    class _FakeGraphiti:
        def __init__(self, **kw): seen["graphiti"] = kw
        async def build_indices_and_constraints(self): pass

    class _FakeLLMClient:
        def __init__(self, config): self.config = config

    patches = [(rl, "AsyncFalkorDB", _FakeDB), (zm, "FalkorDriver", lambda falkor_db: object()),
               (g_openai, "OpenAIEmbedder", _FakeEmbedder), (zm, "Graphiti", _FakeGraphiti),
               (zm, "OpenAIClient", _FakeLLMClient),
               (zm, "CachedBGEReranker", lambda model, device=None: "reranker")]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    for obj, name, value in patches:
        setattr(obj, name, value)
    tmp = tempfile.mkdtemp()
    m = zm.ZepMemo(config=_resolved("unified", db_root=tmp))
    try:
        asyncio.run(m._ensure())
        cfg = seen["embedder_config"]
        assert cfg.embedding_model == "text-embedding-3-small" and cfg.embedding_dim == 1536
        g = seen["graphiti"]
        assert g["max_coroutines"] == 20
        assert g["llm_client"].config.model == "gpt-5-mini"
        assert g["llm_client"].config.small_model == "gpt-5-mini"
        assert g["cross_encoder"] == "reranker"
    finally:
        for obj, name, value in saved:
            setattr(obj, name, value)
        m._db_path = None   # nothing on disk to clean up
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_memo_implements_the_three_hook_contract():
    from common.memo_class import MemoClass
    from baselines.harness.zep.memo import ZepMemo
    assert issubclass(ZepMemo, MemoClass)
    for hook in ("build_memory_from_data", "retrieve_memory_for_query"):
        assert callable(getattr(ZepMemo, hook, None)), hook
    # use_memory_to_answer must NOT be overridden: the shared QA agent answers.
    assert "use_memory_to_answer" not in vars(ZepMemo)


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
