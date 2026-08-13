"""Tests for the MemoryOS harness baseline (imports, init→dialogue-page mapping,
vendored-package identity, hooks). Zero-dependency runner (no pytest in the
venvs) — memoryos's OWN venv (heavy imports: torch/sentence-transformers/faiss):

    uv run --project baselines/harness/memoryos python tests/test_memoryos_baseline.py

Network/LLM calls are NOT exercised here; those need a live key and are covered
by an actual run.
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def test_run_config_keys_match_constructor():
    # Every knob run.py passes must be a real Memoryos parameter — the vendored
    # 0.1.0 has no `embedding_model_name` (that belongs to a later build), and a
    # stale key would blow up only at the first user.
    import inspect
    from baselines.harness.memoryos.memo import Memoryos
    from baselines.harness.memoryos.run import REQUIRED_KEYS
    params = set(inspect.signature(Memoryos.__init__).parameters)
    for key in ("short_term_capacity", "mid_term_capacity", "mid_term_heat_threshold",
                "mid_term_similarity_threshold", "retrieval_queue_capacity",
                "long_term_knowledge_capacity"):
        assert key in REQUIRED_KEYS, f"missing from REQUIRED_KEYS: {key}"
        assert key in params, f"not a Memoryos parameter: {key}"
    assert "embedding_model_name" not in params, "vendored build unexpectedly gained this knob"


def test_embedder_key_is_applied_by_overriding_the_shared_factory():
    """MemoryOS has NO embedder constructor argument (asserted just above): its
    vendored `get_embedding()` carries `all-MiniLM-L6-v2` as a DEFAULT ARGUMENT,
    so the name the vendored code requests is never the configured one. The key
    is therefore applied by OVERRIDING the shared embedder factory."""
    from baselines.harness import model_config as mc
    from baselines.harness.memoryos import memo as memoryos_memo
    from baselines.harness.memoryos.run import REQUIRED_KEYS

    assert "memoryos_embedding_model" in REQUIRED_KEYS

    real = memoryos_memo.Memoryos
    memoryos_memo.Memoryos = lambda **kw: object()
    try:
        m = memoryos_memo.MemoryOSMemo(
            config={"memoryos_embedding_model": "text-embedding-3-small"})
        m._ensure_system()
        assert mc._policy["model"] == "text-embedding-3-small"

        m2 = memoryos_memo.MemoryOSMemo(config={})
        m2._memo = None
        m2._ensure_system()
        assert mc._policy["model"] == "all-MiniLM-L6-v2"   # the paper's embedder
    finally:
        memoryos_memo.Memoryos = real
        mc.set_embedder_policy(None, None)


def test_memo_implements_the_three_hook_contract():
    from common.memo_class import MemoClass
    from baselines.harness.memoryos.memo import MemoryOSMemo
    assert issubclass(MemoryOSMemo, MemoClass)
    for hook in ("build_memory_from_data", "retrieve_memory_for_query"):
        assert callable(getattr(MemoryOSMemo, hook, None)), hook
    # MemoryOS ships its own answerer (get_response); it must stay unused so the
    # comparison is about memory, not about each method's generator.
    assert "use_memory_to_answer" not in vars(MemoryOSMemo)


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
