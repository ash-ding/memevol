"""Tests for the evolvemem evolve baseline (baselines/evolve/evolvemem/).

Zero-dependency runner; run in evolvemem's OWN uv project, not the repo-root dev env:

    uv run --project baselines/evolve/evolvemem python tests/test_evolvemem_baseline.py

No LLM is needed: the vendored EvolutionEngine is faked wherever a call would be
made, so these run offline and in CI.
"""
import asyncio
import json
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------- ingestion dispatch --------------------

def test_init_to_sessions_covers_all_three_datasets():
    from baselines.evolve.evolvemem.memo import init_to_sessions

    lc = init_to_sessions({"conversation": {
        "speaker_a": "Amy", "speaker_b": "Bob",
        "session_1": [{"speaker": "Amy", "dia_id": "D1:1", "text": "hi"}],
        "session_1_date_time": "2023/01/01"}})
    assert len(lc) == 1 and lc[0][1] == "2023/01/01"
    assert lc[0][2][0] == {"speaker": "Amy", "dia_id": "D1:1", "text": "hi"} or \
        lc[0][2][0]["text"] == "hi"

    lme = init_to_sessions({"sessions": [{"session_id": "s1", "date": "2023/05/20",
          "messages": [{"role": "user", "content": "hello"}]}]})
    assert lme[0][0] == "s1" and lme[0][2][0] == {"speaker": "user", "text": "hello"}

    dm = init_to_sessions({"app_logs": [{"timestamp": "T", "app_name": "App",
          "api_name": "x", "request": {}, "response": {}, "metadata": {"domain": "d"}}]})
    assert dm[0][2][0]["text"].startswith("[T] App: App, Action: x")


def test_app_log_passage_text_matches_every_other_baseline():
    """This baseline COPIES `app_log_to_passage` rather than importing it (methods
    never import each other). The copy is only justified while its OUTPUT stays
    identical to hipporag2's, which every other baseline uses — otherwise
    dynamicmem baselines would silently be ingesting different text."""
    from baselines.evolve.evolvemem.memo import app_log_to_passage as ours
    from baselines.harness.hipporag2.memo import app_log_to_passage as theirs
    for entry in (
        {"timestamp": "T", "app_name": "App", "api_name": "x",
         "request": {}, "response": {}, "metadata": {"domain": "d"}},
        {"timestamp": "2023-11-01", "app_name": "LLM Assistant",
         "api_name": "ContinueConversation",
         "request": {"message": "héllo", "n": 1}, "response": {"ok": True},
         "metadata": {"domain": "household"}},
        {},   # every field missing
    ):
        assert ours(entry) == theirs(entry), f"passage text drifted for {entry}"


def test_unrecognized_init_raises():
    from baselines.evolve.evolvemem.memo import init_to_sessions
    try:
        init_to_sessions({"something_else": []})
    except KeyError:
        return
    raise AssertionError("expected KeyError on unrecognized recorder.init keys")


# -------------------- theta (the evolved artifact) --------------------

def test_weak_theta_is_the_papers_bm25_only_start():
    from baselines.evolve.evolvemem.memo import load_theta
    weak = load_theta({"initial_config": "weak"})
    assert weak.fusion_mode == "keyword_only" and weak.semantic_top_k == 0
    strong = load_theta({"initial_config": "strong"})
    assert strong.semantic_top_k > 0


def test_theta_loads_from_file_and_from_summary_and_ignores_unknown_keys():
    from baselines.evolve.evolvemem.memo import load_theta
    with tempfile.TemporaryDirectory() as d:
        bare = Path(d) / "theta.json"
        bare.write_text(json.dumps({"keyword_top_k": 9, "not_a_field": 1}))
        assert load_theta({"theta_path": str(bare)}).keyword_top_k == 9

        summary = Path(d) / "summary.json"
        summary.write_text(json.dumps({"final_config": {"keyword_top_k": 11}}))
        assert load_theta({"theta_path": str(summary)}).keyword_top_k == 11


def test_bad_initial_config_raises():
    from baselines.evolve.evolvemem.memo import load_theta
    try:
        load_theta({"initial_config": "medium"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown initial_config")


# -------------------- the 3-hook contract --------------------

class _FakeResult:
    def __init__(self, prediction):
        self.prediction = prediction


class _FakeEngine:
    """Stands in for the vendored EvolutionEngine — records what it is handed."""

    def __init__(self):
        self.seen_qa = []
        outer = self

        class _Extractor:
            def extract_sessions(self, sessions, cache_dir=None):
                mems = [{"content": t["text"], "memory_type": "episodic",
                         "entities": [], "topics": [], "importance": 0.5,
                         "source": sid}
                        for sid, _date, turns in sessions for t in turns]
                return mems, []

        self.extractor = _Extractor()

    def _generate_answer(self, question, retrieved, category, qa=None):
        return "answer-under-theta"

    def _evaluate_qa(self, index, qa_pairs, ret_config):
        from evolvemem.multi_retriever import RetrievedMemory
        self.seen_qa.extend(qa_pairs)
        # Mirror the real engine: it hands the final retrieved list to
        # _generate_answer. Real RetrievedMemory objects, so the production
        # `format_context` call is exercised rather than stubbed around.
        retrieved = [RetrievedMemory(content="remembered passage", score=1.0, source="keyword")]
        pred = self._generate_answer(qa_pairs[0]["question"], retrieved,
                                     qa_pairs[0].get("category", 0), qa_pairs[0])
        return [_FakeResult(pred)]


class _Rec:
    def __init__(self, init):
        self.init = init
        self.user_id = "u1"


def _memo_with_fake_engine(**config):
    from baselines.evolve.evolvemem.memo import EvolveMemMemo
    memo = EvolveMemMemo(config=dict(initial_config="weak", **config))
    fake = _FakeEngine()
    memo._engine = fake
    return memo, fake


def test_three_hooks_run_and_answer_comes_from_theta_policy():
    memo, _fake = _memo_with_fake_engine(honor_answer_policy=True)
    loop = asyncio.new_event_loop()

    build_rec = _Rec({"conversation": {
        "speaker_a": "A", "speaker_b": "B",
        "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "Amy adopted a beagle"}],
        "session_1_date_time": "d"}})
    loop.run_until_complete(memo.build_memory_from_data(build_rec))
    assert memo._memories, "build_memory_from_data stored no memories"

    q = _Rec({"conversation": build_rec.init["conversation"], "query": "what pet?"})
    retrieved = loop.run_until_complete(memo.retrieve_memory_for_query(q))
    assert "passages" in retrieved and retrieved["passages"]

    answer = loop.run_until_complete(memo.use_memory_to_answer(q, retrieved, "prompt"))
    assert answer == "answer-under-theta", (
        "with honor_answer_policy=true the artifact must answer under the evolved "
        f"answer policy, got {answer!r}")


def test_ablation_defers_to_shared_qa_agent():
    memo, _fake = _memo_with_fake_engine(honor_answer_policy=False)
    loop = asyncio.new_event_loop()
    build_rec = _Rec({"conversation": {
        "speaker_a": "A", "speaker_b": "B",
        "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "Amy adopted a beagle"}],
        "session_1_date_time": "d"}})
    loop.run_until_complete(memo.build_memory_from_data(build_rec))
    q = _Rec({"conversation": build_rec.init["conversation"], "query": "what pet?"})
    retrieved = loop.run_until_complete(memo.retrieve_memory_for_query(q))
    # Retrieval is identical in both modes — that is what makes the delta an
    # isolation of answer policy rather than of the whole pipeline.
    assert retrieved.get("passages")
    answer = loop.run_until_complete(memo.use_memory_to_answer(q, retrieved, "prompt"))
    assert answer is None, "honor_answer_policy=false must defer to the shared QA agent"


def test_gold_answer_never_reaches_the_engine():
    """The load-bearing correctness invariant: upstream's `_evaluate_qa` expects
    QA dicts carrying the reference (it scores in the same pass) and forwards the
    whole dict to a benchmark adapter. The dict this baseline builds must carry
    no gold under any key, whatever the recorder holds."""
    memo, fake = _memo_with_fake_engine(honor_answer_policy=True)
    loop = asyncio.new_event_loop()
    build_rec = _Rec({"conversation": {
        "speaker_a": "A", "speaker_b": "B",
        "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "Amy adopted a beagle"}],
        "session_1_date_time": "d"}})
    loop.run_until_complete(memo.build_memory_from_data(build_rec))

    q = _Rec({"conversation": build_rec.init["conversation"], "query": "what pet?",
              # even if a workflow ever put these in front of the memo:
              "reference": "a beagle", "answer": "a beagle",
              "qa_metadata": {"category": 2}})
    loop.run_until_complete(memo.retrieve_memory_for_query(q))

    assert fake.seen_qa, "engine was never called"
    for qa in fake.seen_qa:
        assert "answer" not in qa, f"gold leaked into the engine's QA dict: {qa}"
        assert "adversarial_answer" not in qa
        assert "reference" not in qa
        blob = json.dumps(qa).lower()
        assert "beagle" not in blob, f"gold text leaked into the engine: {qa}"


# -------------------- config surface --------------------

def test_config_example_passes_strict_validation_unedited():
    from baselines.evolve.evolvemem.config_schema import load_and_validate
    cfg_path = PROJECT_ROOT / "baselines/evolve/evolvemem/config.example.yaml"
    cfg = load_and_validate(cfg_path)
    assert cfg["split"] == "search" and cfg["initial_config"] == "weak"
    # The shipped defaults ARE the paper's / upstream's configuration, so an
    # unedited run is the reproduction rather than a toy: gpt-4o backbone,
    # BAAI/bge-base-en-v1.5 embedder, R_max=7, whole search split. This test is
    # what stops a well-meaning "make the default cheaper" edit from silently
    # turning the documented reproduction into something else.
    assert cfg["max_rounds"] == 7
    assert cfg["evolve_llm_model"] == "gpt-4o"
    assert cfg["embedding_model"] == "BAAI/bge-base-en-v1.5"
    assert cfg["single_stage"] == {"n_conversations": None, "n_qa": None}


def test_config_rejects_missing_and_unknown_keys():
    import yaml
    from baselines.evolve.evolvemem.config_schema import load_and_validate
    cfg_path = PROJECT_ROOT / "baselines/evolve/evolvemem/config.example.yaml"
    base = yaml.safe_load(cfg_path.read_text())

    with tempfile.TemporaryDirectory() as d:
        missing = dict(base); missing.pop("max_rounds")
        p = Path(d) / "missing.yaml"; p.write_text(yaml.safe_dump(missing))
        try:
            load_and_validate(p)
            raise AssertionError("missing key must abort")
        except Exception as exc:
            assert "max_rounds" in str(exc)

        unknown = dict(base); unknown["not_a_key"] = 1
        p = Path(d) / "unknown.yaml"; p.write_text(yaml.safe_dump(unknown))
        try:
            load_and_validate(p)
            raise AssertionError("unknown key must abort")
        except Exception as exc:
            assert "not_a_key" in str(exc)


def test_evolve_refuses_non_search_split():
    """Split discipline is enforced in code, not just documented."""
    from baselines.evolve.evolvemem.evolve import run_evolution
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run_evolution({"dataset": "locomo", "split": "test"}))
    except ValueError as exc:
        assert "search" in str(exc)
        return
    raise AssertionError("evolve.py must refuse a non-search split")


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
