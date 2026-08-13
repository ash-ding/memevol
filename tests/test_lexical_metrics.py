"""Tests for the LoCoMo lexical metrics (token-F1 / BLEU-1) and their
per-category aggregation into score.json's `extra_metrics`.

Locks in the two things that are easy to break silently:

1. **"Alongside, never instead."** The promotion signal
   (`benchmark_eval_score`) comes from the LLM judge. Lexical metrics are
   reporting only — nothing here may leak into it.
2. **The weighted/unweighted distinction.** The papers report the UNWEIGHTED
   mean over the four categories; LoCoMo's own mix is ~55% single-hop / ~6%
   open-domain, so the weighted mean is a materially different number. Both are
   emitted under unambiguous names, and a bare `token_f1` key must NOT exist —
   that key is exactly what would invite comparing our weighted number against a
   paper's unweighted one.

Also guards the metric definitions themselves: BLEU-1 here is unigram precision
WITHOUT brevity penalty (deliberate — matches the LoCoMo-derived scripts the
papers build on), and the formulas must stay identical to the offline
implementation the mem0/memoryos README numbers were produced with.

Zero-dependency runner:

    uv run python tests/test_lexical_metrics.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

from benchmarks.locomo.env import CATEGORY
from benchmarks.locomo.workflow import LoCoMoWorkflow
from common.metric import bleu1, token_f1


# ---------------- helpers ----------------

class FakeRecorder:
    """Duck-typed recorder: only the attrs aggregate_extra_metrics reads."""
    def __init__(self, steps, user_id="u"):
        self.steps = steps
        self.user_id = user_id
        self.reward = 0.0


def step(category, predicted, reference, score=1, **extra):
    s = {
        "query": "q",
        "predicted": predicted,
        "reference": reference,
        "score": score,
        "qa_metadata": {"category": category, "evidence": []},
    }
    s.update(extra)
    return s


def _agg(recorders):
    return LoCoMoWorkflow.aggregate_extra_metrics(None, recorders)


# ---------------- metric definitions ----------------

def test_token_f1_basic():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("cat", "dog") == 0.0
    # Articles and punctuation are normalized away (SQuAD-style).
    assert token_f1("The cat, sat.", "a cat sat") == 1.0
    # Partial overlap: pred=2 tokens, gold=3, hit=2 -> P=1, R=2/3, F1=0.8
    assert abs(token_f1("cat sat", "cat sat down") - 0.8) < 1e-12


def test_bleu1_has_no_brevity_penalty():
    """A one-word prediction fully contained in a long gold scores a PERFECT
    unigram precision. This is deliberate — it is what the LoCoMo-derived
    evaluation scripts the papers build on compute. If someone "fixes" this by
    adding a brevity penalty, our numbers silently stop being comparable to the
    published ones and this test fails."""
    assert bleu1("cat", "the cat sat on the mat") == 1.0
    # ...whereas token-F1 does punish the length mismatch.
    assert token_f1("cat", "the cat sat on the mat") < 0.5


def test_empty_string_edge_cases():
    assert token_f1("", "") == 1.0 and bleu1("", "") == 1.0
    assert token_f1("", "x") == 0.0 and bleu1("", "x") == 0.0
    assert token_f1("x", "") == 0.0 and bleu1("x", "") == 0.0
    # An all-article prediction normalizes to empty — must not ZeroDivisionError.
    assert token_f1("the a an", "cat") == 0.0
    assert bleu1("the a an", "cat") == 0.0


def test_non_string_inputs_do_not_raise():
    """Traces carry whatever the memo returned; a non-str prediction must
    degrade, not crash the whole aggregation."""
    assert token_f1(123, "123") == 1.0
    assert bleu1(None, "none") == 1.0


# ---------------- aggregation ----------------

def test_per_category_aggregation():
    recs = [FakeRecorder([
        step(4, "cat", "cat"),                    # single-hop, F1 1.0
        step(4, "dog", "cat", score=0),           # single-hop, F1 0.0
        step(1, "cat sat", "cat sat"),            # multi-hop,  F1 1.0
    ])]
    out = _agg(recs)["locomo_lexical"]
    per_cat = out["per_category"]
    assert per_cat["single-hop"]["token_f1"] == 50.0, per_cat
    assert per_cat["single-hop"]["n"] == 2
    assert per_cat["multi-hop"]["token_f1"] == 100.0
    assert out["n_questions"] == 3


def test_weighted_and_unweighted_means_both_present_and_differ():
    """The whole point of emitting both: on a skewed category mix they are
    different numbers, and only the unweighted one is comparable to a paper."""
    recs = [FakeRecorder(
        # 9 single-hop, all wrong; 1 multi-hop, perfect.
        [step(4, "dog", "cat", score=0) for _ in range(9)]
        + [step(1, "cat", "cat")]
    )]
    out = _agg(recs)["locomo_lexical"]
    assert out["token_f1_mean_unweighted"] == 50.0   # (0 + 100) / 2 categories
    assert out["token_f1_mean_weighted"] == 10.0     # 1 of 10 questions
    for key in ("token_f1", "bleu1"):
        assert f"{key}_mean_unweighted" in out and f"{key}_mean_weighted" in out
        # A bare key would be the ambiguous one — it must not exist.
        assert key not in out, f"ambiguous bare {key!r} key in extra_metrics"


def test_unweighted_mean_ignores_unmapped_categories():
    """An unknown category id buckets to 'other'. It may appear in per_category
    (visibility) but must never enter the papers' 'Avg.'."""
    recs = [FakeRecorder([
        step(4, "cat", "cat"),          # single-hop, 100
        step(99, "dog", "cat", score=0)  # unmapped -> 'other', 0
    ])]
    out = _agg(recs)["locomo_lexical"]
    assert "other" in out["per_category"]
    assert out["token_f1_mean_unweighted"] == 100.0     # 'other' excluded
    assert out["token_f1_mean_weighted"] == 50.0        # ...but still counted here
    assert out["n_questions"] == 2


def test_recomputes_when_trace_predates_per_step_metrics():
    """Old traces have no token_f1/bleu1 on the step. Aggregation must fall back
    to recomputing from predicted/reference, so historical traces stay
    re-aggregatable without re-running the eval."""
    old = FakeRecorder([step(4, "cat", "cat")])              # no per-step metrics
    new = FakeRecorder([step(4, "cat", "cat", token_f1=1.0, bleu1=1.0)])
    assert (_agg([old])["locomo_lexical"]["per_category"]["single-hop"]["token_f1"]
            == _agg([new])["locomo_lexical"]["per_category"]["single-hop"]["token_f1"])


def test_exceptions_and_empty_input_are_survivable():
    """A crashed user arrives as an Exception in the recorder list; an eval with
    no usable steps must return {} rather than raising or emitting a fake 0."""
    assert _agg([]) == {}
    assert _agg([RuntimeError("dead user")]) == {}
    out = _agg([RuntimeError("dead"), FakeRecorder([step(4, "cat", "cat")])])
    assert out["locomo_lexical"]["n_questions"] == 1


def test_category_map_matches_the_ids_the_loader_keeps():
    """cat-5 (adversarial) is excluded at load time and must stay out of the
    map, or it would silently become a fifth term in the papers' 'Avg.'."""
    assert set(CATEGORY) == {1, 2, 3, 4}
    assert set(CATEGORY.values()) == {"multi-hop", "temporal", "open-domain", "single-hop"}


# ---------------- the promotion-signal boundary ----------------

def test_lexical_metrics_stay_out_of_the_promotion_signal():
    """`extra_metrics` is reporting only: the score.json fields forge promotes
    on must be byte-identical whether or not lexical metrics are present."""
    from common.evaluate import _build_score_json

    recs = [FakeRecorder([step(4, "cat", "cat")], user_id="u1")]
    recs[0].reward = 1.0
    score = _build_score_json(recs)
    assert "extra_metrics" not in score, "shared builder must stay benchmark-agnostic"
    assert set(score) == {"benchmark_eval_score", "per_user", "invalid_users"}
    # Merging extras must not perturb the promotion signal.
    before = dict(score["benchmark_eval_score"])
    score["extra_metrics"] = _agg(recs)
    assert score["benchmark_eval_score"] == before


def test_default_workflow_emits_no_extra_metrics():
    """Other benchmarks must be unaffected — the base hook returns {}."""
    from common.workflow import BaseWorkflow
    assert BaseWorkflow.aggregate_extra_metrics(None, [FakeRecorder([step(4, "a", "a")])]) == {}


# ---------------- the per-step write path ----------------

def _log_one(predicted, reference, score=1, category=4):
    """Drive log_qa_step against the REAL LoCoMoRecorder.

    Deliberately not a duck-typed fake: LoCoMoRecorder overrides log_step with
    an EXPLICIT signature (it does not take **kwargs), so a permissive fake
    would accept a kwarg the real recorder rejects with TypeError — which is
    exactly the bug this test exists to catch."""
    import asyncio
    from benchmarks.locomo.env import LoCoMoRecorder

    rec = LoCoMoRecorder()
    asyncio.run(LoCoMoWorkflow.log_qa_step(
        None, recorder=rec, query="q", predicted=predicted, reference=reference,
        score=score, judge_reason="r",
        qa_metadata={"category": category, "evidence": []},
        retrieved_memory={}, relevant_context=[],
    ))
    return rec.steps[0]


def test_log_qa_step_records_per_step_metrics():
    """The write path itself. Without this, a broken log_qa_step is invisible:
    aggregation silently recomputes from predicted/reference and every other
    test still passes."""
    s = _log_one("the cat sat", "the cat sat")
    assert s["token_f1"] == 1.0 and s["bleu1"] == 1.0
    # ...and the pre-existing fields are untouched.
    assert s["predicted"] == "the cat sat" and s["score"] == 1
    assert s["qa_metadata"]["category"] == 4
    assert set(s) >= {"query", "predicted", "reference", "score", "judge_reason",
                      "qa_metadata", "retrieved_memory", "relevant_turns"}


def test_log_qa_step_survives_the_error_paths():
    """common/workflow.py logs a step with predicted="" when the QA agent or
    retrieval fails. Metric computation must not raise there — an API outage
    must stay a recorded score=0 step, not a crashed user."""
    s = _log_one("", "the cat sat", score=0)
    assert s["token_f1"] == 0.0 and s["bleu1"] == 0.0


def test_aggregation_consumes_what_log_qa_step_actually_wrote():
    """End-to-end on the real field names: whatever log_qa_step writes must be
    the key aggregation reads. A rename on either side breaks this."""
    steps = [_log_one("cat", "cat"), _log_one("dog", "cat", score=0)]
    out = _agg([FakeRecorder(steps)])["locomo_lexical"]
    assert out["per_category"]["single-hop"]["token_f1"] == 50.0
    assert out["n_questions"] == 2


# ---------------- integration: the evaluate_memo seam ----------------

def _run_locomo_eval(steps_per_user, patch_hook=None):
    """Drive the real evaluate_memo with a stubbed run_all_users (no LLM calls)
    and return the written score.json."""
    import asyncio, json, shutil, tempfile
    from pathlib import Path
    from benchmarks.locomo.workflow import LoCoMoWorkflow
    from common.evaluate import evaluate_memo
    from common.memo_class import MemoClass

    async def _fake_run_all(self, task_list, *, stage="stage3", stage_spec=None,
                            max_sample_concurrent=6):
        recs = [FakeRecorder(list(steps_per_user), user_id=str(u)) for u in task_list]
        for r in recs:
            r.reward = 1.0
        return recs, len(recs)

    class _M(MemoClass):
        async def retrieve_memory_for_query(self, r): return {}

    out_dir = Path(tempfile.mkdtemp(prefix="test_lexical_"))
    orig_run, orig_save = LoCoMoWorkflow.run_all_users, LoCoMoWorkflow.save_full_traces
    orig_hook = LoCoMoWorkflow.aggregate_extra_metrics
    LoCoMoWorkflow.run_all_users = _fake_run_all
    LoCoMoWorkflow.save_full_traces = lambda self, recs: None
    if patch_hook is not None:
        LoCoMoWorkflow.aggregate_extra_metrics = patch_hook
    try:
        asyncio.run(evaluate_memo(
            memo_class=_M, dataset="locomo", split="test", progressive=False,
            single_stage={"n_conversations": 1, "n_qa": 2},
            out_dir=out_dir, qa_model="m", judge_model="m",
            max_sample_concurrent=1, memory_cache=False))
        return json.loads((out_dir / "score.json").read_text(encoding="utf-8"))
    finally:
        LoCoMoWorkflow.run_all_users = orig_run
        LoCoMoWorkflow.save_full_traces = orig_save
        LoCoMoWorkflow.aggregate_extra_metrics = orig_hook
        shutil.rmtree(out_dir, ignore_errors=True)


def test_evaluate_memo_writes_extra_metrics_into_score_json():
    """The seam: evaluate_memo must actually call the hook and merge it."""
    score = _run_locomo_eval([step(4, "cat", "cat"), step(1, "dog", "cat", score=0)])
    lex = score["extra_metrics"]["locomo_lexical"]
    assert lex["per_category"]["single-hop"]["token_f1"] == 100.0
    assert lex["token_f1_mean_unweighted"] == 50.0
    # ...without disturbing the promotion signal.
    assert score["benchmark_eval_score"]["benchmark_overall_eval_score"] == 1.0


def test_a_crashing_reporting_metric_does_not_fail_the_stage():
    """Reporting must never take down an otherwise-valid eval: if the hook
    raises, score.json is still written, still has the judge-derived score, and
    simply carries no extra_metrics."""
    def _boom(self, recorder_list):
        raise RuntimeError("reporting metric exploded")

    score = _run_locomo_eval([step(4, "cat", "cat")], patch_hook=_boom)
    assert "extra_metrics" not in score
    assert score["benchmark_eval_score"]["benchmark_overall_eval_score"] == 1.0


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
