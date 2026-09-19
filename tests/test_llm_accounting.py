"""LLM accounting: phase attribution, call counts, per-stage deltas, and the
differential memory-token counter.

    uv run python tests/test_llm_accounting.py

Everything here runs offline — `common.llm._chat_completion` is stubbed, so no
API call is made and the usage numbers are exact and predictable.
"""
import asyncio
import json
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import llm as llm_mod                      # noqa: E402
from common import tokens as T                         # noqa: E402
from common.memo_class import MemoClass                # noqa: E402
from common.recorder import Basic_Recorder             # noqa: E402
from common.workflow import BaseWorkflow               # noqa: E402


# ---------------------------------------------------------------------------
# Offline stubs
# ---------------------------------------------------------------------------

def _usage(prompt: int, completion: int, reasoning: int = 0) -> Dict[str, Any]:
    u: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if reasoning:
        u["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return u


class _StubChat:
    """Patches common.llm._chat_completion. Judge calls (what="Judge") get a
    parseable verdict; everything else gets a plain answer. Each caller kind
    reports a DISTINCT token count so attribution errors are visible."""

    USAGE = {
        "Judge": _usage(7, 3),
        "Agent": _usage(50, 5, reasoning=2),
    }

    def __init__(self):
        self.calls: List[Tuple[str, str]] = []   # (what, phase-at-call-time)
        self._orig = None

    def __enter__(self):
        self._orig = llm_mod._chat_completion

        async def _fake(*, model, messages, timeout, max_retries, json_mode=False,
                        effort=None, max_tokens=None, temperature=None, what="Agent"):
            self.calls.append((what, T.current_phase()))
            await asyncio.sleep(0)     # a real await → forces task interleaving
            if what == "Judge":
                return json.dumps({"reason": "ok", "score": 1}), self.USAGE["Judge"]
            return "an answer", self.USAGE["Agent"]

        llm_mod._chat_completion = _fake
        return self

    def __exit__(self, *exc):
        llm_mod._chat_completion = self._orig
        return False


# ---------------------------------------------------------------------------
# A minimal benchmark + memo, so the REAL run_all_users drives the assertions
# ---------------------------------------------------------------------------

MEMORY_TEXT = "the memory payload that the harness retrieved for this query"


class _Memo(MemoClass):
    """Spends LLM calls in build and in retrieve, like a real memory system."""

    def __init__(self, config=None):
        super().__init__(config)
        self.built = False

    async def build_memory_from_data(self, recorder) -> None:
        from common.llm import Agent
        await Agent(system_prompt="", model="gpt-4o-mini").ask("extract facts")
        self.built = True

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        from common.llm import Agent
        await Agent(system_prompt="", model="gpt-4o-mini").ask("rewrite the query")
        return {"memories": MEMORY_TEXT}


class _Workflow(BaseWorkflow):
    recorder_class = Basic_Recorder
    judge_score_max = 1

    #: user_dir -> number of QA pairs
    N_QA = 2

    async def load_user_data(self, user_dir, eval_n_qa, sample_seed=None):
        init = [{"text": f"log for {user_dir}"}]
        qa = [{"query": f"q{i}", "reference": "r", "metadata": {}}
              for i in range(self.N_QA)]
        return init, qa

    async def phase1_log_init(self, recorder, chunk) -> None:
        await recorder.log_init(items=chunk)

    def build_query_recorder_init(self, init_data, qa) -> Dict:
        return {"items": init_data, "query": qa["query"]}

    def build_qa_prompt(self, query, retrieved, qa_metadata, reference="") -> List[Dict]:
        memories = retrieved.get("memories", "") if retrieved else ""
        return [
            {"role": "system", "content": "You answer from memory."},
            {"role": "user", "content": f"[Memory]\n{memories}\n[/Memory]\n\n{query}"},
        ]

    def extract_relevant_context(self, qa, init_data):
        return init_data

    def build_qa_metadata(self, qa) -> Dict:
        return dict(qa.get("metadata", {}))

    async def log_qa_step(self, recorder, query, predicted, reference, score,
                          judge_reason, qa_metadata, retrieved_memory,
                          relevant_context) -> None:
        await recorder.log_step(
            query=query, predicted=predicted, reference=reference, score=score,
            judge_reason=judge_reason, qa_metadata=qa_metadata,
            retrieved_memory=retrieved_memory, relevant_context=relevant_context,
        )


def _fresh_tracker() -> T.TokenTracker:
    """Install a clean global tracker (the module-level one is a singleton)."""
    tracker = T.TokenTracker()
    T.GLOBAL_TOKEN_TRACKER = tracker
    return tracker


def _run(workflow_cls=_Workflow, users=("u1",), memo_cls=_Memo, concurrent=2):
    tracker = _fresh_tracker()
    wf = workflow_cls(memo_class=memo_cls, model="gpt-5-mini", judge_model="gpt-5-mini")
    with _StubChat() as stub:
        records, _ = asyncio.run(wf.run_all_users(
            list(users), stage="stage1", stage_spec={"n_qa": workflow_cls.N_QA},
            max_sample_concurrent=concurrent,
        ))
    return tracker, records, stub, wf


# ---------------------------------------------------------------------------
# Phase attribution
# ---------------------------------------------------------------------------

def test_every_phase_is_attributed_and_calls_counted():
    tracker, records, stub, _ = _run(users=("u1",))
    by_phase = tracker.summary()["by_phase"]

    # 1 build + (1 retrieve + 1 answer + 1 judge) per QA
    assert by_phase["build"]["calls"] == 1, by_phase
    assert by_phase["retrieve"]["calls"] == _Workflow.N_QA
    assert by_phase["answer"]["calls"] == _Workflow.N_QA
    assert by_phase["judge"]["calls"] == _Workflow.N_QA
    assert "other" not in by_phase, "a call escaped phase attribution"

    # Counts are real token sums, not call tallies.
    agent, judge = _StubChat.USAGE["Agent"], _StubChat.USAGE["Judge"]
    assert by_phase["build"]["total_tokens"] == agent["total_tokens"]
    assert by_phase["judge"]["total_tokens"] == judge["total_tokens"] * _Workflow.N_QA
    # reasoning tokens broken out separately
    assert by_phase["answer"]["reasoning_tokens"] == 2 * _Workflow.N_QA


def test_model_name_does_not_have_to_differ_between_phases():
    """The pre-phase heuristic ("internal model != QA model") was already
    broken for hipporag2, which uses one model for both. Same model in build
    and answer must still separate."""
    class _SameModelMemo(_Memo):
        async def build_memory_from_data(self, recorder) -> None:
            from common.llm import Agent
            await Agent(system_prompt="", model="gpt-5-mini").ask("extract")

        async def retrieve_memory_for_query(self, recorder) -> Dict:
            return {"memories": MEMORY_TEXT}

    tracker, _, _, _ = _run(memo_cls=_SameModelMemo)
    phases = tracker.summary()["by_model_phase"]["gpt-5-mini"]
    assert phases["build"]["calls"] == 1
    assert phases["answer"]["calls"] == _Workflow.N_QA
    assert phases["build"]["total_tokens"] != 0


def test_phase_is_not_raced_across_concurrent_users():
    """The reason the phase lives in a ContextVar: users run concurrently, so
    a module-level 'current phase' would be clobbered mid-flight. Every stubbed
    call records the phase it saw — none may be wrong."""
    tracker, _, stub, _ = _run(users=tuple(f"u{i}" for i in range(6)), concurrent=6)

    seen = {}
    for what, phase in stub.calls:
        seen.setdefault(phase, set()).add(what)
    # Judge calls only ever in `judge`; no call anywhere landed in `other`.
    assert "other" not in seen, stub.calls
    assert seen["judge"] == {"Judge"}
    assert seen["build"] == {"Agent"} and seen["retrieve"] == {"Agent"}

    by_phase = tracker.summary()["by_phase"]
    assert by_phase["build"]["calls"] == 6
    assert by_phase["judge"]["calls"] == 6 * _Workflow.N_QA


def test_build_phase_survives_a_phase1_update_override():
    """REGRESSION (found by a real simplemem/LoCoMo run): the build phase used
    to be set INSIDE BaseWorkflow._phase1_update, but LoCoMo overrides that
    method (its init_data is a conversation dict, not a list), and an override
    inherits nothing from the base implementation's body. 151k build tokens
    were filed under `other`. The phase now lives at the CALL SITE."""
    class _OverridingWorkflow(_Workflow):
        async def _phase1_update(self, memo, init_data):
            # LoCoMo's shape: duplicates the base logic, no super() call.
            r = self.recorder_class()
            await self.phase1_log_init(r, init_data)
            await memo.build_memory_from_data(r)

    tracker, _, _, _ = _run(workflow_cls=_OverridingWorkflow)
    by_phase = tracker.summary()["by_phase"]
    assert by_phase.get("build", {}).get("calls") == 1, (
        f"build tokens escaped the phase: {by_phase}")
    assert "other" not in by_phase


def test_build_wall_clock_is_recorded_for_an_override_too():
    """The same bug also left `build` missing from phase_seconds entirely,
    which is how it was spotted — wall-clock is the only cost signal that
    covers local compute, so a missing phase there hides real cost."""
    class _OverridingWorkflow(_Workflow):
        async def _phase1_update(self, memo, init_data):
            r = self.recorder_class()
            await self.phase1_log_init(r, init_data)
            await memo.build_memory_from_data(r)

    tracker, _, _, _ = _run(workflow_cls=_OverridingWorkflow)
    assert "build" in tracker.summary()["phase_seconds"]


def test_build_phase_is_not_double_counted():
    """The wrapper moved to the call site; a leftover one inside
    _phase1_update would nest build inside build and bill the time twice.
    Build sleeps a known duration, so double counting is measurable."""
    SLEEP = 0.05

    class _SlowBuildMemo(_Memo):
        async def build_memory_from_data(self, recorder) -> None:
            await asyncio.sleep(SLEEP)

    tracker, _, _, _ = _run(memo_cls=_SlowBuildMemo, users=("u1",))
    build = tracker.summary()["phase_seconds"]["build"]
    assert build >= SLEEP * 0.8, f"build wall-clock not recorded: {build}"
    assert build < SLEEP * 1.8, f"build time counted twice: {build} vs {SLEEP}"


def test_phase_survives_a_vendored_thread_pool():
    """Vendored systems (simplemem) fan LLM calls out over their own
    ThreadPoolExecutor, whose `submit` does NOT carry contextvars — those
    calls would land in `other`."""
    import concurrent.futures
    T.install_thread_context_propagation()
    tracker = _fresh_tracker()

    def _worker():
        tracker.update("m", _usage(10, 1))
        return T.current_phase()

    with T.phase(T.BUILD):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            seen = [f.result() for f in [pool.submit(_worker) for _ in range(4)]]

    assert seen == ["build"] * 4, seen
    assert tracker.summary()["by_phase"]["build"]["calls"] == 4


def test_thread_propagation_copies_rather_than_shares_context():
    """A worker's own phase changes must not leak back to the caller or to
    sibling workers."""
    import concurrent.futures
    T.install_thread_context_propagation()

    def _worker():
        with T.phase(T.RETRIEVE):
            pass
        return T.current_phase()

    with T.phase(T.BUILD):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            [f.result() for f in [pool.submit(_worker) for _ in range(2)]]
        assert T.current_phase() == "build"


def test_judge_phase_is_forced_not_inherited():
    """Judging is evaluation overhead — it must never be billed to a memory
    phase, even when scored from inside a build block."""
    from common.metric import Judge
    tracker = _fresh_tracker()
    with _StubChat():
        with T.phase(T.BUILD):
            asyncio.run(Judge(model="gpt-5-mini").score("q", "p", "r"))
    by_phase = tracker.summary()["by_phase"]
    assert "build" not in by_phase and by_phase["judge"]["calls"] == 1


# ---------------------------------------------------------------------------
# Memory tokens
# ---------------------------------------------------------------------------

def _expected_delta(wf: _Workflow, memories: str) -> int:
    """The differential this counter is defined as: the assembled prompt minus
    the same prompt with nothing retrieved."""
    with_mem = wf._prompt_text(wf.build_qa_prompt("q0", {"memories": memories}, {}))
    without = wf._prompt_text(wf.build_qa_prompt("q0", {}, {}))
    return (T.count_text_tokens(with_mem, "gpt-5-mini")
            - T.count_text_tokens(without, "gpt-5-mini"))


def test_memory_tokens_measure_what_the_memory_added_to_the_prompt():
    _, records, _, wf = _run(users=("u1",))
    rec = records[0]

    expected = _expected_delta(wf, MEMORY_TEXT)
    assert rec.memory_tokens_n == _Workflow.N_QA
    assert rec.memory_tokens_total == expected * _Workflow.N_QA
    # per-step values land in the trace (validates against steps[].retrieved_memory)
    assert [s["memory_tokens"] for s in rec.steps] == [expected] * _Workflow.N_QA


def test_differential_is_not_the_same_as_tokenizing_the_payload():
    """Why definition (B): splicing the payload into a template changes the
    tokenization at the seams, so counting the raw payload (definition (A))
    is close but not what actually landed in the prompt. The differential is
    the number that is true by construction."""
    _, _, _, wf = _run(users=("u1",))
    delta = _expected_delta(wf, MEMORY_TEXT)
    payload_only = T.count_text_tokens(MEMORY_TEXT, "gpt-5-mini")
    assert delta > 0 and payload_only > 0
    assert delta != payload_only, "the seam effect is what (B) captures and (A) misses"
    assert abs(delta - payload_only) <= 4, "…but they must stay in the same ballpark"


def test_memory_tokens_are_zero_when_nothing_is_retrieved():
    class _EmptyMemo(_Memo):
        async def retrieve_memory_for_query(self, recorder) -> Dict:
            return {}

    _, records, _, _ = _run(memo_cls=_EmptyMemo)
    assert records[0].memory_tokens_total == 0
    assert records[0].memory_tokens_n == _Workflow.N_QA   # still prompted


def test_denominator_is_prompted_queries_not_sampled_ones():
    """A retrieve failure never reaches a prompt (not counted at all); an
    answer failure DID have memory stitched into a prompt (counted as
    prompted, excluded from answered)."""
    class _FlakyMemo(_Memo):
        def __init__(self, config=None):
            super().__init__(config)
            self.n = 0

        async def retrieve_memory_for_query(self, recorder) -> Dict:
            self.n += 1
            if self.n == 1:
                raise RuntimeError("retrieval exploded")
            return {"memories": MEMORY_TEXT}

    class _AnswerFailsWorkflow(_Workflow):
        N_QA = 3

    tracker = _fresh_tracker()
    wf = _AnswerFailsWorkflow(memo_class=_FlakyMemo, model="gpt-5-mini",
                              judge_model="gpt-5-mini")
    stub = _StubChat()
    with stub:
        orig = llm_mod._chat_completion
        state = {"n": 0}

        async def _fail_second_answer(*, what="Agent", **kw):
            if what == "Agent" and kw["model"] == "gpt-5-mini":
                state["n"] += 1
                if state["n"] == 1:
                    raise RuntimeError("answer transport died")
            return await orig(what=what, **kw)

        llm_mod._chat_completion = _fail_second_answer
        records, _ = asyncio.run(wf.run_all_users(
            ["u1"], stage="stage1", stage_spec={"n_qa": 3}, max_sample_concurrent=1))

    rec = records[0]
    # 3 sampled: #1 failed at retrieve, #2 failed at answer, #3 clean.
    assert len(rec.steps) == 3
    assert rec.memory_tokens_n == 2, "prompted = retrieval succeeded"
    assert rec.memory_tokens_answered_n == 1, "answered = QA call also succeeded"


def test_memory_token_aggregation_across_users():
    from common.evaluate import _memory_token_metrics
    _, records, _, wf = _run(users=("u1", "u2", "u3"), concurrent=3)
    m = _memory_token_metrics(records)
    per_query = _expected_delta(wf, MEMORY_TEXT)
    n = 3 * _Workflow.N_QA
    assert m["memory_tokens_n_queries"] == n
    assert m["memory_tokens_total"] == per_query * n
    assert abs(m["memory_tokens_per_query"] - per_query) < 1e-9


# ---------------------------------------------------------------------------
# Per-user cost tally + the per-query cost metric
# ---------------------------------------------------------------------------

def test_each_user_is_charged_only_its_own_calls():
    """Users overlap on one event loop; each recorder must carry ITS calls."""
    _, records, _, _ = _run(users=("u1", "u2", "u3"), concurrent=3)
    agent = _StubChat.USAGE["Agent"]
    for rec in records:
        assert rec.cost_build_calls == 1, rec.cost_build_calls
        assert rec.cost_build_prompt_tokens == agent["prompt_tokens"]
        assert rec.cost_retrieve_calls == _Workflow.N_QA
        assert rec.cost_retrieve_completion_tokens == agent["completion_tokens"] * _Workflow.N_QA
        # The QA agent and the judge are NOT the memory system's cost.
        assert not hasattr(rec, "cost_answer_calls")
        assert not hasattr(rec, "cost_judge_calls")


def test_a_tally_counts_calls_made_on_worker_threads():
    """Synchronous baselines run their hooks in a worker thread, and some fan
    out over their own pool — both must stay inside the caller's scope."""
    import concurrent.futures
    T.install_thread_context_propagation()
    tracker = _fresh_tracker()

    async def go():
        with T.tally_calls() as tally:
            with T.phase(T.BUILD):
                def _sync_hook():
                    tracker.update("m", _usage(10, 1))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                        for f in [pool.submit(tracker.update, "m", _usage(10, 1))
                                  for _ in range(3)]:
                            f.result()
                await asyncio.to_thread(_sync_hook)
        return tally

    tally = asyncio.run(go())
    assert tally.phase(T.BUILD)["calls"] == 4, tally.snapshot()
    assert tally.phase(T.BUILD)["prompt_tokens"] == 40


def test_embedding_calls_are_excluded_from_the_cost_tally():
    """Embedding tokens cost ~1% of an LLM token, so summing them into one
    number would misprice a system that embeds a lot. They stay visible in
    token_usage.json."""
    tracker = _fresh_tracker()
    with T.tally_calls() as tally:
        with T.phase(T.BUILD):
            tracker.update("gpt-4o-mini", _usage(100, 10))
            tracker.update("text-embedding-3-small", _usage(9999, 0))
    assert tally.phase(T.BUILD)["prompt_tokens"] == 100
    assert tally.phase(T.BUILD)["calls"] == 1
    by_model = tracker.summary()["by_model"]
    assert by_model["text-embedding-3-small"]["prompt_tokens"] == 9999


def test_nested_tallies_do_not_leak_into_each_other():
    tracker = _fresh_tracker()
    with T.tally_calls() as outer:
        with T.phase(T.BUILD):
            tracker.update("m", _usage(1, 0))
            with T.tally_calls() as inner:
                tracker.update("m", _usage(10, 0))
            tracker.update("m", _usage(100, 0))
    assert inner.phase(T.BUILD)["prompt_tokens"] == 10
    assert outer.phase(T.BUILD)["prompt_tokens"] == 101, "inner calls are not the outer's"


def test_cost_per_query_is_amortized_and_meaned_over_users():
    from common.evaluate import _cost_per_query_metrics
    _, records, _, wf = _run(users=("u1", "u2"), concurrent=2)
    agent = _StubChat.USAGE["Agent"]
    call = agent["prompt_tokens"] + agent["completion_tokens"]
    memory_per_query = _expected_delta(wf, MEMORY_TEXT)
    n = _Workflow.N_QA
    # per user: ONE build call amortized over n queries + one retrieve call
    # per query + what the memory added to each prompt.
    expected = (call + call * n + memory_per_query * n) / n

    m = _cost_per_query_metrics(records)
    assert abs(m["cost_tokens_per_query"] - expected) < 1e-9, m
    assert abs(m["cost_build_per_query"] - call / n) < 1e-9
    assert abs(m["cost_retrieve_per_query"] - call) < 1e-9
    assert abs(m["cost_memory_tokens_per_query"] - memory_per_query) < 1e-9
    assert abs(m["cost_prompt_tokens_per_query"] + m["cost_completion_tokens_per_query"]
               - m["cost_tokens_per_query"]) < 1e-9, "the split must add up"
    assert m["cost_n_users"] == 2 and m["cost_n_queries"] == 2 * n
    assert m["cost_users_excluded"] == 0


def test_cost_per_query_drops_users_that_answered_nothing():
    """A user who crashed (no recorder) or was never prompted contributes no
    per-query cost — the same users the score drops — and the count says so."""
    from common.evaluate import _cost_per_query_metrics

    class _Rec:
        def __init__(self, n, build, memory):
            self.memory_tokens_n = n
            self.cost_build_prompt_tokens = build
            self.cost_build_completion_tokens = 0
            self.cost_retrieve_prompt_tokens = 0
            self.cost_retrieve_completion_tokens = 0
            self.memory_tokens_total = memory

    m = _cost_per_query_metrics([
        _Rec(2, 100, 20),          # (100 + 20) / 2 = 60
        _Rec(0, 900, 0),           # built, never answered → dropped
        RuntimeError("boom"),      # crashed → dropped
    ])
    assert m["cost_tokens_per_query"] == 60.0, m
    assert m["cost_n_users"] == 1 and m["cost_users_excluded"] == 2


def test_cost_metrics_are_absent_rather_than_zero_for_old_metrics():
    """A pre-2026-09 metrics dict has no cost field; forge must not record it
    as a free harness. (Asked for the efficiency axis here — whether a run
    records it at all is `metrics:`, covered in test_progressive_flags.)"""
    from forge.orchestrator import _build_objectives
    want = ["accuracy", "efficiency"]
    old = {"locomo": {"raw_score": 0.5, "score_max": 1, "tokens": 10}}
    new = {"locomo": {"raw_score": 0.5, "score_max": 1, "tokens": 10,
                      "cost_tokens_per_query": 123.5}}
    assert "cost_tokens_per_query_locomo" not in _build_objectives(old, Path("/nope"), want)
    assert _build_objectives(new, Path("/nope"), want)["cost_tokens_per_query_locomo"] == 123.5


# ---------------------------------------------------------------------------
# Per-stage accounting + schema
# ---------------------------------------------------------------------------

def test_diff_summary_is_a_true_per_stage_delta():
    tracker = _fresh_tracker()
    with T.phase(T.BUILD):
        tracker.update("m", _usage(100, 10))
    before = tracker.summary()
    with T.phase(T.ANSWER):
        tracker.update("m", _usage(30, 3))
    delta = T.diff_summary(tracker.summary(), before)

    # Stage 2's file must NOT carry stage 1's tokens.
    assert "build" not in delta["by_phase"], delta
    assert delta["totals"]["total_tokens"] == 33
    assert delta["by_phase"]["answer"]["calls"] == 1
    # ...while the cumulative tracker still holds everything.
    assert T.total_tokens(tracker.summary()) == 143


def test_read_total_tokens_accepts_the_pre_phase_schema():
    """Run directories written before the phase dimension must not silently
    read as 0 tokens in forge's frontier telemetry."""
    legacy = {"gpt-4o-mini": {"total_tokens": 900, "prompt_tokens": 800},
              "gpt-5-mini": {"total_tokens": 100}}
    assert T.read_total_tokens(legacy) == 1000

    tracker = _fresh_tracker()
    with T.phase(T.BUILD):
        tracker.update("m", _usage(60, 40))
    assert T.read_total_tokens(tracker.summary()) == 100


def test_the_responses_api_usage_shape_is_understood():
    """Chat Completions says prompt/completion; the Responses API says
    input/output for the same two numbers. Both carry total_tokens, so a
    Responses-only baseline reported the right total with a 0/0 split — and
    its per-query cost, which sums prompt+completion, came out ZERO. zep did
    exactly that: 3,059,840 build tokens, cost_build_per_query 0.0, first
    place on the efficiency axis with the most expensive method."""
    from common.tokens import TokenTracker

    class _Details:
        reasoning_tokens = 5

    class _ResponsesUsage:
        input_tokens = 100
        output_tokens = 20
        total_tokens = 120
        output_tokens_details = _Details()

    tracker = TokenTracker()
    tracker.update("gpt-4.1-mini", _ResponsesUsage())
    entry = tracker.summary()["by_model"]["gpt-4.1-mini"]
    assert entry["prompt_tokens"] == 100, "input_tokens must count as prompt"
    assert entry["completion_tokens"] == 20, "output_tokens must count as completion"
    assert entry["total_tokens"] == 120
    assert entry["reasoning_tokens"] == 5, "reasoning nests under output_tokens_details"


def test_the_chat_completions_usage_shape_still_wins():
    """The older spelling must not be disturbed by the fallback."""
    from common.tokens import TokenTracker

    class _ChatUsage:
        prompt_tokens = 7
        completion_tokens = 3
        total_tokens = 10

    tracker = TokenTracker()
    tracker.update("gpt-4.1-mini", _ChatUsage())
    entry = tracker.summary()["by_model"]["gpt-4.1-mini"]
    assert (entry["prompt_tokens"], entry["completion_tokens"]) == (7, 3)


def test_a_call_with_no_usage_block_still_counts_as_a_call():
    tracker = _fresh_tracker()
    with T.phase(T.ANSWER):
        tracker.update("m", None)
    entry = tracker.summary()["by_phase"]["answer"]
    assert entry["calls"] == 1 and entry["total_tokens"] == 0


def test_stage_metrics_split_cost_by_phase():
    from common.evaluate import _stage_cost_metrics
    tracker = _fresh_tracker()
    with T.phase(T.BUILD):
        tracker.update("m", _usage(1000, 0))
    with T.phase(T.RETRIEVE):
        tracker.update("m", _usage(100, 0))
    with T.phase(T.ANSWER):
        tracker.update("m", _usage(10, 0))
    with T.phase(T.JUDGE):
        tracker.update("m", _usage(1, 0))

    m = _stage_cost_metrics(tracker.summary())
    assert m["tokens"] == 1111, "the flat scalar stays ALL-phase"
    assert m["tokens_build"] == 1000
    assert m["tokens_memory"] == 1100, "build + retrieve"
    assert m["tokens_judge"] == 1
    assert m["llm_calls"] == 4
    assert not any(k.startswith("build_cache") for k in m), "the memory cache is gone"


# ---------------------------------------------------------------------------

def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            failures += 1
            import traceback
            print(f"  FAIL {name}: {exc!r}")
            traceback.print_exc()
    print("ALL PASS" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


def test_a_cost_attribution_gap_is_reported():
    """The two counters measure the same LLM calls from two directions, so
    build tokens with no per-user build cost is a contradiction. It shipped
    once: zep recorded 3,059,840 build tokens against cost_build_per_query
    0.0, which put the most expensive baseline first on the efficiency axis."""
    import logging
    from common import evaluate as E

    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Handler()
    E.log.addHandler(handler)
    chat = {"by_model_phase": {"gpt-4.1-mini": {"build": {"total_tokens": 3059840}}}}
    try:
        E._warn_on_cost_attribution_gap(
            {"cost_build_per_query": 0.0, "cost_retrieve_per_query": 0.0,
             "cost_n_users": 1}, chat)
        assert any("build LLM tokens were measured" in m for m in records), records

        records.clear()
        E._warn_on_cost_attribution_gap(
            {"cost_build_per_query": 1529920.0, "cost_retrieve_per_query": 0.0,
             "cost_n_users": 1}, chat)
        assert not records, f"attributed cost must not warn: {records}"

        # Embeddings never enter the per-user tally (CallTally excludes them),
        # so a vector-search retrieval — mem0, LightMem, zep — is not a gap.
        # Comparing raw tokens_retrieve made this warn on 41 embedding tokens.
        E._warn_on_cost_attribution_gap(
            {"cost_build_per_query": 1.0, "cost_retrieve_per_query": 0.0,
             "cost_n_users": 1},
            {"by_model_phase": {"text-embedding-3-small":
                                {"retrieve": {"total_tokens": 41}}}})
        assert not records, f"embedding-only retrieval must not warn: {records}"
    finally:
        E.log.removeHandler(handler)


if __name__ == "__main__":
    sys.exit(_main())
