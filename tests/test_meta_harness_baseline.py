"""Tests for the meta-harness evolve baseline (config, agent argv, state,
baseline harnesses, candidate handoff). Zero-dependency runner (no pytest) —
meta-harness's OWN uv project:

    uv run --project baselines/evolve/meta-harness python tests/test_meta_harness_baseline.py
"""
import asyncio, codecs, json, sys, tempfile, traceback
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baselines" / "evolve" / "meta-harness"
for _p in (str(PROJECT_ROOT), str(BASELINE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _recorder(**init):
    return SimpleNamespace(init=init)


# ---- config ----

def _args(config=None, **overrides):
    """A parsed-args stand-in: every flag unset unless explicitly overridden."""
    import run as mh_run

    fields = {k: None for k in mh_run.DEFAULT_CONFIG}
    fields.update(overrides)
    return SimpleNamespace(config=config, **fields)


def test_shipped_configs_pass_strict_validation():
    import run as mh_run

    search = mh_run.build_cfg(_args(str(BASELINE_ROOT / "config.example.yaml")))
    assert search["status"] == "search" and search["progressive"] is True
    assert search["stages"]["stage1"]["threshold"] == 0.05

    # --status test needs a run to finalize; the flag supplies it here.
    test = mh_run.build_cfg(_args(str(BASELINE_ROOT / "config.test.yaml"),
                                  run_name="dummy"))
    assert test["status"] == "test" and test["progressive"] is False
    assert test["single_stage"] == {"n_users": None, "n_checkpoints": None,
                                    "n_task_a": None, "n_task_c": None}


def test_strict_config_rejects_a_missing_key():
    import yaml
    from common.config import ConfigCompletenessError
    import run as mh_run

    raw = yaml.safe_load((BASELINE_ROOT / "config.example.yaml").read_text(encoding="utf-8"))
    raw.pop("sampling_seed")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "broken.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        try:
            mh_run.build_cfg(_args(str(path)))
        except ConfigCompletenessError as exc:
            assert "sampling_seed" in str(exc)
            return
    raise AssertionError("a missing key must abort the run")


def test_test_status_requires_a_run_name():
    import run as mh_run

    try:
        mh_run.build_cfg(_args(status="test"))
    except ValueError as exc:
        assert "run_name" in str(exc)
        return
    raise AssertionError("--status test without a run name must be rejected")


# ---- proposer argv (both agents) ----

def test_claude_argv_uses_stream_json_and_isolates_operator_settings():
    from proposer import _cmd_claude, _stdin_claude

    system_file = Path("/tmp/sys.txt")
    cmd = _cmd_claude(model="opus", system_prompt_file=system_file, effort="high")
    assert cmd[:2] == ["claude", "-p"]
    for flag in ("--verbose", "--strict-mcp-config", "--setting-sources"):
        assert flag in cmd, flag
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[cmd.index("--system-prompt-file") + 1] == str(system_file)
    assert cmd[cmd.index("--effort") + 1] == "high"
    # The task rides stdin as one NDJSON user message; the prior does not.
    line = json.loads(_stdin_claude("SYSTEM", "TASK").decode())
    assert line["message"]["content"] == "TASK"


def test_codex_argv_reads_the_prompt_from_stdin_after_overrides():
    from proposer import _cmd_codex, _stdin_codex

    cmd = _cmd_codex(model="gpt-5-codex", cwd=Path("/ws"), effort="high")
    assert cmd[:2] == ["codex", "exec"] and cmd[-1] == "-"
    assert cmd.index("-c") < cmd.index("-"), "-c overrides must precede the stdin marker"
    assert 'model_reasoning_effort="high"' in cmd
    # Codex has no system channel: prior and task arrive as one prompt.
    payload = _stdin_codex("SYSTEM", "TASK").decode()
    assert payload.startswith("SYSTEM") and payload.rstrip().endswith("TASK")


def test_a_null_model_lets_the_cli_use_its_own_default():
    """Guessing a model id the account is not entitled to is how the first live
    run died ("gpt-5-codex is not supported ... with a ChatGPT account")."""
    from proposer import _cmd_claude, _cmd_codex

    assert "--model" not in _cmd_codex(model=None, cwd=Path("/ws"), effort=None)
    assert "--model" not in _cmd_claude(model=None, system_prompt_file=Path("s"), effort=None)
    assert "--model" in _cmd_codex(model="gpt-5.5", cwd=Path("/ws"), effort=None)


def test_agent_reported_failures_are_surfaced_not_swallowed():
    """Both CLIs announce a refused turn on the event stream and say nothing on
    stderr. Replays the events from the run that first hit this."""
    from proposer import ProposeResult, _event_claude, _event_codex

    refusal = json.dumps({"type": "error", "status": 400, "error": {
        "type": "invalid_request_error",
        "message": "The 'gpt-5-codex' model is not supported when using Codex "
                   "with a ChatGPT account."}})

    cx = ProposeResult(exit_code=1)
    _event_codex({"type": "item.completed", "item": {
        "type": "error", "message": "Model metadata for `gpt-5-codex` not found."}}, cx)
    _event_codex({"type": "error", "message": refusal}, cx)
    _event_codex({"type": "turn.failed", "error": {"message": refusal}}, cx)

    # The nested provider error is unwrapped, the duplicate announcement deduped,
    # and an error item is not miscounted as a tool call.
    assert "not supported when using Codex" in cx.failure
    assert cx.failure.count("not supported when using Codex") == 1
    assert cx.tools == []

    cc = ProposeResult(exit_code=1)
    _event_claude({"type": "result", "is_error": True, "result": "credit balance too low"}, cc)
    assert "credit balance too low" in cc.failure

    # Nothing reported anywhere still says something usable.
    assert "exit code 7" in ProposeResult(exit_code=7).failure


def test_subscription_auth_keeps_the_eval_api_keys_out_of_the_proposer():
    """run.py loads .env, so both keys are in os.environ when the proposer
    launches. Under subscription auth the agent must not be able to bill them —
    proposer tokens are invisible to common.tokens."""
    import os

    from proposer import _child_env

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
    os.environ["OPENAI_API_KEY"] = "sk-oai-test"
    try:
        claude_sub = _child_env("claude_code", "subscription")
        assert "ANTHROPIC_API_KEY" not in claude_sub
        # The OTHER agent's key stays: the evaluator needs it, and it is not an
        # auth path for this agent.
        assert claude_sub["OPENAI_API_KEY"] == "sk-oai-test"

        codex_sub = _child_env("codex", "subscription")
        assert "OPENAI_API_KEY" not in codex_sub
        assert codex_sub["ANTHROPIC_API_KEY"] == "sk-ant-test"

        # api_key mode is the deliberate opt-in to billing the API.
        assert _child_env("claude_code", "api_key")["ANTHROPIC_API_KEY"] == "sk-ant-test"
        assert _child_env("codex", "api_key")["OPENAI_API_KEY"] == "sk-oai-test"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)


def test_codex_api_key_auth_stages_its_own_home_not_the_operators():
    """Codex reads credentials from $CODEX_HOME. api_key auth points it at a
    staged home so the full model catalogue is reachable (a ChatGPT-account
    login refuses gpt-5 / gpt-5-codex / o3) WITHOUT rewriting ~/.codex."""
    import json as _json
    import os

    from proposer import CODEX_HOME_DIR, _child_env

    os.environ["OPENAI_API_KEY"] = "sk-oai-test"
    try:
        env = _child_env("codex", "api_key")
        home = Path(env["CODEX_HOME"])
        assert home == CODEX_HOME_DIR
        # Never the operator's own codex home.
        assert home != Path.home() / ".codex"
        auth = _json.loads((home / "auth.json").read_text(encoding="utf-8"))
        assert auth == {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-oai-test"}

        # subscription auth leaves CODEX_HOME alone entirely.
        assert "CODEX_HOME" not in _child_env("codex", "subscription")
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


def test_api_key_auth_without_a_key_fails_loudly():
    import os

    from proposer import _child_env

    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        _child_env("codex", "api_key")
    except ValueError as exc:
        assert "OPENAI_API_KEY" in str(exc)
        return
    finally:
        if saved:
            os.environ["OPENAI_API_KEY"] = saved
    raise AssertionError("api_key auth with no key must not fall through silently")


def test_preflight_reports_a_bad_model_before_anything_is_evaluated():
    """Model availability is account-scoped and not otherwise discoverable (a
    ChatGPT-account codex login rejects gpt-5, gpt-5-codex and o3). Without this
    check the run finds out only after phase 0 has spent evaluation tokens."""
    import proposer as mod
    from proposer import ProposeResult

    calls = {}

    async def fake_propose(**kw):
        calls.update(kw)
        return ProposeResult(exit_code=1, errors=[
            "The 'gpt-5' model is not supported when using Codex with a ChatGPT account."])

    real, mod.propose = mod.propose, fake_propose
    try:
        error = asyncio.run(mod.preflight(
            agent="codex", model="gpt-5", cwd=Path("."), log_dir=Path("."), effort="high"))
        assert error and "not supported" in error
        # The real argv is exercised, so a bad effort or model is caught too.
        assert calls["model"] == "gpt-5" and calls["effort"] == "high"
        assert calls["name"] == "preflight"

        async def ok_propose(**kw):
            return ProposeResult(exit_code=0)

        mod.propose = ok_propose
        assert asyncio.run(mod.preflight(
            agent="codex", model="gpt-5.5", cwd=Path("."), log_dir=Path("."))) is None
    finally:
        mod.propose = real


def test_event_handlers_collect_text_tools_and_usage():
    from proposer import ProposeResult, _event_claude, _event_codex

    cc = ProposeResult(exit_code=0)
    _event_claude({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "name": "Write", "input": {"file_path": "h.py"}},
    ]}}, cc)
    _event_claude({"type": "result", "usage": {"input_tokens": 5}, "total_cost_usd": 1.5}, cc)
    assert cc.text == "hello" and len(cc.tools) == 1 and cc.cost_usd == 1.5

    cx = ProposeResult(exit_code=0)
    _event_codex({"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}, cx)
    _event_codex({"type": "item.completed", "item": {"type": "command_call", "command": "ls"}}, cx)
    _event_codex({"type": "turn.completed", "usage": {"input_tokens": 7}}, cx)
    assert cx.text == "hi" and len(cx.tools) == 1 and cx.usage["input_tokens"] == 7


# ---- state: summary + Pareto frontier + finalization lock ----

def _paths(tmp):
    from state import RunPaths

    paths = RunPaths(root=Path(tmp), run_name="t")
    paths.mkdirs()
    return paths


def test_frontier_keeps_the_pareto_front_and_drops_eliminated_rows():
    import state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        rows = [
            ("cheap", 0.40, 100.0, False),   # on the front (cheapest)
            ("best", 0.60, 900.0, False),    # on the front (highest score)
            ("dominated", 0.30, 950.0, False),
            ("crashed", 0.99, 1.0, True),    # eliminated: never on the front
        ]
        for i, (name, score, cost, elim) in enumerate(rows):
            state.append_row(paths, state.summary_row(
                iteration=i, system=name,
                metrics={"raw_score": score, "score_max": 1,
                         "memory_tokens_per_query": cost, "eliminated": elim},
            ))
        frontier = state.rebuild_frontier(paths)

        assert frontier["best"]["system"] == "best"
        front = [e["system"] for e in frontier["_pareto"]]
        assert front == ["best", "cheap"], front
        assert "crashed" not in {e["system"] for e in frontier["all"]}
        assert state.best_score(paths) == 0.60


def test_harness_dirs_are_per_run_so_runs_cannot_collide():
    """A shared candidate directory let one run see -- and --fresh delete --
    another run's harnesses, and made cross-run name collisions possible."""
    import loop
    from state import RunPaths

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b = RunPaths(root=root, run_name="a"), RunPaths(root=root, run_name="b")
        assert a.harnesses != b.harnesses
        assert a.harnesses.is_relative_to(a.logs)
        # The tracked baselines live outside any run and seed all of them.
        assert a.seed_harnesses == b.seed_harnesses == root / "harnesses"

        a.mkdirs(); b.mkdirs()
        a.seed_harnesses.mkdir(parents=True, exist_ok=True)
        for name in ("no_memory", "full_context"):
            (a.seed_harnesses / f"{name}.py").write_text("x = 1", encoding="utf-8")

        cfg = {"baselines": ["no_memory", "full_context"]}
        loop._seed_baselines(a, cfg)
        assert {f.name for f in a.harnesses.glob("*.py")} == {"no_memory.py", "full_context.py"}

        # A candidate in run a is invisible to run b, and --fresh on b spares it.
        (a.harnesses / "candidate.py").write_text("x = 2", encoding="utf-8")
        loop.fresh_start(b)
        assert (a.harnesses / "candidate.py").exists()
        assert not b.logs.exists()


def test_seed_baselines_names_a_missing_baseline():
    import loop
    from state import RunPaths

    with tempfile.TemporaryDirectory() as tmp:
        paths = RunPaths(root=Path(tmp), run_name="r"); paths.mkdirs()
        paths.seed_harnesses.mkdir(parents=True, exist_ok=True)
        try:
            loop._seed_baselines(paths, {"baselines": ["nope"]})
        except FileNotFoundError as exc:
            assert "nope" in str(exc)
            return
    raise AssertionError("a missing baseline must abort the run")


def test_proposer_usage_is_recorded_because_common_tokens_cannot_see_it():
    """The proposer is a CLI subprocess, so its tokens never reach
    common.tokens. Both CLIs report usage on their event stream; this is where
    it is kept."""
    import state

    class _Session:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        state.record_proposer_usage(paths, 1, "codex", None, _Session(
            usage={"input_tokens": 795508, "cached_input_tokens": 731520,
                   "output_tokens": 13202, "reasoning_output_tokens": 3325},
            exit_code=0, duration_s=340.2, tools=["a"] * 35, cost_usd=0.0))
        state.record_proposer_usage(paths, 2, "claude_code", "opus", _Session(
            usage={"input_tokens": 1000, "output_tokens": 200,
                   "cache_read_input_tokens": 900},
            exit_code=0, duration_s=60.0, tools=[], cost_usd=1.25))

        rows = state.read_proposer_usage(paths)
        assert [r["iteration"] for r in rows] == [1, 2]
        assert rows[0]["model"] == "(cli default)"   # null model is recorded honestly
        assert rows[0]["tool_calls"] == 35

        total = state.proposer_usage_total(paths)
        assert total["sessions"] == 2
        assert total["input_tokens"] == 796508 and total["output_tokens"] == 13402
        assert total["cost_usd"] == 1.25
        # Cached reads are carried for context, never added on top of input.
        assert total["cached_input_tokens"] == 731520
        assert total["cache_read_input_tokens"] == 900


def test_frontier_breaks_score_ties_by_the_cheaper_system():
    import state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        for i, (name, cost) in enumerate([("pricey", 1562.0), ("cheap", 1245.2)]):
            state.append_row(paths, state.summary_row(
                iteration=i, system=name,
                metrics={"raw_score": 0.875, "score_max": 1,
                         "memory_tokens_per_query": cost, "eliminated": False}))
        assert state.rebuild_frontier(paths)["best"]["system"] == "cheap"


def test_summary_row_normalizes_score_against_score_max():
    import state

    row = state.summary_row(
        iteration=1, system="s",
        metrics={"raw_score": 3.0, "score_max": 5, "memory_tokens_per_query": 12.4},
        candidate={"hypothesis": "h", "axis": "exploration"}, best_score=0.5,
    )
    assert row["score"] == 0.6 and row["delta"] == 0.1
    assert row["context_cost"] == 12.4 and row["hypothesis"] == "h"


def test_last_iteration_supports_resume():
    import state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        assert state.last_iteration(paths) == 0
        for i in (1, 2, 2):
            state.append_row(paths, {"iteration": i, "system": f"s{i}"})
        assert state.last_iteration(paths) == 2


def test_finalization_lock_round_trips():
    import state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        assert not state.is_finalized(paths)
        state.mark_finalizing(paths, ["a", "b"])
        assert not state.is_finalized(paths)      # in progress is not complete
        state.mark_finalized(paths, {"a": 0.5})
        assert state.is_finalized(paths)


def test_finalize_evaluates_the_pareto_front_plus_baselines_exactly_once():
    """Control-flow only — the evaluator is stubbed, so the test split is never
    touched by the test suite."""
    import loop, state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        for name, score, cost in [("no_memory", 0.1, 0.0), ("best", 0.6, 900.0),
                                  ("dominated", 0.2, 950.0)]:
            state.append_row(paths, state.summary_row(
                iteration=1, system=name,
                metrics={"raw_score": score, "score_max": 1,
                         "memory_tokens_per_query": cost, "eliminated": False}))
        state.rebuild_frontier(paths)

        evaluated = []

        async def fake_evaluate_all(*, names, split, **kw):
            evaluated.append((sorted(names), split))
            return {n: {"raw_score": 0.4, "score_max": 1} for n in names}

        cfg = {"dataset": "locomo", "baselines": ["no_memory", "full_context"]}
        real, loop._evaluate_all = loop._evaluate_all, fake_evaluate_all
        try:
            asyncio.run(loop.finalize(paths, cfg))
            # Pareto front (best, no_memory) union the baselines; not `dominated`.
            assert evaluated == [(["best", "full_context", "no_memory"], "test")]
            assert state.is_finalized(paths)

            asyncio.run(loop.finalize(paths, cfg))   # second call is a no-op
            assert len(evaluated) == 1
        finally:
            loop._evaluate_all = real


def test_evolve_refuses_to_reopen_a_finalized_run():
    import loop, state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        state.mark_finalizing(paths, ["a"])
        state.mark_finalized(paths, {"a": 0.5})
        try:
            asyncio.run(loop.evolve(paths, {"skip_baselines": True, "iterations": 1,
                                            "dataset": "locomo", "agent": "codex",
                                            "agent_model": "m"}))
        except SystemExit as exc:
            assert "finalized" in str(exc)
            return
    raise AssertionError("evolution must not continue under a finalized run name")


# ---- candidate handoff ----

def test_pending_eval_drops_unknown_files_and_duplicate_names():
    import loop

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        (paths.harnesses / "good.py").write_text("x = 1", encoding="utf-8")
        paths.pending.write_text(json.dumps({"candidates": [
            {"name": "good"},        # kept
            {"name": "ghost"},       # no file on disk
            {"name": "already"},     # name already evaluated
            {"name": ""},            # unusable
        ]}), encoding="utf-8")

        kept, error = loop._read_pending(paths, known={"already"})
        assert [c["name"] for c in kept] == ["good"] and error is None


def test_pending_eval_accepts_a_utf8_bom():
    """PowerShell's Set-Content / Out-File write UTF-8 WITH a BOM by default.
    A strict utf-8 read rejects it, which threw away 16 written candidates
    across 5 iterations of the first real LoCoMo run."""
    import loop

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        (paths.harnesses / "good.py").write_text("x = 1", encoding="utf-8")
        paths.pending.write_bytes(
            codecs.BOM_UTF8 + json.dumps({"candidates": [{"name": "good"}]}).encode())
        assert paths.pending.read_bytes().startswith(codecs.BOM_UTF8)

        kept, error = loop._read_pending(paths, known=set())
        assert [c["name"] for c in kept] == ["good"] and error is None


def test_an_unusable_handoff_is_reported_not_silently_skipped():
    """A handoff the loop cannot read records nothing, so the proposer cannot
    learn from it and will repeat it — it must count as a failure."""
    import loop

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        paths.pending.write_text("{not json", encoding="utf-8")
        kept, error = loop._read_pending(paths, known=set())
        assert kept == [] and error and "could not be read" in error

        paths.pending.unlink()
        kept, error = loop._read_pending(paths, known=set())
        assert kept == [] and error and "no pending_eval.json" in error

        paths.pending.write_text(json.dumps({"candidates": []}), encoding="utf-8")
        kept, error = loop._read_pending(paths, known=set())
        assert kept == [] and error and "no usable candidate" in error


def test_task_prompt_names_the_run_paths_the_proposer_must_use():
    import loop

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        prompt = loop._task_prompt(paths, {"dataset": "locomo", "n_candidates": 3}, 4)
        assert "iteration 4" in prompt and "locomo" in prompt
        for path in (paths.summary, paths.frontier, paths.pending, paths.evals):
            assert str(path) in prompt


# ---- the proposer prior ----

def test_proposer_prior_is_a_plain_system_prompt_both_agents_can_read():
    import loop

    text = loop.PROPOSER_SYSTEM.read_text(encoding="utf-8")
    assert loop.PROPOSER_SYSTEM.name == "proposer_system.md"
    # Not a Claude Code skill: codex receives the identical bytes, so skill
    # frontmatter would just be junk shipped to the model.
    assert not text.startswith("---"), "skill frontmatter must not come back"


def test_proposer_prior_forbids_the_runtime_cheat_paths():
    """Nothing sandboxes this proposer, so the prompt is the only thing standing
    between a candidate and the gold answers. Guard the rule's presence."""
    import loop

    text = loop.PROPOSER_SYSTEM.read_text(encoding="utf-8")
    for gold in ("task_packs.json", "locomo10.json", "longmemeval_"):
        assert gold in text, f"prior must name {gold} as off-limits at eval time"
    assert "benchmark or dataset name" in text


def test_proposer_prior_routes_llm_calls_through_common_llm():
    import loop

    text = loop.PROPOSER_SYSTEM.read_text(encoding="utf-8")
    assert "common.llm.Agent" in text and "common.llm.Embedding" in text
    # The load-bearing half: a raw client bypasses token accounting silently.
    assert "import openai" in text and "max_retries" in text


def test_proposer_prior_fences_off_the_search_loop():
    """Appendix D lists "what files it can and cannot modify" as core skill
    content, and the proposer runs unsandboxed with write access to all of it."""
    import loop

    text = loop.PROPOSER_SYSTEM.read_text(encoding="utf-8")
    assert "can and cannot modify" in text
    for owned in ("loop.py", "evaluator.py", "state.py", "run.py", "history.py"):
        assert owned in text, f"prior must fence off {owned}"


def test_effort_defaults_to_the_papers_setting_per_agent():
    """Paper 4.1 runs the proposer at max reasoning; codex tops out at high."""
    from proposer import DEFAULT_EFFORT

    assert DEFAULT_EFFORT["claude_code"] == "max"
    assert DEFAULT_EFFORT["codex"] == "high"


def test_proposer_prior_does_not_name_this_repo_harness_baselines():
    """Pointing the proposer at the systems it is being compared against would
    contaminate the comparison. Upstream names methods outside its own set."""
    import loop

    text = loop.PROPOSER_SYSTEM.read_text(encoding="utf-8")
    for baseline in ("Mem0", "A-Mem", "HippoRAG", "Zep", "LightMem", "SimpleMem"):
        assert baseline not in text, f"prior must not name the {baseline} baseline"


# ---- baseline harnesses ----

def test_baseline_harnesses_load_as_memoclass_subclasses():
    from common.memo_class import MemoClass
    from launch import load_harness_class

    for name in ("no_memory", "full_context"):
        cls = load_harness_class(str(BASELINE_ROOT / "harnesses" / f"{name}.py"))
        assert issubclass(cls, MemoClass), name


def test_no_memory_returns_nothing():
    from launch import load_harness_class

    memo = load_harness_class(str(BASELINE_ROOT / "harnesses" / "no_memory.py"))()
    asyncio.run(memo.build_memory_from_data(_recorder(conversation={})))
    assert asyncio.run(memo.retrieve_memory_for_query(_recorder(query="q"))) == {}


def test_full_context_accumulates_across_build_calls_on_every_shape():
    from launch import load_harness_class

    cls = load_harness_class(str(BASELINE_ROOT / "harnesses" / "full_context.py"))

    # DynamicMem: per-checkpoint deltas must accumulate, with app_log_ids kept.
    memo = cls()
    asyncio.run(memo.build_memory_from_data(_recorder(app_logs=[
        {"app_log_id": "L1", "timestamp": "t1", "app_name": "a", "api_name": "b",
         "request": "r1", "response": "s1"}])))
    asyncio.run(memo.build_memory_from_data(_recorder(app_logs=[
        {"app_log_id": "L2", "timestamp": "t2", "app_name": "a", "api_name": "b",
         "request": "r2", "response": "s2"}])))
    blocks = asyncio.run(memo.retrieve_memory_for_query(
        _recorder(app_logs=[], query="q")))["inline_memory_blocks"]
    assert len(blocks) == 2 and "L1" in blocks[0] and "L2" in blocks[1]

    # LoCoMo: sessions in numeric order, not lexicographic (session_10 last).
    memo = cls()
    conv = {"speaker_a": "A", "speaker_b": "B", "session_1_date_time": "d1",
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "first"}],
            "session_10": [{"speaker": "B", "dia_id": "D10:1", "text": "last"}],
            "session_2": [{"speaker": "A", "dia_id": "D2:1", "text": "middle"}]}
    asyncio.run(memo.build_memory_from_data(_recorder(conversation=conv)))
    blocks = asyncio.run(memo.retrieve_memory_for_query(
        _recorder(conversation=conv, query="q")))["inline_memory_blocks"]
    assert [b.split()[0] for b in blocks] == ["[D1:1]", "[D2:1]", "[D10:1]"]

    # LongMemEval.
    memo = cls()
    asyncio.run(memo.build_memory_from_data(_recorder(sessions=[
        {"session_id": "session_001", "date": "2023/05/20",
         "messages": [{"role": "user", "content": "hello"}]}])))
    blocks = asyncio.run(memo.retrieve_memory_for_query(
        _recorder(sessions=[], query="q", question_date="2023/06/01")))["inline_memory_blocks"]
    assert "session_001" in blocks[0] and "hello" in blocks[0]


def test_full_context_respects_its_char_budget_keeping_the_newest():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fc", BASELINE_ROOT / "harnesses" / "full_context.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    memo = module.FullContextHarness()
    filler = "x" * 5000
    asyncio.run(memo.build_memory_from_data(_recorder(app_logs=[
        {"app_log_id": f"L{i}", "timestamp": "t", "app_name": "a", "api_name": "b",
         "request": filler, "response": ""} for i in range(20)])))
    blocks = asyncio.run(memo.retrieve_memory_for_query(
        _recorder(app_logs=[], query="q")))["inline_memory_blocks"]
    assert 0 < len(blocks) < 20
    assert sum(len(b) for b in blocks) <= module.MAX_CHARS
    assert "L19" in blocks[-1], "the budget must keep the NEWEST units"


# ---- evaluator ----

def test_read_metrics_falls_back_to_score_json_then_to_an_error():
    import evaluator

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        assert evaluator.read_metrics(run_dir)["eliminated"] is True

        (run_dir / "score.json").write_text(json.dumps({"benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.25}}), encoding="utf-8")
        assert evaluator.read_metrics(run_dir)["raw_score"] == 0.25

        (run_dir / "metrics.json").write_text(json.dumps({"raw_score": 0.5}), encoding="utf-8")
        assert evaluator.read_metrics(run_dir)["raw_score"] == 0.5

    assert evaluator.normalized_score({"raw_score": 2.0, "score_max": 4}) == 0.5


def test_sanity_errors_applies_forges_rule():
    import evaluator

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        # No score.json at all — the subprocess died before writing anything.
        assert "no score.json" in evaluator.sanity_errors(run_dir)

        def write(payload):
            (run_dir / "score.json").write_text(json.dumps(payload), encoding="utf-8")

        write({"per_user": {"u1": {"reward": 1.0, "failure_info": None}},
               "invalid_users": []})
        assert evaluator.sanity_errors(run_dir) is None      # clean run

        write({"per_user": {}, "invalid_users": [{"user_id": "u1", "error": "boom"}]})
        assert "boom" in evaluator.sanity_errors(run_dir)

        # A user that only partly completed fails too, exactly as in forge.
        write({"per_user": {"u1": {"reward": 0.5, "failure_info": "phase2 timeout"}},
               "invalid_users": []})
        assert "phase2 timeout" in evaluator.sanity_errors(run_dir)


def test_smoke_flag_reaches_the_subprocess_argv():
    import evaluator

    seen = {}

    async def fake_exec(*cmd, **kw):
        seen["cmd"] = cmd
        raise FileNotFoundError("not actually launching")

    real, asyncio.create_subprocess_exec = asyncio.create_subprocess_exec, fake_exec
    try:
        cfg = {"execution_model": "m", "judge_model": "j", "max_sample_concurrent": 1,
               "sampling_seed": 42, "progressive": True, "random_sample": False,
               "memory_cache": True, "max_logs": None, "stages": None,
               "single_stage": None}
        with tempfile.TemporaryDirectory() as tmp:
            for smoke, expected in ((True, "--smoke"), (False, "--no-smoke")):
                try:
                    asyncio.run(evaluator.evaluate_candidate(
                        name="c", harness_file=Path(tmp) / "c.py", out_dir=Path(tmp) / "o",
                        dataset="locomo", split="search", cfg=cfg, smoke=smoke))
                except FileNotFoundError:
                    pass
                assert expected in seen["cmd"], (smoke, seen["cmd"])
    finally:
        asyncio.create_subprocess_exec = real


def test_rejected_candidates_are_recorded_so_the_proposer_can_read_them():
    """A candidate that fails a gate still gets a row — eliminated, so the
    frontier skips it, but visible in the proposer's only feedback channel."""
    import loop, state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        loop._reject(paths, 3, {"name": "broken", "hypothesis": "h"},
                     "sanity", "user u1: KeyError('query')")

        row = state.read_rows(paths)[-1]
        assert row["system"] == "broken" and row["eliminated"] is True
        assert row["score"] == 0.0 and "sanity:" in row["error"]
        assert row["hypothesis"] == "h"
        # Eliminated rows never reach the frontier.
        assert state.rebuild_frontier(paths)["all"] == []


def test_import_gate_rejects_and_records_without_running_anything():
    import loop, state

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        (paths.harnesses / "good.py").write_text(
            (BASELINE_ROOT / "harnesses" / "no_memory.py").read_text(encoding="utf-8"),
            encoding="utf-8")
        (paths.harnesses / "bad.py").write_text("import nope_xyz\n", encoding="utf-8")

        passed = loop._import_gate([{"name": "good"}, {"name": "bad"}], paths, 1)
        assert [c["name"] for c in passed] == ["good"]
        rows = state.read_rows(paths)
        assert len(rows) == 1 and rows[0]["system"] == "bad"
        assert rows[0]["error"].startswith("import:")


def test_import_check_accepts_a_harness_and_reports_a_broken_one():
    import evaluator

    assert evaluator.import_check(BASELINE_ROOT / "harnesses" / "no_memory.py") is None
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken.py"
        broken.write_text("import nonexistent_module_xyz\n", encoding="utf-8")
        assert evaluator.import_check(broken) is not None
        empty = Path(tmp) / "empty.py"
        empty.write_text("x = 1\n", encoding="utf-8")
        assert "MemoClass" in (evaluator.import_check(empty) or "")


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
