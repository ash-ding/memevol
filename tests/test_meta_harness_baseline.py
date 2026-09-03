"""Tests for the meta-harness evolve baseline (config, agent argv, state,
baseline harnesses, candidate handoff). Zero-dependency runner (no pytest) —
meta-harness's OWN uv project:

    uv run --project baselines/evolve/meta-harness python tests/test_meta_harness_baseline.py
"""
import asyncio, json, sys, tempfile, traceback
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


def test_subscription_auth_drops_the_api_key_only_for_claude():
    import os

    from proposer import _child_env

    os.environ["ANTHROPIC_API_KEY"] = "sk-test"
    try:
        assert "ANTHROPIC_API_KEY" not in _child_env("claude_code", "subscription")
        assert _child_env("claude_code", "api_key")["ANTHROPIC_API_KEY"] == "sk-test"
        assert _child_env("codex", "subscription")["ANTHROPIC_API_KEY"] == "sk-test"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)


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

        kept = loop._read_pending(paths, known={"already"})
        assert [c["name"] for c in kept] == ["good"]


def test_pending_eval_survives_malformed_json():
    import loop

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        paths.pending.write_text("{not json", encoding="utf-8")
        assert loop._read_pending(paths, known=set()) == []


def test_task_prompt_names_the_run_paths_the_proposer_must_use():
    import loop

    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        prompt = loop._task_prompt(paths, {"dataset": "locomo", "n_candidates": 3}, 4)
        assert "iteration 4" in prompt and "locomo" in prompt
        for path in (paths.summary, paths.frontier, paths.pending, paths.evals):
            assert str(path) in prompt


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
