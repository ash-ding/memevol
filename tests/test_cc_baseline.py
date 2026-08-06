"""Tests for the cc harness baseline (baselines/harness/cc/memo.py — native
agentic answer via Claude Code). Zero-dependency runner; run in cc's OWN venv
(baselines/harness/cc/venv/), not the repo-root venv/ dev env:

    baselines/harness/cc/venv/bin/python tests/test_cc_baseline.py
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------- cc (native-answer MemoStructure) --------------------

def test_cc_use_memory_to_answer_runs_cc():
    import asyncio
    from baselines.harness.cc.memo import CCMemo
    from baselines.harness.eval_common import make_memo_class
    async def _fake_ask(question, tmp_dir, model, max_turns, system_prompt=None):
        return ("CCANS:" + question, {}, [])
    Cls = make_memo_class(CCMemo, model="sonnet", max_turns=5, _ask_cc=_fake_ask)
    memo = Cls()
    class _Rec:
        user_id = ""
        init = {"sessions": [{"session_id": "s", "date": "d",
                "messages": [{"role": "user", "content": "hi"}]}], "query": "q?"}
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memo.build_memory_from_data(_Rec()))   # writes context to tmp_dir
    ans = loop.run_until_complete(memo.use_memory_to_answer(_Rec(), {}, "FORMATTED PROMPT"))
    assert ans == "CCANS:FORMATTED PROMPT"


def test_cc_memo_retrieve_empty_and_run_cc_answers():
    import asyncio
    from baselines.harness.cc.memo import CCMemo
    from baselines.harness.eval_common import make_memo_class
    async def _fake_ask(question, tmp_dir, model, max_turns, system_prompt=None):
        return ("STUB:" + question, {}, [])
    Cls = make_memo_class(CCMemo, model="sonnet", max_turns=5, _ask_cc=_fake_ask)
    memo = Cls()
    class _Rec:
        user_id = ""
        init = {"sessions": [{"session_id": "s", "date": "d",
                "messages": [{"role": "user", "content": "hi"}]}], "query": "q?"}
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memo.build_memory_from_data(_Rec()))
    out = loop.run_until_complete(memo.retrieve_memory_for_query(_Rec()))
    assert out == {}                                   # cc injects no memory; answers via files
    ans, _u, _t = loop.run_until_complete(memo._run_cc("FORMATTED PROMPT"))
    assert ans == "STUB:FORMATTED PROMPT"


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
