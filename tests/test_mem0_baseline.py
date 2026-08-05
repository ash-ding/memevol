"""Tests for the Mem0 harness baseline (init→message mapping, config wiring).
Zero-dependency runner (no pytest in the venvs) — mem0's OWN venv (it imports
the mem0 package):

    baselines/harness/mem0/venv/bin/python tests/test_mem0_baseline.py

Network/LLM calls are NOT exercised here; those need a live key and are covered
by an actual run.
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_locomo_turns_carry_speaker_and_date():
    # Mem0's extractor sees only role+content, so speaker and timestamp have to
    # be inside the content or "who said it / when" becomes unanswerable.
    from baselines.harness.mem0.memo import _init_to_messages
    conv = {
        "session_1_date_time": "1:00 pm on 1 May, 2023",
        "session_1": [{"speaker": "Ann", "text": "I adopted a dog."},
                      {"speaker": "Bob", "text": "Nice!"}],
    }
    msgs = _init_to_messages({"conversation": conv})
    assert len(msgs) == 2, msgs
    assert "Ann" in msgs[0]["content"] and "1 May, 2023" in msgs[0]["content"], msgs[0]
    assert msgs[0]["role"] == "user"


def test_locomo_sessions_ordered_numerically():
    # Lexical ordering would give session_1, session_10, session_2 and scramble
    # the timeline Mem0's UPDATE/DELETE bookkeeping depends on.
    from baselines.harness.mem0.memo import _init_to_messages
    conv = {}
    for i in (1, 2, 10):
        conv[f"session_{i}_date_time"] = f"day {i}"
        conv[f"session_{i}"] = [{"speaker": "A", "text": f"turn {i}"}]
    msgs = _init_to_messages({"conversation": conv})
    assert [m["content"].split("turn ")[1] for m in msgs] == ["1", "2", "10"], msgs


def test_locomo_image_caption_is_kept():
    from baselines.harness.mem0.memo import _init_to_messages
    conv = {"session_1_date_time": "d", "session_1": [
        {"speaker": "A", "text": "", "blip_caption": "a red bicycle"}]}
    msgs = _init_to_messages({"conversation": conv})
    assert "red bicycle" in msgs[0]["content"], msgs


def test_longmemeval_roles_preserved():
    from baselines.harness.mem0.memo import _init_to_messages
    init = {"sessions": [{"date": "2023-05-01", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]}
    msgs = _init_to_messages(init)
    assert [m["role"] for m in msgs] == ["user", "assistant"], msgs


def test_dynamicmem_uses_shared_passage_text():
    # Identical passage content across baselines is what keeps dynamicmem
    # numbers comparable, so this must go through hipporag2's renderer.
    from baselines.harness.mem0.memo import _init_to_messages
    from baselines.harness.hipporag2.memo import app_log_to_passage
    entry = {"app_log_id": "1", "timestamp": "2023-01-01", "app_name": "Mail",
             "api_name": "send", "request": {"to": "x"}}
    msgs = _init_to_messages({"app_logs": [entry]})
    assert msgs[0]["content"] == app_log_to_passage(entry)


def test_run_config_keys_match_memo_reads():
    # Every knob the memo reads from _cfg must exist in DEFAULT_CONFIG, or a
    # run silently gets None (e.g. top_k=None -> TypeError deep inside search).
    from baselines.harness.mem0.run import DEFAULT_CONFIG
    for key in ("mem0_llm_model", "embedding_model", "base_url",
                "add_batch_size", "infer", "top_k", "threshold"):
        assert key in DEFAULT_CONFIG, key


def test_memo_implements_the_three_hook_contract():
    from common.harness_base import MemoStructure
    from baselines.harness.mem0.memo import Mem0Memo
    assert issubclass(Mem0Memo, MemoStructure)
    for hook in ("build_memory_from_data", "retrieve_memory_for_query"):
        assert callable(getattr(Mem0Memo, hook, None)), hook
    # use_memory_to_answer must NOT be overridden: the shared QA agent answers.
    assert "use_memory_to_answer" not in vars(Mem0Memo)


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
