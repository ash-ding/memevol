"""Tests for trace persistence (common/workflow.py::save_full_traces).

Regression guard for the `_MemoryEncoder` NameError (dump-removal fallout,
found in the 2026-07-08 audit): every trace file came out 0-byte because the
encoder class was deleted while its usage survived.

Zero-dependency runner:

    uv run python tests/test_traces.py
"""
import json
import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

from common.workflow import _MemoryEncoder, _json_safe_keys, BaseWorkflow


# ---------------- helpers ----------------

class FakeRecorder:
    """Duck-typed recorder: only the attrs save_full_traces reads."""
    def __init__(self, user_id, steps, reward=0.5, failure_info=None):
        self.user_id = user_id
        self.steps = steps
        self.reward = reward
        self.failure_info = failure_info


class _Custom:
    def __repr__(self):
        return "<Custom xyz>"


class _ShellWorkflow(BaseWorkflow):
    """Minimal concrete subclass — only save_full_traces (concrete on the
    base) is exercised; the hooks are inert stubs."""
    recorder_class = FakeRecorder  # type: ignore[assignment]

    async def load_user_data(self, user_dir, eval_n_qa):  # pragma: no cover
        return [], []

    async def phase1_log_init(self, recorder, chunk):  # pragma: no cover
        pass

    def build_query_recorder_init(self, init_data, qa):  # pragma: no cover
        return {}

    def build_qa_prompt(self, query, retrieved, qa_metadata, reference=""):  # pragma: no cover
        return [{"content": ""}, {"content": ""}]

    def extract_relevant_context(self, qa, init_data):  # pragma: no cover
        return None

    def build_qa_metadata(self, qa):  # pragma: no cover
        return {}

    async def log_qa_step(self, **kwargs):  # pragma: no cover
        pass


def _make_workflow_shell(out_dir: Path):
    wf = _ShellWorkflow(memo_class=object, model="test-model")
    wf.output_run_dir = out_dir
    return wf


def _saved_payload(out_dir: Path, user_id: str):
    p = out_dir / "traces" / f"{user_id}.json"
    assert p.exists(), f"trace file missing: {p}"
    size = p.stat().st_size
    assert size > 0, f"trace file is 0-byte: {p} (the audited regression)"
    with p.open() as f:
        return json.load(f)


# ---------------- encoder unit tests ----------------

def test_encoder_handles_common_types():
    obj = {
        "a_set": {"b", "a"},
        "mixed_set": {1, "x"},
        "counter": Counter({"k": 2}),
        "dd": defaultdict(list, {"d": [1]}),
        "dt": datetime(2026, 7, 8, 12, 0, 0),
        "date": date(2026, 7, 8),
        "custom": _Custom(),
    }
    text = json.dumps(obj, cls=_MemoryEncoder)
    round_tripped = json.loads(text)
    assert round_tripped["a_set"] == ["a", "b"]
    assert sorted(map(str, round_tripped["mixed_set"])) == ["1", "x"]
    assert round_tripped["counter"] == {"k": 2}
    assert round_tripped["dd"] == {"d": [1]}
    assert round_tripped["dt"].startswith("2026-07-08T12:00")
    assert round_tripped["date"] == "2026-07-08"
    assert round_tripped["custom"] == "<Custom xyz>"


def test_json_safe_keys_coerces_tuple_keys():
    obj = {("2026", "07"): {"nested": {(1, 2): "v"}}, "plain": [ {(3,): 1} ]}
    safe = _json_safe_keys(obj)
    text = json.dumps(safe)  # must not raise
    rt = json.loads(text)
    assert "('2026', '07')" in rt
    assert "(1, 2)" in rt["('2026', '07')"]["nested"]


# ---------------- save_full_traces integration ----------------

def test_traces_roundtrip_plain():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        wf = _make_workflow_shell(out)
        steps = [{"query": "q1", "score": 1, "retrieved_memory": {"k": "v"}}]
        wf.save_full_traces([FakeRecorder("user_a", steps)])
        payload = _saved_payload(out, "user_a")
        assert payload["user_id"] == "user_a"
        assert payload["n_qa"] == 1
        assert payload["steps"][0]["retrieved_memory"] == {"k": "v"}


def test_traces_roundtrip_hostile_retrieved_memory():
    """The exact class of content the encoder exists for: harness-returned
    retrieved_memory with sets, datetimes, tuple keys, custom objects."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        wf = _make_workflow_shell(out)
        steps = [{
            "query": "q1", "score": 0.5,
            "retrieved_memory": {
                "tags": {"z", "a"},
                "by_month": {(2026, 7): ["log_1"]},
                "when": datetime(2026, 7, 8),
                "obj": _Custom(),
                "counts": Counter({"x": 3}),
            },
        }]
        wf.save_full_traces([FakeRecorder("user_b", steps)])
        payload = _saved_payload(out, "user_b")
        rm = payload["steps"][0]["retrieved_memory"]
        assert rm["tags"] == ["a", "z"]
        assert rm["by_month"] == {"(2026, 7)": ["log_1"]}
        assert rm["obj"] == "<Custom xyz>"
        assert rm["counts"] == {"x": 3}


def test_traces_never_leave_zero_byte_file():
    """Serialization happens before the file is opened: even a pathological
    payload must not leave a 0-byte file behind."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        wf = _make_workflow_shell(out)

        class _Bomb:
            def __repr__(self):
                raise RuntimeError("unserializable even via repr")
            # defeat the __float__/__int__ duck-typing probes
            __float__ = property()  # type: ignore[assignment]

        steps = [{"query": "q", "score": 0, "retrieved_memory": {"bomb": _Bomb()}}]
        wf.save_full_traces([FakeRecorder("user_c", steps)])
        p = out / "traces" / "user_c.json"
        # Either the file doesn't exist (serialization failed cleanly) or it
        # is valid, non-empty JSON. A 0-byte file is the one forbidden state.
        if p.exists():
            assert p.stat().st_size > 0
            json.load(p.open())


def test_traces_skip_exceptions_and_sanitize_ids():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        wf = _make_workflow_shell(out)
        exc = RuntimeError("boom")
        exc.user_id = "dead_user"
        wf.save_full_traces([
            exc,
            FakeRecorder("a/b\\c", [{"query": "q", "score": 1, "retrieved_memory": {}}]),
        ])
        traces = sorted(p.name for p in (out / "traces").iterdir())
        assert traces == ["a_b_c.json"], traces


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


