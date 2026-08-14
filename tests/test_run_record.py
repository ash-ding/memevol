"""The run record: what a run cost that tokens cannot say.

    uv run python tests/test_run_record.py

Local models (embedders, rerankers, LLMlingua-2) are not API calls and produce
no usage object, so they can never appear in token counts. The mitigation is
disclosure — these tests pin that the disclosure actually happens.
"""
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import local_models as L    # noqa: E402


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_register_dedupes_by_model_and_role_but_counts_instances():
    L.reset()
    L.register("BAAI/bge-m3", role="embedder", device="cuda:0")
    L.register("BAAI/bge-m3", role="embedder", device="cuda:0")
    L.register("bge-reranker-v2-m3", role="reranker", device="cuda:0")

    got = L.summary()
    assert [m["name"] for m in got] == ["BAAI/bge-m3", "bge-reranker-v2-m3"]
    assert got[0]["instances"] == 2, "per-user reloads are visible, not merged away"
    assert got[1]["role"] == "reranker"


def test_a_device_learned_later_fills_in_unknown():
    L.reset()
    L.register("qwen3", role="embedder", device=None)
    assert L.summary()[0]["device"] == "unknown"
    L.register("qwen3", role="embedder", device="cpu")
    assert L.summary()[0]["device"] == "cpu"


def test_sentence_transformer_construction_self_registers():
    """The registration point is the library constructor, because these models
    are built deep inside vendored src/ trees that must not be edited."""
    import sentence_transformers as st

    L.reset()
    real_init = st.SentenceTransformer.__init__
    L._installed = False       # re-arm; install() is process-idempotent
    try:
        # Stand in for the real (weight-downloading) constructor. `device` is
        # a read-only property on the real class, so it is left alone here —
        # device resolution is covered by test_device_is_read_off_the_model.
        def _noop(self, *args, **kwargs):
            pass

        st.SentenceTransformer.__init__ = _noop
        L.install()
        obj = st.SentenceTransformer.__new__(st.SentenceTransformer)
        st.SentenceTransformer.__init__(obj, "all-MiniLM-L6-v2")
    finally:
        st.SentenceTransformer.__init__ = real_init
        L._installed = True

    got = L.summary()
    assert len(got) == 1 and got[0]["name"] == "all-MiniLM-L6-v2"
    assert got[0]["role"] == "embedder"
    assert got[0]["library"] == "sentence-transformers"


def test_device_is_read_off_the_model():
    class _OnCuda:
        device = "cuda:1"

    class _ViaParameters:
        def parameters(self):
            class _P:
                device = "cpu"
            yield _P()

    class _Opaque:
        pass

    assert L._device_of(_OnCuda()) == "cuda:1"
    assert L._device_of(_ViaParameters()) == "cpu"
    assert L._device_of(_Opaque()) is None


def test_registration_survives_a_baseline_st_shim_taking_the_name_first():
    """REGRESSION (found by a real simplemem/LoCoMo run, which reported NO
    local models while Qwen3-Embedding-0.6B was loaded): lightmem's and
    simplemem's `_st_shim` rebind `sentence_transformers.SentenceTransformer`
    to a memoizing FUNCTION at memo.py import time — before evaluate_memo
    installs this. Patching `__init__` on a function object silently does
    nothing, so the real class has to be resolved explicitly."""
    import sentence_transformers as st

    real_cls = st.SentenceTransformer
    L.reset()
    L._installed = False
    try:
        # Stand in for the baseline shim: the package attribute is now a
        # plain function, exactly as _st_shim leaves it.
        def _memoizing_factory(*args, **kwargs):
            return real_cls(*args, **kwargs)

        _memoizing_factory._real_sentence_transformer = real_cls
        st.SentenceTransformer = _memoizing_factory

        assert L._sentence_transformer_class() is real_cls, "resolved the wrong object"

        real_init = real_cls.__init__
        try:
            real_cls.__init__ = lambda self, *a, **k: None
            L.install()
            obj = real_cls.__new__(real_cls)
            real_cls.__init__(obj, "Qwen/Qwen3-Embedding-0.6B")
        finally:
            real_cls.__init__ = real_init
    finally:
        st.SentenceTransformer = real_cls
        L._installed = True

    assert [m["name"] for m in L.summary()] == ["Qwen/Qwen3-Embedding-0.6B"]


def test_install_patches_the_local_model_entry_points():
    import sentence_transformers as st
    import transformers as tf

    L._installed = False
    try:
        L.install()
    finally:
        L._installed = True
    assert getattr(st.SentenceTransformer.__init__, "__wrapped_by_memevol__", False)
    # LLMlingua-2 (compressor) and the bge cross-encoder (reranker)
    assert getattr(tf.AutoModelForTokenClassification.from_pretrained,
                   "__wrapped_by_memevol__", False)
    assert getattr(tf.AutoModelForSequenceClassification.from_pretrained,
                   "__wrapped_by_memevol__", False)


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------

class _FakeRec:
    def __init__(self, user_id, reward):
        self.user_id = user_id
        self.reward = float(reward)
        self.steps = [{"q": "x", "score": reward}]
        self.failure_info = None


def _run_stage(out_dir: Path):
    from benchmarks.locomo.workflow import LoCoMoWorkflow
    from common.evaluate import evaluate_memo
    from common.memo_class import MemoClass

    class _StubMemo(MemoClass):
        async def retrieve_memory_for_query(self, r): return {}

    async def _fake(self, task_list, *, stage="stage3", stage_spec=None,
                    max_sample_concurrent=6):
        recs = [_FakeRec(u, 1.0) for u in task_list]
        return recs, len(recs)

    orig = LoCoMoWorkflow.run_all_users
    LoCoMoWorkflow.run_all_users = _fake
    try:
        return asyncio.run(evaluate_memo(
            memo_class=_StubMemo, dataset="locomo", split="test",
            progressive=False, single_stage={"n_conversations": 1, "n_qa": 1},
            out_dir=out_dir, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            max_sample_concurrent=1, memory_cache=False))
    finally:
        LoCoMoWorkflow.run_all_users = orig


def test_run_record_names_the_local_models_that_ran():
    L.reset()
    L.register("BAAI/bge-m3", role="embedder", device="cuda:0")
    L.register("bge-reranker-v2-m3", role="reranker", device="cuda:0")

    out_dir = Path(tempfile.mkdtemp(prefix="test_run_record_"))
    try:
        metrics = _run_stage(out_dir)
        record = json.loads((out_dir / "run_record.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    names = [m["name"] for m in record["local_models"]]
    assert names == ["BAAI/bge-m3", "bge-reranker-v2-m3"]
    assert record["local_models"][1]["device"] == "cuda:0"
    # A cost figure must never read as complete when it is not.
    assert "API calls only" in record["coverage_caveat"]
    assert "phase_seconds" in record and "build_cache" in record
    # ...and it reaches every consumer of evaluate_memo, not just the file.
    assert [m["name"] for m in metrics["local_models"]] == names


def test_run_record_is_copied_to_the_run_root():
    """Consumers read the run root, not the stage dir."""
    L.reset()
    out_dir = Path(tempfile.mkdtemp(prefix="test_run_record_root_"))
    try:
        _run_stage(out_dir)
        assert (out_dir / "single" / "run_record.json").exists()
        assert (out_dir / "run_record.json").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_no_local_models_is_reported_as_an_empty_list_not_a_missing_key():
    """An all-API baseline (hipporag2 with OpenAI embeddings) must still say
    so explicitly — absence of the key would be ambiguous with 'not checked'."""
    L.reset()
    out_dir = Path(tempfile.mkdtemp(prefix="test_run_record_empty_"))
    try:
        _run_stage(out_dir)
        record = json.loads((out_dir / "run_record.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    assert record["local_models"] == []


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


if __name__ == "__main__":
    sys.exit(_main())
