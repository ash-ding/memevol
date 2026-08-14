"""EvolveMem's evolved artifact as a 3-hook `MemoClass`.

EvolveMem evolves a CONFIGURATION, not code: the search loop's output is a
`RetrievalConfig` theta (a JSON object), not a Python class. So this file IS the
artifact loader that `baselines/README.md` § "Adding an evolve baseline" point 2
requires — it loads a theta and runs the paper's own pipeline under it.

WHY IT RUNS THE UPSTREAM PATH RATHER THAN REIMPLEMENTING IT
-----------------------------------------------------------
Choosing a retrieval strategy is method logic: `EvolutionEngine._evaluate_qa`
picks intent-planning > query-decomposition > plain multiview, then optionally
applies coverage reflection, reflection rounds, and an answer-verification pass —
each gated by a theta field the search loop is free to flip. Re-implementing that
ladder here would mean the artifact we SCORE drifts from the artifact the loop
OPTIMISED. So the hooks drive the vendored engine directly (`src/evolvemem/`,
byte-identical @ db80b6a — see README.md).

`_evaluate_qa` answers a question but returns only counts/sources, not the
retrieved text this repo's traces need. Rather than retrieve twice (double the
LLM cost, and a second retrieval could differ), `_run_pipeline` temporarily wraps
the engine's `_generate_answer` to record the final retrieved list it is handed.
The wrapper is integration glue living here; no vendored file is touched.

GOLD NEVER REACHES INFERENCE. `_evaluate_qa` takes QA dicts that upstream
populates with the reference answer (it scores in the same pass). The dict built
here carries ONLY question / category / meta — never `answer` or
`adversarial_answer` — so no benchmark adapter downstream can see the gold even
by accident. The f1 that `_evaluate_qa` computes against the empty reference is
discarded; scoring in this repo is `common.evaluate`'s job.

THE THIRD HOOK. Unlike every harness baseline, this one overrides
`use_memory_to_answer`, because theta's action space includes answer policy
(`answer_style`, `enable_answer_verification` / `verification_style`,
`answer_model`, per-category overrides) — the paper's two largest gains (R4
per-category styles, R7 answer verifier +8.9pp) are exactly those fields. Set
`honor_answer_policy: false` to defer to the shared QA agent instead; that is the
ablation, and it isolates the answer policy because retrieval is identical in
both modes.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.memo_class import MemoClass

# `import evolvemem` must resolve to the byte-identical vendored copy under src/,
# never to anything ambient. Prepended, same pattern as the harness baselines.
_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from evolvemem.evolution import (  # noqa: E402  (vendored)
    EvolutionConfig,
    EvolutionEngine,
    strong_initial_config,
    weak_initial_config,
)
from evolvemem.extractor import ExtractionConfig  # noqa: E402
from evolvemem.multi_retriever import (  # noqa: E402
    MultiViewIndex,
    RetrievalConfig,
    format_context,
)

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# Upstream ships answer-prompt adapters for these; theta's `locomo_*` /
# `longmemeval_*` prompt flags only take effect through them. dynamicmem has no
# upstream counterpart and falls back to the engine's generic answer prompt —
# recorded in README.md's faithfulness table.
_ADAPTER_FOR_DATASET = {"locomo": "locomo", "longmemeval_s": "longmemeval"}


def app_log_to_passage(log_entry: dict) -> str:
    """dynamicmem app-log entry -> passage text.

    COPIED, not imported, from `baselines/harness/hipporag2/memo.py` — methods
    never import each other (`baselines/README.md`: "Duplication over coupling"),
    and an evolve baseline owns its internals. The copy is deliberate and must
    stay byte-identical in OUTPUT to hipporag2's, so every baseline ingests the
    same dynamicmem text; `tests/test_evolvemem_baseline.py` asserts exactly that
    and will fail if either side drifts.
    """
    ts = log_entry.get("timestamp", ""); app = log_entry.get("app_name", "")
    api = log_entry.get("api_name", "")
    req = json.dumps(log_entry.get("request", {}), ensure_ascii=False)
    resp = json.dumps(log_entry.get("response", {}), ensure_ascii=False)
    domain = log_entry.get("metadata", {}).get("domain", "")
    return (f"[{ts}] App: {app}, Action: {api}\nDomain: {domain}\n"
            f"Request: {req}\nResponse: {resp}")


def theta_from_dict(data: Optional[Dict[str, Any]]) -> RetrievalConfig:
    """Build a `RetrievalConfig` from a theta dict, ignoring unknown keys.

    Unknown keys are dropped rather than raising: upstream's own contract is
    that every action-space field carries a backward-compatible default (see
    `RetrievalConfig`'s docstring), so a theta produced by a newer/older
    evolution run still loads.
    """
    if not data:
        return RetrievalConfig()
    fields = getattr(RetrievalConfig, "__dataclass_fields__", {})
    return RetrievalConfig(**{k: v for k, v in data.items() if k in fields})


def load_theta(config: Dict[str, Any]) -> RetrievalConfig:
    """Resolve theta from config: explicit dict > JSON file > named initial config.

    The JSON file may be either a bare theta object or an evolution summary
    carrying one under `final_config` (what `evolve.py` archives).
    """
    if config.get("theta"):
        return theta_from_dict(config["theta"])

    path = config.get("theta_path")
    if path:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "final_config" in data:
            data = data["final_config"]
        return theta_from_dict(data)

    initial = (config.get("initial_config") or "weak").lower()
    if initial == "strong":
        return strong_initial_config()
    if initial == "weak":
        return weak_initial_config()
    raise ValueError(
        f"initial_config must be 'weak' or 'strong', got {initial!r} "
        "(or set theta_path / theta)"
    )


def init_to_sessions(init: Dict) -> List[Tuple[str, str, List[Dict]]]:
    """recorder.init -> EvolveMem `(session_id, date_str, turns)` triples.

    Turns are `{"speaker", "text"}` dicts, which is what the vendored extractor
    reads. Dispatch mirrors `baselines/harness/hipporag2/memo.py`'s
    `_init_to_passages` so every baseline ingests the same content.
    """
    if init.get("app_logs") is not None:
        # dynamicmem: one turn per log entry, using the shared passage text so
        # passage content is identical across baselines.
        turns = [{"speaker": "system", "text": app_log_to_passage(e)}
                 for e in init["app_logs"] if isinstance(e, dict)]
        return [("app_logs", "", turns)] if turns else []

    if init.get("conversation") is not None:
        # locomo: EvolveMem's native shape — turns already carry speaker/text.
        from benchmarks.locomo.env import extract_sessions
        out: List[Tuple[str, str, List[Dict]]] = []
        for idx, date_time, turns in extract_sessions(init["conversation"]):
            keep = [t for t in turns if isinstance(t, dict)]
            if keep:
                out.append((f"session_{idx}", str(date_time), keep))
        return out

    if init.get("sessions") is not None:
        # longmemeval: role/content -> speaker/text.
        out = []
        for i, sess in enumerate(init["sessions"]):
            if not isinstance(sess, dict):
                continue
            turns = [{"speaker": str(m.get("role", "user")), "text": str(m.get("content", ""))}
                     for m in (sess.get("messages") or []) if isinstance(m, dict)]
            if turns:
                out.append((str(sess.get("session_id") or f"session_{i}"),
                            str(sess.get("date", "")), turns))
        return out

    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


class EvolveMemMemo(MemoClass):

    def __init__(self, config=None):
        super().__init__(config)
        self._theta = load_theta(self.config)
        self._dataset = self.config.get("dataset") or ""
        self._honor_answer_policy = bool(self.config.get("honor_answer_policy", True))
        self._memories: List[Dict] = []
        self._index: Optional[MultiViewIndex] = None
        self._engine: Optional[EvolutionEngine] = None
        self._embedder: Any = None
        self._answer: Optional[str] = None      # stashed by RETRIEVE, read by ANSWER
        # A fresh instance per user is the documented invariant; key any on-disk
        # artifact on an instance id rather than recorder.user_id, which is ""
        # in practice (see the note in hipporag2's memo.py).
        self._instance_id = uuid.uuid4().hex[:12]

    # ---------------- engine wiring ----------------

    def _make_llm_call(self):
        """Sync `llm_call` for the vendored engine — see `llm_bridge.py`."""
        from baselines.evolve.evolvemem.llm_bridge import make_llm_call

        return make_llm_call(self.config.get("evolve_llm_model") or "gpt-4o-mini")

    def _ensure_embedder(self):
        """Load the sentence-transformers embedder only if theta uses the
        semantic view. A weak (BM25-only) theta needs no model at all."""
        if self._embedder is not None:
            return self._embedder
        if self._theta.fusion_mode == "keyword_only" or self._theta.semantic_top_k <= 0:
            return None
        name = self.config.get("embedding_model") or "all-MiniLM-L6-v2"
        self._embedder = _shared_embedder(name)
        return self._embedder

    def _ensure_engine(self) -> EvolutionEngine:
        if self._engine is not None:
            return self._engine
        adapter = None
        adapter_name = self.config.get("benchmark_adapter") or _ADAPTER_FOR_DATASET.get(self._dataset)
        if adapter_name:
            from evolvemem.benchmarks import get_adapter
            adapter = get_adapter(adapter_name)
        ec = ExtractionConfig(
            window_size=int(self.config.get("extraction_window_size") or 40),
            overlap=int(self.config.get("extraction_overlap") or 2),
        )
        self._engine = EvolutionEngine(
            llm_call=self._make_llm_call(),
            embedder=self._ensure_embedder(),
            config=EvolutionConfig(initial_retrieval_config=self._theta,
                                   extraction_config=ec),
            adapter=adapter,
        )
        return self._engine

    @staticmethod
    async def _run_blocking(fn, *args):
        """Run the synchronous engine off the event loop."""
        return await asyncio.to_thread(fn, *args)

    # ---------------- the three hooks ----------------

    async def build_memory_from_data(self, recorder) -> None:
        sessions = init_to_sessions(recorder.init)
        if not sessions:
            return
        engine = self._ensure_engine()

        def _extract():
            memories, _results = engine.extractor.extract_sessions(sessions)
            return memories

        new = await self._run_blocking(_extract)
        if new:
            self._memories.extend(new)
            self._index = None   # rebuilt lazily on the next retrieve

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        query = str(recorder.init.get("query", ""))
        self._answer = None
        if not self._memories or not query:
            return {}

        engine = self._ensure_engine()
        if self._index is None:
            self._index = MultiViewIndex(self._memories, embedder=self._ensure_embedder())

        qa = self._build_qa(recorder, query)
        retrieved, prediction = await self._run_blocking(self._run_pipeline, engine, qa)
        self._answer = prediction

        passages = [r.content for r in retrieved if getattr(r, "content", "")]
        if not passages:
            return {}
        return {"passages": passages,
                "memory_context": format_context(retrieved, max_items=self._theta.max_context)}

    async def use_memory_to_answer(self, recorder, retrieved: Dict, prompt: str) -> Optional[str]:
        # None defers to the benchmark's shared QA agent — the ablation path.
        if not self._honor_answer_policy:
            return None
        return self._answer or None

    # ---------------- internals ----------------

    def _build_qa(self, recorder, query: str) -> Dict[str, Any]:
        """QA dict for the engine. Deliberately carries no reference answer.

        KNOWN LIMITATION — the question CATEGORY is not available here. A
        MemoClass sees only what `build_query_recorder_init` puts in
        `recorder.init`: locomo passes `{conversation, query}`, longmemeval
        `{sessions, query, question_date}`, dynamicmem `{app_logs, query}` —
        none carries `qa_metadata`. So category is 0 and theta's
        `per_category_overrides` (and per-category answer styles) fall back to
        the global config at scoring time, even though the search loop CAN
        evolve them: `evolve.py` reads the raw split, where categories exist.
        Widening `build_query_recorder_init` is a shared-eval-surface change and
        therefore a manager decision — tracked in README.md's faithfulness table,
        not worked around here.

        `question_date` IS available on longmemeval and feeds theta's time-decay
        anchor.
        """
        extras: Dict[str, Any] = {}
        for key in ("question_date", "question_time"):
            value = recorder.init.get(key)
            if value:
                extras[key] = value
        return {"question": query, "category": 0, "meta": {"extras": extras}}

    def _run_pipeline(self, engine: EvolutionEngine, qa: Dict[str, Any]):
        """Run upstream's own retrieve→answer→verify path for ONE question.

        Returns `(retrieved, prediction)`. The `_generate_answer` wrapper is the
        only way to see the FINAL retrieved list (post intent-planning / coverage
        reflection / reflection rounds) without retrieving a second time.
        """
        captured: Dict[str, Any] = {}
        original = engine._generate_answer

        def spy(question, retrieved, category, qa_dict=None):
            captured["retrieved"] = retrieved
            return original(question, retrieved, category, qa_dict)

        engine._generate_answer = spy   # type: ignore[method-assign]
        try:
            results = engine._evaluate_qa(self._index, [qa], self._theta)
        finally:
            engine._generate_answer = original   # type: ignore[method-assign]

        prediction = results[0].prediction if results else ""
        return captured.get("retrieved", []), prediction


_EMBEDDER_CACHE: Dict[str, Any] = {}


def _shared_embedder(name: str):
    """One SentenceTransformer per process, not per user — a fresh memo instance
    is created per user and the weights are hundreds of MB (zep/simplemem do the
    same)."""
    if name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER_CACHE[name] = SentenceTransformer(name)
    return _EMBEDDER_CACHE[name]
