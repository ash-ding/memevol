"""Benchmark-agnostic per-user two-phase execution scaffold.

Subclass `BaseWorkflow` for each benchmark; implement the abstract hooks.
The base class handles (all unchanged from the prior DynamicMem_Workflow):

  - concurrency over users (semaphore) + per-user Exception capture
  - QA progress tracking (`_QAProgressTracker`)
  - Phase 1 dispatch: `all_at_once` / `chunked` / `sequential`
  - Phase 2 QA loop with timeout + partial-preservation + failure_info
  - `save_full_traces` (one JSON per user under traces/)
  - memory dumping (Chroma / NetworkX / dict introspection)
  - per-user isolation (fresh MemoStructure instance per user)

To add a new benchmark, subclass `BaseWorkflow` and override the hooks listed
under "subclass hooks" below. `DynamicMemWorkflow` in
`datasets/dynamicmem/workflow.py` is the reference implementation.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import traceback
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from common.harness_base import Basic_Recorder, MemoStructure, Sub_memo_layer
from common.logger import get_logger

log = get_logger("main")


# ---------------------------------------------------------------------------
# JSON / memory-dump helpers (unchanged — benchmark-agnostic)
# ---------------------------------------------------------------------------

class _MemoryEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serializable types in memory databases."""

    def default(self, obj):
        if isinstance(obj, set):
            return sorted(obj) if all(isinstance(x, str) for x in obj) else list(obj)
        if isinstance(obj, (Counter, defaultdict)):
            return dict(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, '__float__'):
            return float(obj)
        if hasattr(obj, '__int__'):
            return int(obj)
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def _count_entries(db) -> dict:
    """Count entries in a dict/list database (one level)."""
    if isinstance(db, list):
        return {"_total": len(db)}
    if isinstance(db, dict):
        counts = {}
        for k, v in db.items():
            if isinstance(v, (list, set)):
                counts[k] = len(v)
            elif isinstance(v, dict):
                counts[k] = len(v)
            else:
                counts[k] = 1
        return counts
    return {}


def _dump_layer(attr, full: bool) -> dict:
    """Extract data from a single Sub_memo_layer.

    full=True  → complete database content (for status=test).
    full=False → statistics summary only (for status=search).
    """
    layer_data = {"layer_intro": getattr(attr, "layer_intro", "")}
    db = attr.database

    if db is None:
        layer_data["status"] = "empty"
        layer_data["n_entries"] = 0
        return layer_data

    db_type = type(db).__name__

    # --- NetworkX graphs ---
    if db_type in ("Graph", "DiGraph"):
        import networkx as nx
        n_nodes = db.number_of_nodes()
        n_edges = db.number_of_edges()
        layer_data["type"] = "networkx"
        layer_data["n_nodes"] = n_nodes
        layer_data["n_edges"] = n_edges
        layer_data["status"] = "ok" if n_nodes > 0 else "empty"
        type_counts = {}
        for _, d in db.nodes(data=True):
            t = d.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        layer_data["node_type_counts"] = type_counts
        layer_data["sample_nodes"] = list(db.nodes())[:5]
        if full:
            layer_data["database"] = nx.node_link_data(db)
        return layer_data

    # --- Chroma vector store ---
    if "Chroma" in db_type:
        try:
            chroma_data = db.get(include=["documents", "metadatas"])
            ids = chroma_data.get("ids", [])
            docs = chroma_data.get("documents", [])
            metas = chroma_data.get("metadatas", [])
            layer_data["type"] = "chroma"
            layer_data["n_documents"] = len(ids)
            layer_data["status"] = "ok" if len(ids) > 0 else "empty"
            layer_data["sample_documents"] = docs[:3]
            if metas:
                layer_data["metadata_keys"] = sorted(set(k for m in metas[:10] for k in (m or {}).keys()))
            if full:
                layer_data["database"] = {"ids": ids, "documents": docs, "metadatas": metas}
        except Exception as e:
            layer_data["type"] = "chroma_error"
            layer_data["status"] = f"extraction failed: {e}"
        return layer_data

    # --- Plain dict/list ---
    if isinstance(db, (dict, list)):
        layer_data["type"] = "dict" if isinstance(db, dict) else "list"
        layer_data["entry_counts"] = _count_entries(db)
        layer_data["status"] = "ok" if len(db) > 0 else "empty"
        if isinstance(db, dict):
            layer_data["top_level_keys"] = list(db.keys())[:20]
        if full:
            layer_data["database"] = db
        return layer_data

    # --- Unknown type ---
    layer_data["type"] = db_type
    layer_data["status"] = "unknown_type"
    if full:
        layer_data["database"] = repr(db)
    return layer_data


def _dump_memory(memo: MemoStructure, user_id: str, dump_dir: Path, full: bool = False) -> None:
    """Dump memory state after Phase 1 to dump_dir/<user_id>.json."""
    safe_user_id = user_id.replace("/", "_").replace("\\", "_")
    dump_dir.mkdir(parents=True, exist_ok=True)

    dump = {}
    for attr_name in dir(memo):
        if attr_name.startswith("_"):
            continue
        attr = getattr(memo, attr_name, None)
        if not isinstance(attr, Sub_memo_layer):
            continue
        dump[attr_name] = _dump_layer(attr, full=full)

    file_path = dump_dir / f"{safe_user_id}.json"
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2, ensure_ascii=False, cls=_MemoryEncoder)
    except Exception as e:
        log.warning(f"Failed to dump memory for {user_id}: {e}")


class _QAProgressTracker:
    """Shared counter for QA progress across concurrent users."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self._lock = asyncio.Lock()
        self._last_pct = -1

    async def increment(self) -> None:
        async with self._lock:
            self.completed += 1
            pct = int(self.completed / self.total * 100) if self.total > 0 else 100
            if pct // 10 > self._last_pct // 10:
                self._last_pct = pct
                log.info(f"[Phase 2] QA Progress: {self.completed}/{self.total} ({pct}%)")


# ---------------------------------------------------------------------------
# BaseWorkflow
# ---------------------------------------------------------------------------

class BaseWorkflow(ABC):
    """Per-user two-phase scheduler. Subclass per benchmark."""

    # ---- Subclass overrides (class attributes) ----

    recorder_class: Type[Basic_Recorder]
    """Recorder class used for both the accumulator recorder and per-QA
    retrieve recorders. Must be set by subclass."""

    _phase1_item_label: str = "items"
    """Human-readable label for init_data elements, used in Phase 1 log
    messages. DynamicMem uses "logs"; LoCoMo might use "sessions"."""

    _default_qa_per_user_hint: int = 178
    """Progress-bar hint when eval_n_qa is None (purely cosmetic)."""

    judge_score_max: int = 10
    """Maximum integer score the judge can produce for this benchmark.
    Single source of truth for the score range — `_make_judge` reads it,
    and the host orchestrator reads it (via score.json) to normalize each
    benchmark's accuracy to [0, 1] before averaging across benchmarks
    (so DynamicMem 0-10 and LoCoMo 0-1 carry equal weight in the mean).
    Override per benchmark: 10 for DynamicMem (partial credit), 1 for
    LoCoMo / LongMemEval (binary)."""

    def _qa_per_user_estimate(self, mode: str, check_n_qa: int) -> int:
        """Expected number of QAs per user/sample — drives the Phase-2 progress-tracker total.

        Default honours `--eval-n-qa` (or `--check-n-qa` via `check_n_qa`
        in check mode). Benchmarks where the QA count per sample is fixed
        (e.g. LongMemEval, always 1) should override this to ignore
        user-supplied `--*-n-qa`."""
        if mode == "check":
            return check_n_qa
        return self.eval_n_qa or self._default_qa_per_user_hint

    def _phase1_item_count(self, init_data) -> "int | str":
        """Return the count of "items" in Phase 1's init_data — used purely
        for the end-of-Phase-1 log message. Default: `len(init_data)` when
        sized (works for list-shaped init like DynamicMem). Subclasses whose
        init_data is a container of sub-items should override (e.g.
        LoCoMo returns the session count, not the dict key count)."""
        try:
            return len(init_data)  # type: ignore[arg-type]
        except TypeError:
            return "?"

    # ---- Constructor ----

    def __init__(
        self,
        memo_class: Type[MemoStructure],
        model: str,
        update_type: str = "all_at_once",
        n_chunks: int = 5,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        judge_model: str = "gpt-5-mini",
        memory_dumps: str = "full",
    ):
        self.memo_class = memo_class
        self.memo_sha = ""                  # set by caller (for logging only)
        self.status = "search"
        self.update_type = update_type
        self.n_chunks = n_chunks
        self.max_logs = max_logs
        self.eval_n_qa = eval_n_qa
        self.judge_model = judge_model
        # memory_dumps controls what (if anything) is written under
        # output_run_dir/memory_dumps/<user>.json after Phase 1 completes.
        # Only honoured when mode="eval"; mode="check" always skips dumps.
        #   "full"  — full contents (Chroma docs+metas, NetworkX nodes+edges, dict/list raw)
        #   "stats" — only sizes + small samples (cheap; matches alma's search-mode behavior)
        #   "none"  — skip the dump entirely
        if memory_dumps not in ("full", "stats", "none"):
            raise ValueError(
                f"memory_dumps must be one of full/stats/none, got {memory_dumps!r}"
            )
        self.memory_dumps = memory_dumps
        # Output directory — set by launch.py before run_all_users. Memory dumps
        # go to {output_run_dir}/memory_dumps/, full traces to {output_run_dir}/traces/.
        self.output_run_dir: Optional[Path] = None
        # Lazy-constructed Judge (per common.judge.Judge). Subclasses can
        # override `_make_judge` to customize prompt / score range.
        self._judge_instance = None

        # Parse "model/reasoning_effort" format (e.g. "gpt-4o-mini/low")
        if "/" in model:
            self.model, self.reasoning_effort = model.split("/", 1)
        else:
            self.model = model
            self.reasoning_effort = None

    # ---- Subclass hooks (must implement) ----

    @abstractmethod
    async def load_user_data(
        self, user_dir: str, eval_n_qa: Optional[int]
    ) -> Tuple[Any, List[Dict]]:
        """Load raw data for one user.

        Returns (init_data, qa_pairs):
          init_data: opaque payload passed back to `phase1_log_init` and
                     `build_query_recorder_init`. For DynamicMem, the
                     full `app_logs` list.
          qa_pairs:  list of QA dicts. Each is expected to have at least
                     `query`, `reference`, and an optional `metadata` dict.
        """

    @abstractmethod
    async def phase1_log_init(self, recorder: Basic_Recorder, chunk: Any) -> None:
        """Populate `recorder.init` with a Phase 1 chunk of init_data.

        For DynamicMem:  `await recorder.log_init(chunk)`   (chunk is a list of
        app_log dicts, possibly a singleton for sequential mode).
        """

    @abstractmethod
    def build_query_recorder_init(self, init_data: Any, qa: Dict) -> Dict:
        """Build the `recorder.init` dict for a single Phase 2 retrieve call.

        For DynamicMem:  {"app_logs": init_data, "query": qa["query"]}
        """

    @abstractmethod
    def build_qa_prompt(
        self,
        query: str,
        retrieved: Dict,
        qa_metadata: Dict,
        reference: str = "",
    ) -> List[Dict]:
        """Two-message prompt (system + user) for the QA agent.

        `reference` is the gold answer; most benchmarks ignore it (they shouldn't
        leak gold to the QA agent), but LoCoMo's category-5 adversarial QAs use
        it to construct a binary-choice prompt per the LoCoMo paper / A-mem
        baseline (the model is shown two options including the truth and asked
        which is correct — testing whether it hallucinates or correctly says
        "Not mentioned in the conversation").

        `qa_metadata` is the dict returned by `build_qa_metadata(qa)` for this
        step — benchmarks that need extra context (e.g. LongMemEval uses the
        question_date for temporal reasoning) can pull fields from here.
        Simple benchmarks can ignore it.
        """

    @abstractmethod
    def extract_relevant_context(self, qa: Dict, init_data: Any) -> Any:
        """Ground-truth context relevant to this QA (logged in the trace).

        For DynamicMem: look up app_log_ids listed in qa['metadata'].
        """

    @abstractmethod
    def build_qa_metadata(self, qa: Dict) -> Dict:
        """Per-step metadata to persist in the trace."""

    @abstractmethod
    async def log_qa_step(
        self,
        recorder: Basic_Recorder,
        query: str,
        predicted: str,
        reference: str,
        score: int,
        judge_reason: str,
        qa_metadata: Dict,
        retrieved_memory: Dict,
        relevant_context: Any,
    ) -> None:
        """Call `recorder.log_step(...)` with the benchmark-specific field
        names. DynamicMem maps `relevant_context` to `relevant_app_logs`."""

    def _make_judge(self):
        """Construct the judge for this workflow. Override to customize
        prompt template / score range / model — e.g. for benchmark-specific
        judges (LoCoMo binary, LongMemEval per-question-type, ...)."""
        from common.judge import Judge
        return Judge(
            model=self.judge_model,
            timeout=180,        # match Agent in this workflow (a bit longer than Judge default 150)
            max_retries=5,      # match Agent's retry budget
        )

    async def judge(
        self,
        query: str,
        predicted: str,
        reference: str,
        qa_metadata: Optional[Dict] = None,
    ) -> Tuple[int, str]:
        """Default impl uses self._make_judge() lazily. Subclasses with a
        single benchmark-wide prompt only need to override `_make_judge`.

        `qa_metadata` carries per-question fields (question_type, etc.) and
        is unused by the default impl; LongMemEval overrides this method to
        dispatch among multiple prompts based on question_type.
        """
        if self._judge_instance is None:
            self._judge_instance = self._make_judge()
        return await self._judge_instance.score(query, predicted, reference)

    # ---- Main entry point: run all users ----

    async def run_all_users(
        self,
        task_list: List[str],
        mode: str = "eval",
        max_sample_concurrent: int = 6,
        check_n_samples: int = 6,
        check_n_qa: int = 3,
    ) -> Tuple[List[Any], int]:
        """Run Phase 1 + Phase 2 for every user in task_list.

        mode='check' samples check_n_samples users with check_n_qa QA pairs each.

        Returns (results_list, results_len). Failed users appear as Exception
        objects (with `.user_id` attribute) in the list. The returned length
        equals len(results_list); downstream filters out Exceptions itself.
        """
        if mode == "check":
            task_list = random.sample(task_list, min(check_n_samples, len(task_list)))

        qa_per_user = self._qa_per_user_estimate(mode, check_n_qa)
        total_qa = len(task_list) * qa_per_user
        qa_tracker = _QAProgressTracker(total_qa)
        log.info(f"Starting evaluation: {len(task_list)} users × ~{qa_per_user} QA = ~{total_qa} total")

        semaphore = asyncio.Semaphore(max_sample_concurrent)

        async def run_one(user_dir: str):
            async with semaphore:
                try:
                    return await self.run_single_user(user_dir, mode=mode, qa_tracker=qa_tracker, check_n_qa=check_n_qa)
                except Exception as exc:
                    user_tag = user_dir[-15:]
                    log.error(f"[ERROR] User {user_tag} failed: {exc}\n{traceback.format_exc()}")
                    try:
                        exc.user_id = user_tag
                    except Exception:
                        pass
                    return exc

        t0 = time.time()
        results = await asyncio.gather(*[run_one(u) for u in task_list])
        elapsed = time.time() - t0
        valid_count = sum(1 for r in results if not isinstance(r, Exception))
        log.info(f"Evaluation complete: {valid_count}/{len(task_list)} users succeeded in {elapsed:.1f}s")
        return list(results), len(results)

    # ---- Single-user execution ----

    async def run_single_user(
        self,
        user_dir: str,
        mode: str = "eval",
        qa_tracker: Optional[_QAProgressTracker] = None,
        check_n_qa: int = 3,
    ) -> Basic_Recorder:
        """Run Phase 1 (update) then Phase 2 (retrieve + QA) for one user."""
        # Agent is lazy-imported so module load stays cheap & avoids cycles.
        from common.llm import Agent

        user_tag = user_dir[-15:]

        qa_size = check_n_qa if mode == "check" else self.eval_n_qa
        init_data, qa_pairs = await self.load_user_data(user_dir, qa_size)

        memo = self.memo_class()

        t1 = time.time()
        await self._phase1_update(memo, init_data)
        t1_elapsed = time.time() - t1
        item_count = self._phase1_item_count(init_data)
        log.info(
            f"[Phase 1] User {user_tag} update complete "
            f"({item_count} {self._phase1_item_label}, {t1_elapsed:.1f}s)"
        )

        # Dump memory database for post-hoc inspection. Policy:
        #   - mode=="check": never dump (smoke/sanity runs stay cheap + clean)
        #   - mode=="eval":  dump per self.memory_dumps (full / stats / none)
        if (
            mode == "eval"
            and self.memory_dumps != "none"
            and self.output_run_dir is not None
        ):
            _dump_memory(
                memo, user_tag,
                self.output_run_dir / "memory_dumps",
                full=(self.memory_dumps == "full"),
            )

        t2 = time.time()
        recorder = self.recorder_class()
        recorder.user_id = user_tag
        await self.phase1_log_init(recorder, init_data)

        agent = Agent(system_prompt="", model=self.model, timeout=180, max_retries=5)

        for qa in qa_pairs:
            retrieve_recorder = self.recorder_class()
            retrieve_recorder.init = self.build_query_recorder_init(init_data, qa)

            try:
                retrieved = await asyncio.wait_for(
                    memo.general_retrieve(retrieve_recorder), timeout=300
                )
            except asyncio.TimeoutError:
                log.warning(
                    f"general_retrieve timed out for user {user_tag}, question: {qa['query'][:50]} "
                    f"— stopping user at {len(recorder.steps)}/{len(qa_pairs)} QAs"
                )
                recorder.failure_info = (
                    f"[Phase2_Retrieve] TimeoutError at QA {len(recorder.steps)+1}/{len(qa_pairs)}, "
                    f"query: {qa['query'][:120]}"
                )
                break
            except Exception as exc:
                log.warning(
                    f"general_retrieve failed for {user_tag}: {exc} "
                    f"— stopping user at {len(recorder.steps)}/{len(qa_pairs)} QAs"
                )
                recorder.failure_info = (
                    f"[Phase2_Retrieve] {type(exc).__name__}: {exc} "
                    f"(at QA {len(recorder.steps)+1}/{len(qa_pairs)}, query: {qa['query'][:120]})"
                )
                break

            relevant_context = self.extract_relevant_context(qa, init_data)
            qa_metadata = self.build_qa_metadata(qa)

            prompt = self.build_qa_prompt(
                qa["query"], retrieved, qa_metadata,
                reference=qa.get("reference", ""),
            )
            system_msg = prompt[0]["content"]
            user_msg = prompt[1]["content"]

            agent.messages = [{"role": "system", "content": system_msg}]
            try:
                answer = await agent.ask(
                    user_msg,
                    with_history=False,
                    reasoning_effort=self.reasoning_effort,
                )
            except Exception as exc:
                log.warning(f"Agent ask failed for {user_dir}: {exc}")
                answer = ""

            score, judge_reason = await self.judge(
                qa["query"], answer, qa.get("reference", ""),
                qa_metadata=qa_metadata,
            )

            await self.log_qa_step(
                recorder=recorder,
                query=qa["query"],
                predicted=answer,
                reference=qa.get("reference", ""),
                score=score,
                judge_reason=judge_reason,
                qa_metadata=qa_metadata,
                retrieved_memory=retrieved,
                relevant_context=relevant_context,
            )

            if qa_tracker:
                await qa_tracker.increment()

        t2_elapsed = time.time() - t2
        scores = [s["score"] for s in recorder.steps]
        avg = sum(scores) / len(scores) if scores else 0.0
        await recorder.set_reward(avg)
        log.info(f"[Phase 2] User {user_tag} QA complete: reward={avg:.2f}/10 ({len(scores)} QA, {t2_elapsed:.1f}s)")
        return recorder

    # ---- Phase 1 dispatch ----

    async def _phase1_update(self, memo: MemoStructure, init_data: Any) -> None:
        """Dispatch general_update calls according to update_type.

        Default impl assumes `init_data` is a list whose elements can be
        passed to `phase1_log_init`. Subclasses may override entirely if
        their init_data is not list-shaped.
        """

        async def _call_update(chunk: Any) -> None:
            r = self.recorder_class()
            await self.phase1_log_init(r, chunk)
            try:
                await memo.general_update(r)
            except Exception as exc:
                log.warning(f"general_update failed: {exc}")
                raise RuntimeError(f"[Phase1_Update] {type(exc).__name__}: {exc}") from exc

        if not isinstance(init_data, list):
            raise TypeError(
                f"{type(self).__name__}: default _phase1_update expects init_data to be a list; "
                f"got {type(init_data).__name__}. Override _phase1_update for non-list init."
            )

        total = len(init_data)

        if self.update_type == "sequential":
            for idx, item in enumerate(init_data, 1):
                if idx == 1 or idx % max(1, total // 10) == 0 or idx == total:
                    log.info(f"[Phase 1] general_update progress: {idx}/{total} ({idx*100//total}%)")
                await _call_update([item])

        elif self.update_type == "chunked":
            n = max(1, self.n_chunks)
            chunk_size = max(1, (total + n - 1) // n)
            chunks = list(range(0, total, chunk_size))
            for chunk_idx, i in enumerate(chunks, 1):
                log.info(f"[Phase 1] general_update progress: chunk {chunk_idx}/{len(chunks)}")
                await _call_update(init_data[i: i + chunk_size])

        else:  # all_at_once
            items = init_data[-self.max_logs:] if self.max_logs else init_data
            log.info(f"[Phase 1] general_update started ({len(items)} {self._phase1_item_label}, mode=all_at_once)")
            await _call_update(items)

    # ---- Full-trace persistence (no sampling) ----

    def save_full_traces(self, recorder_list: List[Any]) -> None:
        """Write every user's full QA trajectory to output_run_dir/traces/<user_id>.json.

        Failed users (Exception objects) are skipped; partial recorders are
        serialized with their failure_info preserved.
        """
        if self.output_run_dir is None:
            log.warning("save_full_traces: output_run_dir not set, skipping")
            return

        traces_dir = self.output_run_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        for rec in recorder_list:
            if isinstance(rec, Exception):
                continue
            user_id = getattr(rec, "user_id", "") or "unknown"
            steps = getattr(rec, "steps", [])
            payload = {
                "user_id": user_id,
                "reward": float(getattr(rec, "reward", 0.0)),
                "failure_info": getattr(rec, "failure_info", None),
                "n_qa": len(steps),
                "steps": steps,
            }
            safe = user_id.replace("/", "_").replace("\\", "_") or "unknown"
            file_path = traces_dir / f"{safe}.json"
            try:
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False, cls=_MemoryEncoder)
            except Exception as e:
                log.warning(f"Failed to save trace for {user_id}: {e}")
