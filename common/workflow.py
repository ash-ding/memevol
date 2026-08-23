"""Benchmark-agnostic per-user two-phase execution scaffold.

Subclass `BaseWorkflow` for each benchmark; implement the abstract hooks.
The base class handles (all unchanged from the prior DynamicMem_Workflow):

  - concurrency over users (semaphore) + per-user Exception capture
  - QA progress tracking (`_QAProgressTracker`)
  - Phase 1 dispatch: single build_memory_from_data call per visible-data batch
  - Phase 2 QA loop with per-QA failure isolation (a retrieve/answer error
    records a score=0 step with a `[Phase2_*_ERROR]` marker and continues;
    the per-user error tally lands in `recorder.failure_info`, which the
    orchestrator's sanity gate inspects)
  - `save_full_traces` (one JSON per user under traces/)
  - per-user isolation (fresh MemoClass instance per user)

To add a new benchmark, subclass `BaseWorkflow` and override the hooks listed
under "subclass hooks" below. `DynamicMemWorkflow` in
`benchmarks/dynamicmem/workflow.py` is the reference implementation.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from common.memo_class import MemoClass
from common.logger import get_logger
from common.recorder import Basic_Recorder

log = get_logger("main")


class _MemoryEncoder(json.JSONEncoder):
    """JSON encoder for common non-serializable types harnesses put into
    `steps[].retrieved_memory` (sets, Counters, datetimes, numpy scalars...).
    Used by `save_full_traces` — anything still unknown degrades to repr()."""

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


_JSON_KEY_NATIVE = (str, int, float, bool, type(None))


def _json_safe_keys(obj):
    """Recursively coerce dict keys that aren't JSON-native (str/int/float/bool/None)
    to their `str(...)` form. Returns a new structure; leaves values alone.

    Why this can't live in `_MemoryEncoder.default()`: Python's `json.dump`
    enforces the dict-key type constraint BEFORE calling the encoder's
    `default()` hook, so a custom encoder can't intercept e.g. tuple keys —
    they must be coerced via pre-processing. Harnesses commonly use
    tuple-keyed inverted indices (`Dict[Tuple[int, int], ...]` for
    (year, month), (app, api), etc.) in retrieved_memory; without this,
    every such harness's per-step retrieval record would be lost.
    """
    if isinstance(obj, dict):
        return {
            (k if isinstance(k, _JSON_KEY_NATIVE) else str(k)): _json_safe_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe_keys(x) for x in obj]
    return obj


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
    """Maximum score the judge can produce for this benchmark. Single
    source of truth for the score range — `_make_judge` reads it, and the
    host orchestrator reads it (via score.json::score_max) to normalize
    each benchmark's accuracy to [0, 1] before averaging across benchmarks.
    All three current benchmarks override it to 1 (DynamicMem: official TCE
    holistic 0.0-1.0 partial credit; LoCoMo / LongMemEval: 0/1 binary);
    the class default of 10 only applies to the generic Judge prompt."""

    def _qa_per_user_estimate(self, stage_spec: Optional[Dict]) -> int:
        """Expected number of QAs per user/sample — drives the Phase-2 progress-tracker total.

        Default honours the stage spec's `n_qa` (falling back to the legacy
        `eval_n_qa` ctor param used by the alma baseline path). Benchmarks
        where the QA count per sample is fixed (LongMemEval, always 1) or
        derived from other spec fields (DynamicMem) override this."""
        n_qa = (stage_spec or {}).get("n_qa")
        if n_qa is not None:
            return int(n_qa)
        return self.eval_n_qa or self._default_qa_per_user_hint

    def _new_memo(self):
        """The ONE way a memo instance is constructed.

        Shared by the normal per-user path and by the memory cache's
        `memo_factory`. They used to build memos differently: the cache passed
        the bare class, so a hook-restored memo came back with an EMPTY config
        while a freshly-built one got `memo_config`. Any `load_memory` that
        needs config to rebuild its backend (simplemem needs the model names to
        reopen its store) would have silently restored a broken instance.
        """
        if self.memo_config is not None:
            return self.memo_class(config=self.memo_config)
        return self.memo_class()

    # ---- Cross-stage memory cache (common/memory_cache.py) ----

    def _cache_meta(self) -> Dict:
        """Sidecar fields that must match for a cache entry to be reused —
        different ingest config or harness code ⇒ different memory."""
        from common.memory_cache import config_fingerprint
        return {
            "model": self.model,
            "max_logs": self.max_logs,
            "harness_fingerprint": self.harness_fingerprint,
            # The memo's CONFIG, not just its source. Two runs of identical code
            # with a different embedder / internal LLM / window size build
            # materially different memory; without this the second silently
            # reuses the first's. See config_fingerprint.
            "memo_config": config_fingerprint(self.memo_config),
        }

    def _cache_load(self, key: str, extra_meta: Optional[Dict] = None):
        if self.memory_cache_dir is None or not self.memory_cache_read:
            return None
        from common import memory_cache as mc
        expect = self._cache_meta()
        if extra_meta:
            expect.update(extra_meta)
        loaded = mc.load_memo(self.memory_cache_dir, key, expect,
                              memo_factory=self._new_memo)
        if loaded is not None:
            log.info(f"[memcache] hit: {key}")
        return loaded

    def _cache_save(self, memo, key: str, extra_meta: Optional[Dict] = None) -> None:
        if self.memory_cache_dir is None:
            return
        from common import memory_cache as mc
        meta = self._cache_meta()
        if extra_meta:
            meta.update(extra_meta)
        if mc.save_memo(memo, self.memory_cache_dir, key, meta):
            log.info(f"[memcache] saved: {key}")

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
        memo_class: Type[MemoClass],
        model: str,
        max_logs: Optional[int] = None,
        eval_n_qa: Optional[int] = None,
        judge_model: str = "gpt-5-mini",
        memo_config: Optional[Dict] = None,
    ):
        self.memo_class = memo_class
        # Optional constructor config for the per-user memo instances. None →
        # zero-arg instantiation (forge-evolved harnesses; their settings are
        # baked into the generated code). A dict → passed as memo_class(config=...)
        # (the parameterized harness baselines).
        self.memo_config = memo_config
        self.memo_sha = ""                  # set by caller (for logging only)
        self.status = "search"
        self.max_logs = max_logs
        self.eval_n_qa = eval_n_qa
        self.judge_model = judge_model
        # Output directory — set by launch.py before run_all_users. Full
        # traces go to {output_run_dir}/traces/.
        self.output_run_dir: Optional[Path] = None
        # Cross-stage memory cache (common/memory_cache.py). Set by launch.py
        # when the orchestrator passes a cache mount; None = caching off
        # (baselines / sanity / dev runs). `harness_fingerprint` guards
        # against reusing memory built by different harness code.
        self.memory_cache_dir: Optional[Path] = None
        # False under `memory_cache: rebuild` — writes continue, reads are
        # skipped, so Phase 1 is built fresh once per run.
        self.memory_cache_read: bool = True
        self.harness_fingerprint: str = ""
        # Lazy-constructed Judge (per common.metric.Judge). Subclasses can
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
        self, user_dir: str, eval_n_qa: Optional[int], sample_seed: Optional[str] = None
    ) -> Tuple[Any, List[Dict]]:
        """Load raw data for one user.

        `sample_seed` carries the per-step seed (stage_spec["sample_seed"],
        via common.sampling.combine_seed) for benchmarks that sample a QA
        subset per user. None (default) preserves the historical seed
        (user_dir alone) — back-compat.

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

        `reference` is the gold answer. No current benchmark feeds it into
        the prompt (gold must not leak to the QA agent); the parameter is
        kept on the hook signature for benchmarks whose official protocol
        requires reference-derived prompt content.

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

    def aggregate_extra_metrics(self, recorder_list: List[Any]) -> Dict[str, Any]:
        """Benchmark-specific aggregate metrics, merged into score.json under
        `extra_metrics`. Default: none.

        This is the extension point for anything a single benchmark wants to
        report that is NOT the promotion signal — the shared score-summary
        builder (`common.evaluate._build_score_json`) stays benchmark-agnostic,
        so per-benchmark reporting never turns into a special case inside it.

        Contract for overrides:
        - REPORTING ONLY. `benchmark_eval_score` is the promotion signal and is
          computed from the judge; nothing returned here may feed back into it.
        - Must never raise. A reporting metric that crashes must not take down
          an otherwise-valid eval — return {} instead (the caller also guards).
        - Return a JSON-serializable dict keyed by a namespace you own, e.g.
          {"locomo_lexical": {...}}.
        """
        return {}

    def _make_judge(self):
        """Construct the judge for this workflow. Override to customize
        prompt template / score range / model — e.g. for benchmark-specific
        judges (LoCoMo binary, LongMemEval per-question-type, ...)."""
        from common.metric import Judge
        # timeout / max_retries deliberately NOT overridden here — the Judge
        # defaults in common/metric.py are the single source of truth.
        return Judge(model=self.judge_model)

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
        *,
        stage: str = "stage3",
        stage_spec: Optional[Dict] = None,
        max_sample_concurrent: int = 6,
    ) -> Tuple[List[Any], int]:
        """Run Phase 1 + Phase 2 for every user in task_list.

        `stage` is the staged-evaluation tier this run belongs to
        (sanity / stage1 / stage2 / stage3). The base implementation treats
        it as informational (the memory cache is gated host-side by
        launch.py: `memory_cache_dir` is only set for stage1..3); DynamicMem's
        legacy alma branch also dispatches on it. `stage_spec` carries the
        benchmark-native size fields (e.g. n_qa, n_checkpoints, ...).

        task_list is expected to be pre-capped by the caller (a deterministic
        prefix of the split — no resampling here, staged nesting depends on
        every stage seeing a prefix-consistent task list).

        Returns (results_list, results_len). Failed users appear as Exception
        objects (with `.user_id` attribute) in the list. The returned length
        equals len(results_list); downstream filters out Exceptions itself.
        """
        qa_per_user = self._qa_per_user_estimate(stage_spec)
        total_qa = len(task_list) * qa_per_user
        qa_tracker = _QAProgressTracker(total_qa)
        log.info(
            f"Starting evaluation [{stage}]: {len(task_list)} users × ~{qa_per_user} QA = ~{total_qa} total"
        )

        semaphore = asyncio.Semaphore(max_sample_concurrent)

        async def run_one(user_dir: str):
            async with semaphore:
                try:
                    return await self.run_single_user(
                        user_dir, stage=stage, stage_spec=stage_spec, qa_tracker=qa_tracker
                    )
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
        self._raise_on_judge_outage(results)
        return list(results), len(results)

    # Fraction of judged steps whose judge failed at transport level above
    # which the whole eval aborts instead of publishing a fake near-0 score.
    JUDGE_OUTAGE_FRACTION = 0.5

    def _raise_on_judge_outage(self, results: List[Any]) -> None:
        """Abort the eval when the judge itself was down for most steps.

        The official DynamicMem protocol aborts on judge failure; our port
        scores isolated judge failures 0 per item (documented deviation).
        Without this guard, an OpenAI outage during Phase 2 produces a
        legitimate-looking near-0 score.json that can wrongly ELIMINATE a
        harness at a gauntlet threshold. Raising here means no score.json
        is written — the orchestrator records an eval failure (retryable)
        rather than a score.

        Counts only steps the judge actually ran on (Phase2_* error steps
        are harness-side, judge never invoked). Needs >= 2 judged steps so
        a single flaky call in a 1-QA sanity run doesn't abort.
        """
        judged = 0
        failed = 0
        for r in results:
            if isinstance(r, Exception):
                continue
            for s in getattr(r, "steps", []):
                reason = str(s.get("judge_reason", ""))
                if reason.startswith("[Phase2_"):
                    continue
                judged += 1
                # "Judge error:" = common.metric.Judge transport exhaustion;
                # "JUDGE_ERROR:" = DynamicMem TCE judge-call failure marker.
                if reason.startswith("Judge error:") or "JUDGE_ERROR:" in reason:
                    failed += 1
        if judged >= 2 and failed / judged > self.JUDGE_OUTAGE_FRACTION:
            raise RuntimeError(
                f"judge outage: {failed}/{judged} judged steps failed at the "
                f"transport level — aborting eval instead of publishing a "
                f"fake near-0 score (official protocol aborts on judge failure)"
            )

    # ---- Single-user execution ----

    async def run_single_user(
        self,
        user_dir: str,
        *,
        stage: str = "stage3",
        stage_spec: Optional[Dict] = None,
        qa_tracker: Optional[_QAProgressTracker] = None,
    ) -> Basic_Recorder:
        """Run Phase 1 (update) then Phase 2 (retrieve + QA) for one user."""
        # Agent is lazy-imported so module load stays cheap & avoids cycles.
        from common.llm import Agent

        user_tag = user_dir[-15:]

        qa_size = (stage_spec or {}).get("n_qa", self.eval_n_qa)
        sample_seed = (stage_spec or {}).get("sample_seed")
        init_data, qa_pairs = await self.load_user_data(user_dir, qa_size, sample_seed=sample_seed)

        memo = self._new_memo()

        # Cross-stage memory cache: Phase-1 input for this benchmark family is
        # independent of the QA count, so the final memory built at an earlier
        # stage is bit-for-bit reusable here.
        from common.memory_cache import user_key as _mc_user_key
        cache_key = f"{_mc_user_key(user_dir)}__final"

        t1 = time.time()
        cached = self._cache_load(cache_key)
        if cached is not None:
            memo = cached
            log.info(f"[Phase 1] User {user_tag} memory loaded from cache "
                     f"({time.time() - t1:.1f}s)")
        else:
            await self._phase1_update(memo, init_data)
            t1_elapsed = time.time() - t1
            item_count = self._phase1_item_count(init_data)
            log.info(
                f"[Phase 1] User {user_tag} update complete "
                f"({item_count} {self._phase1_item_label}, {t1_elapsed:.1f}s)"
            )
            self._cache_save(memo, cache_key)

        t2 = time.time()
        recorder = self.recorder_class()
        recorder.user_id = user_tag
        await self.phase1_log_init(recorder, init_data)

        # timeout / max_retries deliberately NOT overridden — Agent defaults
        # in common/llm.py are the single source of truth.
        agent = Agent(system_prompt="", model=self.model)

        # Phase-2 error tally → recorder.failure_info. The sanity gate
        # (forge/orchestrator.py::_collect_sanity_errors) fails a harness
        # whose per_user entries carry a non-empty failure_info, so a
        # broken retrieve_memory_for_query is caught at sanity size instead of
        # silently burning a full stage1 eval on all-zero steps.
        n_retrieve_err = 0
        n_answer_err = 0
        first_err = ""

        for qa in qa_pairs:
            retrieve_recorder = self.recorder_class()
            retrieve_recorder.init = self.build_query_recorder_init(init_data, qa)

            try:
                retrieved = await memo.retrieve_memory_for_query(retrieve_recorder)
            except Exception as exc:
                # Per-QA failure isolation: log a score=0 step and continue
                # to the next QA — a single bad query must not silently
                # truncate the rest of the user's QA loop (which would
                # under-report the harness's real capability).
                n_retrieve_err += 1
                first_err = first_err or f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning(
                    f"retrieve_memory_for_query failed for {user_tag} on QA "
                    f"{len(recorder.steps)+1}/{len(qa_pairs)} "
                    f"({type(exc).__name__}: {str(exc)[:120]}) "
                    f"— recording score=0 and continuing"
                )
                await self.log_qa_step(
                    recorder=recorder,
                    query=qa["query"],
                    predicted="",
                    reference=qa.get("reference", ""),
                    score=0,
                    judge_reason=(
                        f"[Phase2_Retrieve_ERROR] {type(exc).__name__}: "
                        f"{str(exc)[:200]}"
                    ),
                    qa_metadata=self.build_qa_metadata(qa),
                    retrieved_memory={},
                    relevant_context=self.extract_relevant_context(qa, init_data),
                )
                if qa_tracker:
                    await qa_tracker.increment()
                continue

            relevant_context = self.extract_relevant_context(qa, init_data)
            qa_metadata = self.build_qa_metadata(qa)

            prompt = self.build_qa_prompt(
                qa["query"], retrieved, qa_metadata,
                reference=qa.get("reference", ""),
            )
            system_msg = prompt[0]["content"]
            user_msg = prompt[1]["content"]

            full_prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
            try:
                answer = await memo.use_memory_to_answer(retrieve_recorder, retrieved, full_prompt)
                if answer is None:
                    agent.messages = [{"role": "system", "content": system_msg}]
                    answer = await agent.ask(
                        user_msg, with_history=False, reasoning_effort=self.reasoning_effort
                    )
            except Exception as exc:
                # QA-agent transport failure (retries already exhausted in
                # common.llm). Mirror the retrieve-error path: record a
                # marked score=0 step instead of judging an empty answer —
                # otherwise an API outage is indistinguishable in traces
                # from a genuinely wrong answer.
                n_answer_err += 1
                first_err = first_err or f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning(f"Agent ask failed for {user_tag}: {exc}")
                await self.log_qa_step(
                    recorder=recorder,
                    query=qa["query"],
                    predicted="",
                    reference=qa.get("reference", ""),
                    score=0,
                    judge_reason=(
                        f"[Phase2_Answer_ERROR] {type(exc).__name__}: "
                        f"{str(exc)[:200]}"
                    ),
                    qa_metadata=qa_metadata,
                    retrieved_memory=retrieved,
                    relevant_context=relevant_context,
                )
                if qa_tracker:
                    await qa_tracker.increment()
                continue

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

        if n_retrieve_err or n_answer_err:
            recorder.failure_info = (
                f"phase2_errors: retrieve={n_retrieve_err}/{len(qa_pairs)} "
                f"answer={n_answer_err}/{len(qa_pairs)}; first: {first_err}"
            )

        t2_elapsed = time.time() - t2
        scores = [s["score"] for s in recorder.steps]
        avg = sum(scores) / len(scores) if scores else 0.0
        await recorder.set_reward(avg)
        log.info(f"[Phase 2] User {user_tag} QA complete: reward={avg:.2f}/{self.judge_score_max} ({len(scores)} QA, {t2_elapsed:.1f}s)")
        return recorder

    # ---- Phase 1 dispatch ----

    async def _phase1_update(self, memo: MemoClass, init_data: Any) -> None:
        """Hand the whole visible `init_data` to the memo in ONE build_memory_from_data
        call. Ingestion granularity is the memo's own choice; the workflow
        does not chunk."""
        if not isinstance(init_data, list):
            raise TypeError(
                f"{type(self).__name__}: default _phase1_update expects init_data to be a list; "
                f"got {type(init_data).__name__}. Override _phase1_update for non-list init."
            )
        # NOTE: max_logs is NOT segment-aware — it trims THIS call's init_data, so for DynamicMem's per-checkpoint delta it would trim each delta, not the total visible pool (latent: defaults to None, unused in forge).
        items = init_data[-self.max_logs:] if self.max_logs else init_data
        log.info(f"[Phase 1] build_memory_from_data ({len(items)} {self._phase1_item_label})")
        r = self.recorder_class()
        await self.phase1_log_init(r, items)
        try:
            await memo.build_memory_from_data(r)
        except Exception as exc:
            log.warning(f"build_memory_from_data failed: {exc}")
            raise RuntimeError(f"[Phase1_Update] {type(exc).__name__}: {exc}") from exc

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
                # Serialize FIRST, write only on success — a serialization
                # error must never leave a truncated/0-byte trace file.
                text = json.dumps(
                    _json_safe_keys(payload),
                    indent=2, ensure_ascii=False, cls=_MemoryEncoder,
                )
                file_path.write_text(text, encoding="utf-8")
            except Exception as e:
                log.warning(f"Failed to save trace for {user_id}: {e}")
