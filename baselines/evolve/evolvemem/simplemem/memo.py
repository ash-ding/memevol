"""SimpleMem (aiming-lab/SimpleMem, MIT) as a retrieval MemoStructure —
evolvemem's faithful SUBSTRATE.

SimpleMem is the base memory system the EvolveMem paper evolves on top of
("Extending SimpleMem with self-evolving retrieval infrastructure" — its
code lives inside the SimpleMem repo). It lives INSIDE evolvemem's method
directory (not under harness/) because it exists for evolvemem's use —
per the baselines method-boundary rules, methods never import each other,
so evolvemem owns its substrate. Evaluate it standalone through
evolvemem's own launch path (`launch.py --substrate simplemem`), which
runs the FIXED text pipeline, default configuration:

BUILD — Semantic Structured Compression (paper §3.1) + Online Semantic
Synthesis (§3.2): ingestion units become `Dialogue`s fed to
`MemoryBuilder` — sliding windows (WINDOW_SIZE=40, overlap 2), one LLM
call per window performing semantic density gating Φ_gate into
`MemoryEntry`s (lossless restatement with resolved coreferences +
absolute timestamps, keywords, symbolic metadata), multi-view indexed in
LanceDB.

RETRIEVE — Intent-Aware Retrieval Planning (§3.3): `HybridRetriever`
with LLM query planning, parallel semantic/keyword/structured searches,
LLM adequacy check + up to 2 reflection rounds. Read-only w.r.t. the
store (DynamicMem query non-pollution holds). The shared QA agent
answers — `use_memory_to_answer` is NOT overridden (hipporag2/amem
pattern; upstream's AnswerGenerator is vendored but unused).

Ingestion units (recorder.init dispatch):
  locomo ("conversation"): one Dialogue per turn (speaker/text, session
    date_time → ISO timestamp).  longmemeval ("sessions"): one Dialogue
    per message (role as speaker, session date → ISO).  dynamicmem
    ("app_logs"): one Dialogue per log entry, content = the same passage
    text other baselines use (`_app_log_to_passage`, COPIED VERBATIM from
    harness/hipporag2 per duplication-over-coupling — methods do not
    import each other), speaker = app name, timestamp → ISO.

Vendored code: `vendor/simplemem/` (core + text only). Patches, all
marked in-file and listed in README.md:
  #1 minimal package __init__ (drop router/multimodal surface)
  #2 EmbeddingModel → OpenAI embeddings (drops torch/sentence-transformers;
     $SIMPLEMEM_EMBED_MODEL, default text-embedding-3-small)
  #3 llm_client records token usage into a tally this adapter drains into
     the framework's GLOBAL_TOKEN_TRACKER (run with USE_STREAMING=false)
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from common.harness_base import MemoStructure

import json as _json


def _app_log_to_passage(log_entry: dict) -> str:
    """COPIED VERBATIM from baselines/harness/hipporag2/memo.py::
    app_log_to_passage (duplication over coupling — methods never import
    each other). Keeping the byte-identical passage text means DynamicMem
    ingestion content stays comparable across baselines."""
    ts = log_entry.get("timestamp", ""); app = log_entry.get("app_name", "")
    api = log_entry.get("api_name", "")
    req = _json.dumps(log_entry.get("request", {}), ensure_ascii=False)
    resp = _json.dumps(log_entry.get("response", {}), ensure_ascii=False)
    domain = log_entry.get("metadata", {}).get("domain", "")
    return (f"[{ts}] App: {app}, Action: {api}\nDomain: {domain}\n"
            f"Request: {req}\nResponse: {resp}")


_VENDOR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# Streaming responses would bypass the usage tally (PATCH #3) and add no
# value headless; set BEFORE the vendored settings object is first read.
os.environ.setdefault("USE_STREAMING", "false")


def _import_lancedb_shielded():
    """Import lancedb with memevol's `datasets` package masked.

    lancedb registers optional HuggingFace converters when it sees a
    `datasets` module already imported (`from datasets import Dataset`) —
    memevol's benchmark package shadows HF's and lacks that attribute, so
    the unguarded import crashes (same collision amem's _st_shim handles
    for sentence-transformers). Masking `datasets`/submodules for the one
    lancedb import makes its `"datasets" in sys.modules` probe come up
    false; nothing HF-related is used afterwards.
    """
    masked = {}
    if "lancedb" not in sys.modules:
        masked = {name: sys.modules.pop(name) for name in list(sys.modules)
                  if name == "datasets" or name.startswith("datasets.")}
    try:
        import lancedb  # noqa: F401
    finally:
        sys.modules.update(masked)
    # The probe is ALSO lazy (re-run at table.add() time, when memevol's
    # `datasets` is back in sys.modules) — pre-seat the registration marker
    # so it never fires. We never hand lancedb an HF Dataset, so skipping
    # the converter costs nothing. Best-effort: the private attribute may
    # move across lancedb versions; the import-time mask above still covers
    # the common path if it does.
    try:
        from lancedb import scannable as _scannable
        _scannable._registered_modules.add("datasets")
    except (ImportError, AttributeError):
        pass


_import_lancedb_shielded()


def _to_iso(raw) -> Optional[str]:
    """Best-effort benchmark-timestamp → ISO 8601 (symbolic layer expects it)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        ts = float(raw) / 1000.0 if raw > 1e11 else float(raw)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    s = str(raw).strip()
    m = re.match(r"(\d{4}[-/]\d{2}[-/]\d{2})(?:\s*\(\w+\))?\s*(\d{2}:\d{2}(?::\d{2})?)?", s)
    if m:
        date = m.group(1).replace("/", "-")
        time_part = m.group(2) or "00:00"
        return f"{date}T{time_part}" + ("" if time_part.count(":") == 2 else ":00")
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return None


def _init_to_units(init: Dict) -> List[Dict]:
    """recorder.init → ordered [{speaker, content, timestamp}, ...]."""
    if "app_logs" in init:
        return [{"speaker": str(e.get("app_name", "app")),
                 "content": _app_log_to_passage(e),
                 "timestamp": _to_iso(e.get("timestamp"))}
                for e in init["app_logs"]]
    if "conversation" in init:
        conv = init["conversation"]
        units: List[Dict] = []
        for key in sorted(conv):
            if not re.fullmatch(r"session_\d+", key):
                continue
            iso = _to_iso(conv.get(f"{key}_date_time", ""))
            for t in conv.get(key) or []:
                if isinstance(t, dict):
                    units.append({"speaker": str(t.get("speaker", "?")),
                                  "content": str(t.get("text", "")),
                                  "timestamp": iso})
        return units
    if "sessions" in init:
        units = []
        for s in init["sessions"]:
            iso = _to_iso(s.get("date", ""))
            for m in s.get("messages", []) or []:
                units.append({"speaker": str(m.get("role", "?")),
                              "content": str(m.get("content", "")),
                              "timestamp": iso})
        return units
    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


class SimpleMemMemo(MemoStructure):
    _cfg: Dict = {}   # optional per-run overrides (evolvemem launch sets these)

    def __init__(self):
        super().__init__()
        self._system = None
        self._db_dir: Optional[str] = None
        self._n_ingested = 0   # append-only-prefix cursor (DynamicMem checkpoints)

    def __del__(self):
        if self._db_dir:
            shutil.rmtree(self._db_dir, ignore_errors=True)

    def _ensure_system(self):
        if self._system is not None:
            return
        # θ wiring: when the evolution loop stages a config file (same
        # $EVOLVEMEM_CONFIG channel the native memo uses), its dimensions
        # override _cfg — this is what makes the substrate evolvable.
        cfg_path = os.environ.get("EVOLVEMEM_CONFIG")
        if cfg_path and Path(cfg_path).exists():
            from baselines.evolve.evolvemem.action_space_simplemem import load_config
            theta = load_config(Path(cfg_path))
            self._cfg = {**self._cfg, **{k: v for k, v in theta.items() if k != "extras"}}
        # Run-level knobs → the vendored settings' env-var resolution path.
        # Same values for every instance in a run, so cross-instance env
        # writes are benign.
        for env_key, cfg_key in (("WINDOW_SIZE", "window_size"),
                                 ("ENABLE_PLANNING", "enable_planning"),
                                 ("ENABLE_REFLECTION", "enable_reflection"),
                                 ("MAX_REFLECTION_ROUNDS", "max_reflection_rounds"),
                                 ("SEMANTIC_TOP_K", "semantic_top_k"),
                                 ("KEYWORD_TOP_K", "keyword_top_k"),
                                 ("STRUCTURED_TOP_K", "structured_top_k")):
            val = self._cfg.get(cfg_key)
            if val is not None:
                os.environ[env_key] = str(val).lower() if isinstance(val, bool) else str(val)

        from simplemem.text.system import SimpleMemSystem
        self._db_dir = tempfile.mkdtemp(prefix="simplemem_lancedb_")
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            self._system = SimpleMemSystem(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self._cfg.get("simplemem_llm_model", "gpt-4.1-mini"),
                db_path=self._db_dir,
                table_name=f"mem_{uuid.uuid4().hex[:10]}",
            )

    async def _drain_usage(self):
        from simplemem.core.utils.embedding import drain_usage_tally
        from common.tokens import GLOBAL_TOKEN_TRACKER
        tally = drain_usage_tally()
        if GLOBAL_TOKEN_TRACKER is not None:
            for model, usage in tally.items():
                await GLOBAL_TOKEN_TRACKER.update(model_name=model, usage=usage)

    async def build_memory_from_data(self, recorder) -> None:
        import asyncio
        self._ensure_system()
        units = _init_to_units(recorder.init)
        fresh = units[self._n_ingested:]   # visible data is an append-only prefix
        self._n_ingested = len(units)
        if not fresh:
            return

        def _ingest():
            from simplemem.core.models.memory_entry import Dialogue
            dialogues = [Dialogue(dialogue_id=self._n_ingested - len(fresh) + i + 1,
                                  speaker=u["speaker"], content=u["content"],
                                  timestamp=u["timestamp"])
                         for i, u in enumerate(fresh)]
            # Upstream prints per-window progress; silence (amem pattern —
            # console-only adaptation, algorithm untouched). Sync + threaded,
            # so the redirect cannot leak across asyncio tasks... it CAN leak
            # across worker threads of other instances, which also print —
            # acceptable, both sinks are /dev/null-or-console noise.
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                self._system.add_dialogues(dialogues)
                # Flush the partial tail window now: with checkpoint-interleaved
                # benchmarks a query may come before more data arrives, and an
                # unflushed buffer would silently hide the newest dialogues.
                # (Deviation from upstream's one-shot flow, where finalize()
                # runs exactly once — noted in README.)
                self._system.finalize()

        await asyncio.to_thread(_ingest)
        await self._drain_usage()

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        import asyncio
        self._ensure_system()
        query = str(recorder.init.get("query", ""))

        def _retrieve():
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                return self._system.hybrid_retriever.retrieve(query)

        entries = await asyncio.to_thread(_retrieve)
        await self._drain_usage()
        memories = []
        for e in entries or []:
            item = {"memory": e.lossless_restatement}
            if e.timestamp:
                item["timestamp"] = e.timestamp
            if e.topic:
                item["topic"] = e.topic
            if e.persons:
                item["persons"] = e.persons
            memories.append(item)
        return {"memories": memories} if memories else {}
