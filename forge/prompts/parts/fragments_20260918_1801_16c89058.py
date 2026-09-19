"""Keyed substitution data for prompt version 20260918_1801_16c89058.

Not prose: every value here is indexed by the renderer — by sentinel name
(SANITY_*_SUBS), by the run's sorted `metrics` tuple (OBJECTIVE_AXES_SUBS), or
by dataset key (DATASET_INFO). That is why it stays Python while the four
prose prompts live beside it as evolving_/design_/task_/fix_ .md.

Sliced verbatim from the pre-split template so the hand-formatted blocks are
unchanged.
"""

PROMPT_VERSION = "20260918_1801_16c89058"


#: `<<OBJECTIVE_AXES_BLOCK>>` per `metrics:` setting — the axes the frontier
#: actually carries for this run, plus what to optimize. Keyed by the metric
#: set as a sorted tuple; the renderer picks one.
OBJECTIVE_AXES_SUBS = {
    ("accuracy",): """      There is NO algorithmic selection — YOU pick which prior(s) to build on.
      This run optimizes ACCURACY only; cost axes are not recorded on the
      frontier. Compare `accuracy_<dataset>` at the same `stage_<dataset>`;
      use robustness as a tie-breaker to avoid brittle priors.""",
    ("accuracy", "efficiency"): """        • cost_tokens_per_query_<dataset>
                              — what the memory system costs per query on that
                                benchmark: (build + retrieve tokens + the
                                tokens the memory adds to the QA prompt),
                                amortized over that user's queries and meaned
                                over users. LOW = cheaper. Excludes the judge
                                and the QA agent's own prompt, so it measures
                                the MEMORY, not the answerer
        • tokens_total        — total LLM tokens used during eval; LOW = cheaper
      There is NO algorithmic selection — YOU pick which prior(s) to build on.
      This run optimizes ACCURACY **and** EFFICIENCY: a design that buys a
      point of accuracy with several times the tokens is not an improvement,
      and neither is a cheap design that cannot answer. Compare
      `accuracy_<dataset>` at the same `stage_<dataset>`, then read
      `cost_tokens_per_query_<dataset>` alongside it; use robustness as a
      tie-breaker.""",
}


SANITY_ON_SUBS = {
    "<<SANITY_TREE_BLOCK>>": "\n          └── sanity/          pre-eval sanity-check artifacts (smaller)\n              ├── score.json\n              └── traces/",
    "<<RUNS_SANITY_SUFFIX>>": "[_sanity]",
    "<<LOG_SANITY_NOTE>>": ", which sanity passes/fails",
    "<<SELF_VAL_COMPARE>>": "~30s for the sanity check",
    "<<SANITY_SECTION>>": """
# Sanity check (post-propose, automatic)

After you finish, your harness runs on a tiny sanity check (a couple of
tasks on 1 sample per benchmark, real data). If it crashes, you'll be asked
to Read your memo.py (and src/), diagnose from the error trace, and Edit —
up to a few attempts. Write code robust to realistic input, not just the
happy path. Self-validation above prevents most sanity-check fixups.
""",
}

SANITY_OFF_SUBS = {
    "<<SANITY_TREE_BLOCK>>": "",
    "<<RUNS_SANITY_SUFFIX>>": "",
    "<<LOG_SANITY_NOTE>>": "",
    "<<SELF_VAL_COMPARE>>": "minutes for the full eval",
    "<<SANITY_SECTION>>": "",
}


# ===========================================================================
# DATASET_INFO + DATASET_RENDER_ORDER
# ===========================================================================

DATASET_INFO = {
    "dynamicmem": {
        "display_name": "DynamicMem",
        "qa_metadata": """    DynamicMem:    {task_family, checkpoint_id, state_key, qa_id,
                      service_family, domain, app_log_ids, field_judgments,
                      evidence_prf}
                     • task_family: "state_completion" (fill a state template)
                       or "apply_service" (personalized service task)
                     • field_judgments: the official judge's per-field
                       Core+Detail verdicts — gold for diagnosing WHERE a
                       retrieval missed
                     • evidence_prf: set-overlap P/R/F1 of the answer's cited
                       app_log_ids vs gold evidence""",
        "relevant": "    DynamicMem:    relevant_app_logs    log entries the gold evidence ids point to",
        "shape": """## DynamicMem (`app_logs`-based, official TCE v2 checkpoint protocol)
  Phase 1 chunk:    recorder.init = {"app_logs": List[dict]}
                    each app_log has: app_log_id, timestamp, app_name,
                    api_name, request, response
                    CHECKPOINT-INTERLEAVED: each user's ~1500-log stream is
                    ingested in 5 chronological checkpoint segments; after
                    each segment, that checkpoint's queries run against the
                    CURRENT memory state. Never assume the stream is
                    complete; user states DRIFT over time (a habit at cp1
                    may change by cp5), so retrieval must reflect the
                    latest ingested state, not the earliest match.
  Phase 2 retrieve: recorder.init = {"app_logs": List[dict], "query": str}
                    `app_logs` is the prefix visible at the current
                    checkpoint. `query` is a TCE task query — either a
                    state-completion template ("Infer the user's current
                    state for ... using this template: {...}") or a
                    personalized-service scenario ("[Scenario]...[Task
                    Instruction]..." possibly with a [Required Output
                    Object] JSON to fill). Answers are judged field-by-field
                    (0-1) against golden states / reference outputs, plus
                    evidence P/R/F1 on cited app_log_ids — so retrieval
                    should surface the SOURCE LOGS (with their app_log_id)
                    that ground each answer.
                    Return-dict tip: {"inline_memory_blocks": [str, ...]}
                    renders each block verbatim into the official answer
                    prompt's [Memory] section (blocks joined by "<->");
                    any other dict shape is serialized as one JSON block.
""",
        "dispatch_check": '"app_logs" in init',
        "dispatch_update_comment": "# DynamicMem Phase 1 (one checkpoint segment)",
        "dispatch_retrieve_comment": "# DynamicMem Phase 2 (TCE task query)",
    },
    "locomo": {
        "display_name": "LoCoMo",
        "qa_metadata": """    LoCoMo:        {category, evidence}
                     • category 1-4 = factual questions
                     • category 5   = adversarial ("Not mentioned"; tests
                                       hallucination resistance)
                     • evidence     = list of dia_ids (e.g. "D1:9")""",
        "relevant": "    LoCoMo:        relevant_turns       conversation turns evidence resolves to",
        "shape": """## LoCoMo (multi-session two-person conversation)
  Phase 1 chunk:    recorder.init = {"conversation": dict}
                    conversation has keys: speaker_a (str), speaker_b (str),
                    session_1..session_N (List[turn_dict]),
                    session_N_date_time (str)
                    each turn_dict has: speaker, dia_id (e.g. "D1:3"), text
                    (NOTE: only `conversation` is provided — NO summaries,
                     observations, or event_summary.)
  Phase 2 retrieve: recorder.init = {"conversation": dict, "query": str}
""",
        "dispatch_check": '"conversation" in init',
        "dispatch_update_comment": "# LoCoMo Phase 1",
        "dispatch_retrieve_comment": "# LoCoMo Phase 2",
    },
    "longmemeval": {
        "display_name": "LongMemEval",
        "qa_metadata": """    LongMemEval:   {question_type, question_date, answer_session_ids}
                     • question_type ∈ {single-session-user,
                                         single-session-assistant,
                                         multi-session,
                                         temporal-reasoning,
                                         knowledge-update}""",
        "relevant": "    LongMemEval:   relevant_context     sessions answer_session_ids point to",
        "shape": """## LongMemEval  (haystack of chat sessions with one question each)
`longmemeval_s` — ~48 sessions/sample, ~120k tokens per haystack.

  Phase 1 chunk:    recorder.init = {"sessions": List[session_dict]}
                    each session_dict has:
                      session_id (str, "session_001", "session_002", ... —
                                  positional, in haystack order),
                      date (str, e.g. "2023/05/20 (Sat) 02:21"),
                      messages (List[{"role": "user"|"assistant", "content": str}])
                    The session_id is a positional label and carries NO
                    gold/distractor signal (gold sessions sit uniformly across
                    the haystack), so there is nothing to key off. Upstream's
                    own reader labels sessions the same way. Retrieval has to
                    be content-based; that is the whole task.
  Phase 2 retrieve: recorder.init = {"sessions": List[session_dict],
                                     "query": str,
                                     "question_date": str   # "YYYY/MM/DD ..." }
                    The question_date is the user's reference time at ask-
                    time — critical for temporal-reasoning and knowledge-
                    update questions.
""",
        "dispatch_check": '"sessions" in init',
        "dispatch_update_comment": "# LongMemEval Phase 1 (list of session dicts)",
        "dispatch_retrieve_comment": "# LongMemEval Phase 2\n        # init[\"question_date\"] is the user's reference time",
    },
}

DATASET_RENDER_ORDER = ["dynamicmem", "locomo", "longmemeval"]
