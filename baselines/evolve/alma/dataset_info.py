"""Per-dataset prompt fragments for the ALMA meta-agent.

Each of the three prompt builders in meta_agent_prompt.py interpolates the
DATASET_INFO[dataset] block for the run's dataset. The `dynamicmem` block is
the exact text that used to be inline in meta_agent_prompt.py — a byte-identical
snapshot test (tests/test_alma_multidataset.py) guarantees the extraction did
not drift, so existing ALMA DynamicMem results stay comparable.
"""
from __future__ import annotations

from typing import Dict

# --- dynamicmem: verbatim extraction (see the source-line map in the plan) ---
_DYNAMICMEM = {
    "task_description": """The evaluation of downstream agent system is based on DynamicMem:
- Each task is a single user's digital life: months of app activity logs spanning apps like Fitbit, banking (Chase), calendar, messaging, and LLM assistants.
- The agent must answer natural-language questions about the user's habits, preferences, and behavioural patterns.
- Questions span life domains: work routines, health habits, financial behaviours, social activities, and leisure preferences.
- Different users have **entirely different lifestyles, occupations, and activity patterns** — memory must be personalised per user.
- The memory structure should strengthen the following abilities of the agent:
    - Personalised Profile Construction: building a structured user profile from raw app-log sequences (recorder.init['app_logs']).
    - Longitudinal Pattern Extraction: identifying recurring behaviours (weekly meetings, daily exercise, monthly financial transfers) from timestamped log entries.
    - Cross-Domain Synthesis: connecting behavioural patterns across different life domains (e.g. work stress → reduced fitness activity).
    - Temporal Precision: storing and retrieving when (day-of-week, time-of-day, frequency) specific behaviours occur.
    - Selective Retrieval: for a given QA question, surfacing only the relevant subset of the user profile rather than the entire log history.
""",  # meta_agent_prompt.py TASK_DESCRIPTION['dynamicmem'], lines 9-20
    "gen_intro": """The agent will be used in the DynamicMem personalisation benchmark.
Your memory structure aims to help a downstream QA agent accurately answer questions about a specific user's habits, preferences, and behavioural patterns.""",  # lines 445-446
    "recorder_env_import": "from datasets.dynamicmem.env import Basic_Recorder",
    "recorder_class_name": "DynamicMemRecorder",
    "gen_protocol": """    == DynamicMem Two-Phase Protocol ==

    Phase 1 — build_memory_from_data(recorder):
      Called N times to build user memory from app logs.
      recorder.init['app_logs']     = List[dict]  — current batch of app log entries
      Each app_log entry has: app_log_id, timestamp, app_name, api_name, request, response.
      Expected behaviour: extract and store user habits, preferences, and behavioural patterns into internal DB.
      NOTE: build_memory_from_data may be called multiple times (once per chunk of logs); your internal DB must be
      persistent across calls (store it as self.* attributes on the MemoStructure instance).

    Phase 2 — retrieve_memory_for_query(recorder):
      Called once per QA question (after Phase 1 is complete).
      recorder.init['query']        = str          — current natural-language question
      recorder.init['app_logs']     = List[dict]   — all app logs (reference only)
      Return value: Dict  — memory context passed directly to the answering agent.
      Expected behaviour: retrieve facts relevant to the query from the stored memory.
      The returned dict is sent verbatim to the agent — keep it clean, structured, and non-redundant.
      IMPORTANT: the downstream QA agent ONLY sees the returned Dict and the question.
      It does NOT have access to app_logs.
    """,  # lines 423-441
    "code_usage": """Your memory structure will be used in the two-phase DynamicMem workflow:
    - `build_memory_from_data(recorder)`: called during Phase 1 (once or multiple times) to ingest app logs
      and build a personalised user profile in the internal database.
    - `retrieve_memory_for_query(recorder)`: called during Phase 2 (once per QA question) to retrieve
      relevant facts from the stored profile. The returned Dict is given directly to the answering agent.
    IMPORTANT: each user gets a fresh MemoStructure() instance — there is NO cross-user memory sharing.
    The same instance persists across all Phase 1 + Phase 2 calls for one user.""",  # lines 468-474
    "design_goals": """### Your Task:
Modify or create code that fully satisfies the following design goals:

1. **Multiple Memory Layers:**
   - Create multiple subclasses of `Sub_memo_layer`.
   - Each layer must have a clear responsibility and maintain its own database (Chroma or NetworkX).
   - Think carefully about what type of behavioural data belongs in each layer
     (e.g., temporal patterns, cross-domain correlations, preference summaries).

2. **General Retrieve/Update Orchestration:**
   - Create a subclass of `MemoStructure` that orchestrates all layers.
   - `build_memory_from_data()` should propagate app-log information to relevant layers.
   - `retrieve_memory_for_query()` should chain layer outputs intelligently for the given query.
     Output from one layer can become input to the next.

3. **Personalisation and Temporal Precision:**
   - Store *when* behaviours occur (day-of-week, time-of-day, frequency), not just *what*.
   - Design retrieval to surface only the facts relevant to the current question.
   - Avoid returning the entire stored profile; be selective and query-aware.

4. **Out-of-the-Box Reasoning:**
   - Think about the semantic flow of information across layers.
   - Avoid hard-coded if/else branches for specific apps or domains.
   - Express generalizable principles: the structure should work for users with very different lifestyles.
   - Keep retrieved memory clean and useful — avoid repetition or truncated meaningful text.

5. **Integration with Utilities:**
   - Use Chroma for semantic similarity search (query-aware retrieval).
   - Use NetworkX for structural/relational patterns across apps or time slots.
   - Use Agent for LLM-based summarisation or pattern extraction if needed.

6. **Code Quality:**
   - Output clean, runnable Python code following PEP8.
   - `retrieve_memory_for_query()` and `build_memory_from_data()` accept a `Basic_Recorder` and orchestrate end-to-end.
   - Initialize all layers in `MemoStructure.__init__`.
   - Avoid placeholders like `pass` or `# TODO`.
   - Do not overuse defensive programming; raise exceptions for unexpected conditions.

The goal is to build a memory system that accurately captures individual user behaviour from app logs
and retrieves the right facts for any personalisation question.

### Important:
- Think creatively about data flow — outputs of one layer can feed into the next.
- Provide **only the final rewritten code**, no explanations.""",  # lines 490-533
    "reflection_protocol": """    == DynamicMem Two-Phase Protocol ==
    Phase 1 — build_memory_from_data(recorder):
      recorder.init['app_logs']     = List[dict]  — current batch of app log entries
    Phase 2 — retrieve_memory_for_query(recorder):
      recorder.init['query']        = str          — current question
      recorder.init['app_logs']     = List[dict]   — all logs (reference)
      Must return a Dict for the downstream agent.
      NOTE: the QA agent only sees the returned Dict + question, NOT app_logs.
    """,  # lines 567-574
    "reflection_code_usage": """Your memory structure will be used in the DynamicMem two-phase workflow:
    - `build_memory_from_data(recorder)`: Phase 1 — ingest app logs and build user profile.
    - `retrieve_memory_for_query(recorder)`: Phase 2 — retrieve facts relevant to the query; return a Dict.
    IMPORTANT: each user gets a fresh MemoStructure() instance — no cross-user sharing.""",  # lines 590-593
    "analysis_protocol": """    - Two-phase protocol:
        - `build_memory_from_data(recorder)`: called during Phase 1 with chunks of app logs.
          `recorder.init['app_logs']` = current batch of log entries.
          Expected behaviour: extract and store user habits, preferences, and behavioural patterns.
        - `retrieve_memory_for_query(recorder)`: called during Phase 2 before each QA question.
          `recorder.init['query']` = current question; `recorder.init['app_logs']` = all logs (reference).
          Returns Dict passed to the answering agent.
          Expected behaviour: retrieve facts relevant to the query from the stored memory.
          NOTE: the downstream QA agent only sees the returned Dict and the question — it does NOT
          have access to app_logs directly.""",  # lines 279-288
    "analysis_shape_a": """    **Shape A — sampled QA pair (the usual case):**
        - `user_id`: which user this QA pair belongs to (e.g. "user_001"). Examples are sampled from one randomly-chosen valid user, binned by judge score.
        - `query`: the natural-language question asked
        - `retrieved_memory`: the dict returned by `retrieve_memory_for_query` for this question
        - `predicted`: the agent's answer
        - `reference`: the ground-truth answer
        - `score`: float 0.0–1.0, rated by the official DynamicMem TCE holistic judge (per-field 0.8·core_correct + 0.2·detail_quality/2, averaged across the item's fields; missing fields score 0). Interpretation:
            - 0.0–0.2: mostly or completely incorrect — core facts wrong or missing for nearly all fields.
            - 0.3–0.7: partially correct — a subset of fields has the core fact right; details and/or the remaining fields are wrong or missing.
            - 0.8–0.9: all (or nearly all) core facts correct, with imperfect detail quality.
            - 1.0: fully correct on every field, core and details.
        - `judge_reason`: the LLM judge's brief explanation of why it gave that score — highlights which key points were hit or missed
        - `relevant_app_logs`: the specific app log entries that contain the ground-truth evidence for this question

    Analyze failure patterns both globally and per-user — some failures may be user-specific (e.g. a user with unusual data patterns) while others are systematic. Compare `relevant_app_logs` against `retrieved_memory` to identify what the memory should have stored but didn't. Use `judge_reason` to understand exactly which key points were missing or incorrect. High-score examples are provided as contrast — they show what the memory does well; avoid breaking these.""",  # lines 292-306 (line 291 blank / line 290 mixed-list intro stay in the builder template)
    "evidence_key": "relevant_app_logs",
}

# --- locomo: authored ---
_LOCOMO = {
    "task_description": """The evaluation of the downstream agent system is based on LoCoMo:
- Each task is a very long multi-session conversation between two speakers, unfolding over many dated sessions.
- The agent must answer natural-language questions about what was said, decided, or changed across those sessions.
- Questions span single-hop factual recall, multi-hop reasoning across sessions, temporal reasoning (when something happened, ordering of events), and knowledge updates (a later session overrides an earlier fact).
- Each conversation is a different pair of speakers with different life events — memory must be grounded in THIS conversation, not general world knowledge.
- The memory structure should strengthen the following abilities of the agent:
    - Conversational Fact Extraction: capturing who said what, about whom, in which session (recorder.init['conversation']).
    - Cross-Session Linking: connecting facts mentioned in different sessions about the same entity, plan, or event.
    - Temporal Grounding: attaching each fact to its session date/time so ordering and "when" questions can be answered.
    - Knowledge-Update Tracking: recognising when a later session supersedes an earlier statement, and preferring the latest state.
    - Selective Retrieval: for a given QA question, surfacing only the relevant turns rather than the whole conversation.
""",
    "gen_intro": "The agent will be used in the LoCoMo conversational-memory benchmark.\nYour memory structure aims to help a downstream QA agent accurately answer questions about a two-speaker, multi-session conversation.",
    "recorder_env_import": "from datasets.locomo.env import Basic_Recorder",
    "recorder_class_name": "LoCoMoRecorder",
    "gen_protocol": """    == LoCoMo Two-Phase Protocol ==

    Phase 1 — build_memory_from_data(recorder):
      Called N times to build memory from the conversation.
      recorder.init['conversation'] = dict — a (subset of the) multi-session conversation. It has keys:
        speaker_a (str), speaker_b (str),
        session_1 .. session_N (each a List[turn_dict]),
        session_N_date_time (str, the session's timestamp).
        Each turn_dict has: speaker (str), dia_id (str, e.g. "D1:3"), text (str).
      (Only `conversation` is provided — there are NO summaries, observations, or event_summary.)
      Expected behaviour: extract and store conversational facts, their speaker, and their session date into an internal DB.
      NOTE: build_memory_from_data may be called multiple times (once per chunk of sessions); your internal DB must be
      persistent across calls (store it as self.* attributes on the MemoStructure instance).

    Phase 2 — retrieve_memory_for_query(recorder):
      Called once per QA question (after Phase 1 is complete).
      recorder.init['query']        = str   — current natural-language question
      recorder.init['conversation'] = dict  — the full conversation (reference only)
      Return value: Dict  — memory context passed directly to the answering agent.
      Expected behaviour: retrieve turns/facts relevant to the query, preferring the latest state when facts were updated.
      The returned dict is sent verbatim to the agent — keep it clean, structured, and non-redundant.
      IMPORTANT: the downstream QA agent ONLY sees the returned Dict and the question. It does NOT have access to the conversation.
    """,
    "code_usage": """Your memory structure will be used in the two-phase LoCoMo workflow:
    - `build_memory_from_data(recorder)`: called during Phase 1 (once or multiple times) to ingest conversation sessions
      and build a structured memory of the dialogue.
    - `retrieve_memory_for_query(recorder)`: called during Phase 2 (once per QA question) to retrieve
      relevant turns/facts. The returned Dict is given directly to the answering agent.
    IMPORTANT: each conversation gets a fresh MemoStructure() instance — there is NO cross-conversation memory sharing.
    The same instance persists across all Phase 1 + Phase 2 calls for one conversation.""",
    "design_goals": """### Your Task:
Modify or create code that fully satisfies the following design goals:

1. **Multiple Memory Layers:**
   - Create multiple subclasses of `Sub_memo_layer`.
   - Each layer must have a clear responsibility and maintain its own database (Chroma or NetworkX).
   - Think carefully about what type of conversational data belongs in each layer
     (e.g., per-speaker facts, entity/event links, temporal index of sessions).

2. **General Retrieve/Update Orchestration:**
   - Create a subclass of `MemoStructure` that orchestrates all layers.
   - `build_memory_from_data()` should propagate conversation-turn information to relevant layers.
   - `retrieve_memory_for_query()` should chain layer outputs intelligently for the given query.
     Output from one layer can become input to the next.

3. **Temporal Grounding and Knowledge Updates:**
   - Store *when* each fact was said (session date/time) so ordering and "when" questions can be answered.
   - When a later session updates an earlier fact, prefer the latest state at retrieval time.
   - Design retrieval to surface only the turns relevant to the current question.
   - Avoid returning the entire conversation; be selective and query-aware.

4. **Out-of-the-Box Reasoning:**
   - Think about the semantic flow of information across layers.
   - Avoid hard-coded if/else branches for specific speakers or topics.
   - Express generalizable principles: the structure should work for very different conversations.
   - Keep retrieved memory clean and useful — avoid repetition or truncated meaningful text.

5. **Integration with Utilities:**
   - Use Chroma for semantic similarity search (query-aware retrieval).
   - Use NetworkX for structural/relational patterns across speakers, entities, or sessions.
   - Use Agent for LLM-based summarisation or fact extraction if needed.

6. **Code Quality:**
   - Output clean, runnable Python code following PEP8.
   - `retrieve_memory_for_query()` and `build_memory_from_data()` accept a `Basic_Recorder` and orchestrate end-to-end.
   - Initialize all layers in `MemoStructure.__init__`.
   - Avoid placeholders like `pass` or `# TODO`.
   - Do not overuse defensive programming; raise exceptions for unexpected conditions.

The goal is to build a memory system that accurately captures a multi-session conversation
and retrieves the right turns for any question about it.

### Important:
- Think creatively about data flow — outputs of one layer can feed into the next.
- Provide **only the final rewritten code**, no explanations.""",
    "reflection_protocol": """    == LoCoMo Two-Phase Protocol ==
    Phase 1 — build_memory_from_data(recorder):
      recorder.init['conversation'] = dict  — a conversation subset (speaker_a/b, session_* turn lists, session_*_date_time)
    Phase 2 — retrieve_memory_for_query(recorder):
      recorder.init['query']        = str   — current question
      recorder.init['conversation'] = dict  — full conversation (reference)
      Must return a Dict for the downstream agent.
      NOTE: the QA agent only sees the returned Dict + question, NOT the conversation.
    """,
    "reflection_code_usage": """Your memory structure will be used in the LoCoMo two-phase workflow:
    - `build_memory_from_data(recorder)`: Phase 1 — ingest conversation sessions and build memory.
    - `retrieve_memory_for_query(recorder)`: Phase 2 — retrieve turns relevant to the query; return a Dict.
    IMPORTANT: each conversation gets a fresh MemoStructure() instance — no cross-conversation sharing.""",
    "analysis_protocol": """    - Two-phase protocol:
        - `build_memory_from_data(recorder)`: called during Phase 1 with chunks of conversation sessions.
          `recorder.init['conversation']` = current subset of the multi-session conversation.
          Expected behaviour: extract and store conversational facts, speakers, and session dates.
        - `retrieve_memory_for_query(recorder)`: called during Phase 2 before each QA question.
          `recorder.init['query']` = current question; `recorder.init['conversation']` = full conversation (reference).
          Returns Dict passed to the answering agent.
          Expected behaviour: retrieve turns relevant to the query, preferring the latest state.
          NOTE: the downstream QA agent only sees the returned Dict and the question — it does NOT
          have access to the conversation directly.""",
    "analysis_shape_a": """    **Shape A — sampled QA pair (the usual case):**
        - `user_id`: which conversation this QA pair belongs to. Examples are sampled from one randomly-chosen valid conversation, binned by judge score.
        - `query`: the natural-language question asked
        - `retrieved_memory`: the dict returned by `retrieve_memory_for_query` for this question
        - `predicted`: the agent's answer
        - `reference`: the ground-truth answer
        - `score`: 0 or 1, from the LoCoMo binary judge (1 = correct, 0 = incorrect).
        - `judge_reason`: the LLM judge's brief explanation of why it gave that score — highlights which key points were hit or missed
        - `relevant_turns`: the specific conversation turns that contain the ground-truth evidence for this question

    Analyze failure patterns both globally and per-conversation — some failures may be conversation-specific (e.g. an unusual dialogue) while others are systematic. Compare `relevant_turns` against `retrieved_memory` to identify what the memory should have stored but didn't. Use `judge_reason` to understand exactly which key points were missing or incorrect. Correct (score 1) examples are provided as contrast — they show what the memory does well; avoid breaking these.""",
    "evidence_key": "relevant_turns",
}

# --- longmemeval: authored (shared text for the s / m variants) ---
_LONGMEMEVAL_COMMON = {
    "task_description": """The evaluation of the downstream agent system is based on LongMemEval:
- Each task is a single user's chat history spread across MANY sessions (a "haystack"), most of which are irrelevant distractors.
- The agent must answer ONE natural-language question that depends on information buried in a small number of relevant sessions.
- Questions span single-session recall (user or assistant turns), multi-session synthesis, temporal reasoning (relative to the question's reference date), and knowledge updates (later sessions override earlier facts).
- The `question_date` is the user's reference time at ask-time — critical for temporal-reasoning and knowledge-update questions.
- The memory structure should strengthen the following abilities of the agent:
    - Needle-in-Haystack Retrieval: finding the few relevant sessions among many distractors (recorder.init['sessions']).
    - Content-Based Indexing: keying memory on content, NOT on session_id prefixes (answer_/sharegpt_/ultrachat_ are construction artefacts, not signals).
    - Temporal Grounding: attaching each session's date so "when" and "since when" questions resolve against question_date.
    - Knowledge-Update Tracking: preferring the most recent session when a fact changes over time.
    - Selective Retrieval: for a given question, surfacing only the relevant sessions/messages rather than the whole haystack.
""",
    "gen_intro": "The agent will be used in the LongMemEval long-context QA benchmark.\nYour memory structure aims to help a downstream QA agent accurately answer a question that depends on a few relevant sessions buried in a large chat-session haystack.",
    "recorder_env_import": "from datasets.longmemeval.env import Basic_Recorder",
    "recorder_class_name": "LongMemEvalRecorder",
    "gen_protocol": """    == LongMemEval Two-Phase Protocol ==

    Phase 1 — build_memory_from_data(recorder):
      Called N times to build memory from the session haystack.
      recorder.init['sessions'] = List[dict] — current batch of chat sessions. Each session dict has:
        session_id (str, e.g. "sharegpt_xxx" distractor or "answer_xxx" gold — do NOT key off this prefix),
        date (str, e.g. "2023/05/20 (Sat) 02:21"),
        messages (List[{"role": "user"|"assistant", "content": str}]).
      Expected behaviour: index the informative content of each session (with its date) into an internal DB.
      NOTE: build_memory_from_data may be called multiple times (once per chunk of sessions); your internal DB must be
      persistent across calls (store it as self.* attributes on the MemoStructure instance).

    Phase 2 — retrieve_memory_for_query(recorder):
      Called once per QA question (after Phase 1 is complete).
      recorder.init['query']         = str   — current natural-language question
      recorder.init['sessions']      = List[dict] — all sessions (reference only)
      recorder.init['question_date'] = str   — the user's reference time ("YYYY/MM/DD ..."), use it for temporal reasoning
      Return value: Dict  — memory context passed directly to the answering agent.
      Expected behaviour: retrieve the sessions/messages relevant to the query, preferring the latest state on updates.
      The returned dict is sent verbatim to the agent — keep it clean, structured, and non-redundant.
      IMPORTANT: the downstream QA agent ONLY sees the returned Dict and the question (plus its date). It does NOT have access to the sessions.
    """,
    "code_usage": """Your memory structure will be used in the two-phase LongMemEval workflow:
    - `build_memory_from_data(recorder)`: called during Phase 1 (once or multiple times) to ingest chat sessions
      and build a searchable memory of the haystack.
    - `retrieve_memory_for_query(recorder)`: called during Phase 2 (once per question) to retrieve
      the relevant sessions/messages. The returned Dict is given directly to the answering agent.
    IMPORTANT: each user gets a fresh MemoStructure() instance — there is NO cross-user memory sharing.
    The same instance persists across all Phase 1 + Phase 2 calls for one user.""",
    "design_goals": """### Your Task:
Modify or create code that fully satisfies the following design goals:

1. **Multiple Memory Layers:**
   - Create multiple subclasses of `Sub_memo_layer`.
   - Each layer must have a clear responsibility and maintain its own database (Chroma or NetworkX).
   - Think carefully about what belongs in each layer
     (e.g., per-session content index, temporal index by date, entity/topic links).

2. **General Retrieve/Update Orchestration:**
   - Create a subclass of `MemoStructure` that orchestrates all layers.
   - `build_memory_from_data()` should propagate session content to relevant layers.
   - `retrieve_memory_for_query()` should chain layer outputs intelligently for the given query.
     Output from one layer can become input to the next.

3. **Retrieval Precision and Temporal Grounding:**
   - Index memory on session CONTENT, not on session_id prefixes.
   - Store each session's date so temporal-reasoning questions resolve against `question_date`.
   - When a fact is updated across sessions, prefer the most recent session at retrieval time.
   - Design retrieval to surface only the sessions/messages relevant to the current question; the haystack is large.

4. **Out-of-the-Box Reasoning:**
   - Think about the semantic flow of information across layers.
   - Avoid hard-coded if/else branches for specific topics or session ids.
   - Express generalizable principles: the structure should work across very different haystacks.
   - Keep retrieved memory clean and useful — avoid repetition or truncated meaningful text.

5. **Integration with Utilities:**
   - Use Chroma for semantic similarity search (query-aware retrieval over sessions).
   - Use NetworkX for structural/relational patterns across sessions, entities, or dates.
   - Use Agent for LLM-based summarisation or fact extraction if needed.

6. **Code Quality:**
   - Output clean, runnable Python code following PEP8.
   - `retrieve_memory_for_query()` and `build_memory_from_data()` accept a `Basic_Recorder` and orchestrate end-to-end.
   - Initialize all layers in `MemoStructure.__init__`.
   - Avoid placeholders like `pass` or `# TODO`.
   - Do not overuse defensive programming; raise exceptions for unexpected conditions.

The goal is to build a memory system that finds and retrieves the few relevant sessions
from a large haystack for any question, grounded in time.

### Important:
- Think creatively about data flow — outputs of one layer can feed into the next.
- Provide **only the final rewritten code**, no explanations.""",
    "reflection_protocol": """    == LongMemEval Two-Phase Protocol ==
    Phase 1 — build_memory_from_data(recorder):
      recorder.init['sessions'] = List[dict]  — batch of chat sessions (session_id, date, messages[])
    Phase 2 — retrieve_memory_for_query(recorder):
      recorder.init['query']         = str   — current question
      recorder.init['sessions']      = List[dict]  — all sessions (reference)
      recorder.init['question_date'] = str   — user's reference time
      Must return a Dict for the downstream agent.
      NOTE: the QA agent only sees the returned Dict + question (+ date), NOT the sessions.
    """,
    "reflection_code_usage": """Your memory structure will be used in the LongMemEval two-phase workflow:
    - `build_memory_from_data(recorder)`: Phase 1 — ingest chat sessions and build memory.
    - `retrieve_memory_for_query(recorder)`: Phase 2 — retrieve sessions relevant to the query; return a Dict.
    IMPORTANT: each user gets a fresh MemoStructure() instance — no cross-user sharing.""",
    "analysis_protocol": """    - Two-phase protocol:
        - `build_memory_from_data(recorder)`: called during Phase 1 with chunks of chat sessions.
          `recorder.init['sessions']` = current batch of sessions (session_id, date, messages).
          Expected behaviour: index each session's content and date into internal memory.
        - `retrieve_memory_for_query(recorder)`: called during Phase 2 before each QA question.
          `recorder.init['query']` = current question; `recorder.init['sessions']` = all sessions (reference);
          `recorder.init['question_date']` = user's reference time.
          Returns Dict passed to the answering agent.
          Expected behaviour: retrieve the sessions relevant to the query, grounded in time.
          NOTE: the downstream QA agent only sees the returned Dict and the question — it does NOT
          have access to the sessions directly.""",
    "analysis_shape_a": """    **Shape A — sampled QA pair (the usual case):**
        - `user_id`: which user (haystack) this QA pair belongs to. Examples are sampled from one randomly-chosen valid user, binned by judge score.
        - `query`: the natural-language question asked
        - `retrieved_memory`: the dict returned by `retrieve_memory_for_query` for this question
        - `predicted`: the agent's answer
        - `reference`: the ground-truth answer
        - `score`: 0 or 1, from the LongMemEval binary judge (1 = correct, 0 = incorrect).
        - `judge_reason`: the LLM judge's brief explanation of why it gave that score — highlights which key points were hit or missed
        - `relevant_sessions`: the specific chat sessions that contain the ground-truth evidence for this question

    Analyze failure patterns both globally and per-user — some failures may be haystack-specific while others are systematic. Compare `relevant_sessions` against `retrieved_memory` to identify what the memory should have surfaced but didn't. Use `judge_reason` to understand exactly which key points were missing or incorrect. Correct (score 1) examples are provided as contrast — they show what the memory does well; avoid breaking these.""",
    "evidence_key": "relevant_sessions",
}

DATASET_INFO: Dict[str, Dict[str, str]] = {
    "dynamicmem": _DYNAMICMEM,
    "locomo": _LOCOMO,
    "longmemeval_s": dict(_LONGMEMEVAL_COMMON),
    "longmemeval_m": dict(_LONGMEMEVAL_COMMON),
}
