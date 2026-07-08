"""Prompt builders for the LoCoMo benchmark — both QA agent and judge."""

import json
from typing import Dict


# --- LLM-as-judge prompt ---------------------------------------------------
# Adapted from the LoCoMo paper / community implementations (verified at
# /export/scratch_large/ding/code/IREM_impl/eval/metrics/llm_judge.py and
# /export/scratch_large/ding/code/LORA_MEM/eval/metrics/llm_judge.py — both
# use the same `ACCURACY_PROMPT`). Two adaptations vs the original:
#   1. Field names normalized to {query} / {reference} / {prediction}
#      (originals used {question} / {gold_answer} / {generated_answer}).
#   2. Output shape normalized to {"score": 0|1, "reason": "..."} so it
#      matches the unified Judge JSON contract (originals used a
#      {"label": "CORRECT"|"WRONG"} schema). Semantics preserved: 1 = CORRECT,
#      0 = WRONG; same generosity rubric (topic-match, time-format-flexibility).

LOCOMO_JUDGE_PROMPT = """Your task is to label an answer to a question as CORRECT (1) or WRONG (0). You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT (1) or WRONG (0).

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT (1).

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT (1). Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {query}
Gold answer: {reference}
Generated answer: {prediction}

First decide CORRECT or WRONG. Output ONLY a JSON object:
{{"reason": "<one sentence explanation>", "score": <1 if CORRECT else 0>}}
"""


# --- QA prompt (per LoCoMo question category) ------------------------------
# LoCoMo has 5 question categories (per the original paper). We run on
# categories 1-4 ONLY (protocol decision 2026-07-08; see env.py::load_user_data
# — cat-5 adversarial QAs carry no gold answer in locomo10.json and were
# judged against an empty reference). Prompt variants follow the IREM_impl
# A-mem baseline (test_advanced.py:140-181): cat 2 (temporal/date), cat 3
# (open-domain phrase), and a default for cat 1 / 4, adapted to our
# two-message wire format (empty system + full user).


_LOCOMO_USER_DEFAULT = """\
Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {query} Short answer:"""


_LOCOMO_USER_CAT2 = """\
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date. Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {query} Short answer:"""


_LOCOMO_USER_CAT3 = """\
Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {query} Short answer:"""


def _format_context(memory_retrived: Dict) -> str:
    if not memory_retrived:
        return "(no relevant context retrieved)"
    return json.dumps(memory_retrived, indent=2, ensure_ascii=False)


def get_locomo_prompt(
    query: str,
    memory_retrived: Dict = {},
    category: int = 0,
    **kwargs,
):
    """Build the per-category QA prompt for the LoCoMo QA agent.

    Returns 2 messages: empty system + full user (matches BaseWorkflow's
    expected wire shape, but the entire prompt lives in the user message
    per the IREM A-mem baseline).

    Only categories 1-4 reach this function (cat-5 adversarial QAs are
    filtered out at load time — see env.py::load_user_data).
    """
    context = _format_context(memory_retrived)

    if category == 2:
        user_message = _LOCOMO_USER_CAT2.format(context=context, query=query)
    elif category == 3:
        user_message = _LOCOMO_USER_CAT3.format(context=context, query=query)
    else:
        # cat 1, 4, or unknown → default phrase prompt
        user_message = _LOCOMO_USER_DEFAULT.format(context=context, query=query)

    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_message},
    ]
