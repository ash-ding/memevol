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
# LoCoMo has 5 question categories (per the original paper). The IREM_impl
# A-mem baseline (test_advanced.py:140-181) uses 4 distinct user-message
# variants: cat 2 (temporal/date), cat 3 (open-domain phrase), cat 5
# (adversarial), and a default for cat 1 / 4. We follow that convention,
# adapted to our two-message wire format (empty system + full user).
#
# Cat 5 is adversarial: the gold answer is empty (the question's answer is
# "Not mentioned in the conversation"). Per LoCoMo / A-mem, the QA prompt for
# cat 5 offers two options — the gold answer (which is "Not mentioned ...")
# and a distractor, in random order — and asks the model to pick. This is
# half-cheating but it's how the original benchmark is run.

import random


_LOCOMO_USER_DEFAULT = """\
Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {query} Short answer:"""


_LOCOMO_USER_CAT2 = """\
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date. Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {query} Short answer:"""


_LOCOMO_USER_CAT3 = """\
Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {query} Short answer:"""


_LOCOMO_USER_CAT5_TEMPLATE = """\
Based on the context: {context}, answer the following question. {query}

Select the correct answer: {opt_a} or {opt_b}  Short answer:"""


def _format_context(memory_retrived: Dict) -> str:
    if not memory_retrived:
        return "(no relevant context retrieved)"
    return json.dumps(memory_retrived, indent=2, ensure_ascii=False)


def get_locomo_prompt(
    query: str,
    memory_retrived: Dict = {},
    category: int = 0,
    reference: str = "",
    rng: random.Random = None,
    **kwargs,
):
    """Build the per-category QA prompt for the LoCoMo QA agent.

    Returns 2 messages: empty system + full user (matches BaseWorkflow's
    expected wire shape, but the entire prompt lives in the user message
    per the IREM A-mem baseline).

    `reference` is needed only for cat 5 (adversarial) — the model is shown
    the gold answer alongside a distractor and asked which is correct.
    """
    context = _format_context(memory_retrived)

    if category == 2:
        user_message = _LOCOMO_USER_CAT2.format(context=context, query=query)
    elif category == 3:
        user_message = _LOCOMO_USER_CAT3.format(context=context, query=query)
    elif category == 5:
        # Adversarial: present gold + distractor in random order.
        # Gold answer for cat 5 is typically empty / "Not mentioned"; we
        # use a fixed canned distractor matching A-mem's design.
        rng = rng or random.Random()
        distractor = "Not mentioned in the conversation"
        gold = (reference or "").strip() or distractor
        # If gold IS the distractor (truly "not mentioned"), still produce
        # two visually distinct options so the question makes sense.
        if gold == distractor:
            opt_a, opt_b = distractor, "An answer found in the conversation"
        elif rng.random() < 0.5:
            opt_a, opt_b = distractor, gold
        else:
            opt_a, opt_b = gold, distractor
        user_message = _LOCOMO_USER_CAT5_TEMPLATE.format(
            context=context, query=query, opt_a=opt_a, opt_b=opt_b,
        )
    else:
        # cat 1, 4, or unknown → default phrase prompt
        user_message = _LOCOMO_USER_DEFAULT.format(context=context, query=query)

    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_message},
    ]
