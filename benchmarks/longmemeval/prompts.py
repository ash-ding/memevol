"""Prompt builders for the LongMemEval benchmark — both QA agent and judge.

The judge has FOUR variants per question_type, lifted from Figure 10 of the
LongMemEval paper. Adaptations applied:
  - Field names normalized to {query}/{reference}/{prediction}
    (paper used: question / correct answer / model response).
  - Output shape normalized to {"score": 0|1, "reason": "..."} so it matches
    the unified Judge JSON contract (paper asks for "yes"/"no" labels;
    semantically yes=1, no=0).
  - Question-type strings match the actual data in
    longmemeval_s_cleaned.json:
      paper "temp-reasoning"            → data "temporal-reasoning"
      paper "knowledge-update"          → data "knowledge-update"
      paper "single-session-preference" → data "single-session-preference"
      paper "Other question types"      → everything else (multi-session,
                                          single-session-user, single-
                                          session-assistant)
"""

import json
from typing import Dict


# --- QA prompt -------------------------------------------------------------
# Compromise between our prior 2-message system+user format and the official
# LongMemEval repo's single-message format
# (https://github.com/xiaowu0162/LongMemEval/blob/main/src/generation/run_generation.py).
# Adopted from official: single user message, `Current Date` adjacent to the
# question, `Answer:` trigger at the end. Kept from ours: one short concision
# instruction (helps the binary judge match yes/no), retrieved memory dumped
# as JSON (preserves structured metadata our memo backends produce).

LONGMEMEVAL_QA_TEMPLATE = """\
I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer concisely (a short phrase or 1 sentence) using exact wording from the chats when possible.

History Chats:
{history}

Current Date: {question_date}
Question: {query}
Answer:"""


def get_longmemeval_prompt(
    query: str,
    memory_retrived: Dict = {},
    question_date: str = "",
    **kwargs,
):
    """Build the QA prompt for the LongMemEval QA agent.

    Returns a 2-message list with an empty system message and the full prompt
    in the user message (the wire format BaseWorkflow expects). This keeps
    the official repo's single-message intent while staying compatible with
    `common/workflow.py`'s system+user split.

    Returns:
        [{'role': 'system', 'content': ''}, {'role': 'user', 'content': <full prompt>}]
    """
    if memory_retrived:
        history = json.dumps(memory_retrived, indent=2, ensure_ascii=False)
    else:
        history = "(no relevant chat history retrieved)"

    user_message = LONGMEMEVAL_QA_TEMPLATE.format(
        history=history,
        question_date=question_date or "(unknown)",
        query=query,
    )

    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_message},
    ]


# ----- LLM-as-judge prompts (per question_type) ----------------------------
# All four end with the same JSON-output instruction so Judge.score() can
# parse them uniformly.

_OUTPUT_INSTRUCTION = """

Question: {query}
Correct answer: {reference}
Model response: {prediction}

Decide yes or no. Output ONLY a JSON object:
{{"reason": "<one sentence explanation>", "score": <1 if yes else 0>}}
"""


LONGMEMEVAL_JUDGE_TEMPORAL_REASONING = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still "
    "correct."
    + _OUTPUT_INSTRUCTION
)


LONGMEMEVAL_JUDGE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with "
    "an updated answer, the response should be considered as correct as long "
    "as the updated answer is the required answer."
    + _OUTPUT_INSTRUCTION
)


LONGMEMEVAL_JUDGE_SINGLE_SESSION_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly."
    + _OUTPUT_INSTRUCTION
)


LONGMEMEVAL_JUDGE_OTHER = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no."
    + _OUTPUT_INSTRUCTION
)


# Dispatch table: question_type → prompt template. Unknown types fall back
# to LONGMEMEVAL_JUDGE_OTHER (the paper's "Other question types" rubric).
LONGMEMEVAL_JUDGE_PROMPTS: Dict[str, str] = {
    "temporal-reasoning":         LONGMEMEVAL_JUDGE_TEMPORAL_REASONING,
    "knowledge-update":           LONGMEMEVAL_JUDGE_KNOWLEDGE_UPDATE,
    "single-session-preference":  LONGMEMEVAL_JUDGE_SINGLE_SESSION_PREFERENCE,
    # multi-session, single-session-user, single-session-assistant → "Other"
}
