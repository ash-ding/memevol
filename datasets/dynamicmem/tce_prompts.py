"""Official DynamicMem TCE v2 protocol — prompts + scoring, ported verbatim.

Source of truth: /export/scratch_large/ding/code/DynamicMem/ @ 2026-07 (the
published taskabc_v2 / rq_v2 benchmark). Every prompt string and scoring
formula below is a faithful port; do NOT reword them — A/B comparability with
the official benchmark depends on textual identity. Origins:

  Answer prompts:   tce_core/prompts.py   (_build_state_completion_prompt :92,
                    _build_task_c_runtime_prompt :1373, memory section :49)
  Judge prompts:    evaluation/prompts_tce.py  (build_snapshot_holistic_judge_prompt :401,
                    build_apply_holistic_judge_prompt :947, category parts :239/:769)
  Field generation: evaluation/eval_tce.py     (_snapshot_holistic_fields :342,
                    _apply_holistic_fields :803)
  Scoring:          evaluation/eval_tce.py     (score = 0.8*core + 0.2*detail/2 :445,
                    identity gate :711-732)
  Golden shaping:   tce_contracts.py            (normalize_task_a_current_value :114)
  Evidence metric:  tce_core/evaluation.py      (score_evidence :690, set P/R/F1)

Note: the official eval runs the HOLISTIC judge track (Core+Detail). The slot
track (snapshot_point_score / rq3_apply_answer_point_score) is disabled by
default in eval_tce.py and cannot run on the published data for Task A
(state_completion_pack scoring_points are empty on disk). This port therefore
implements the holistic track only.

This module is pure (stdlib only): prompt builders + normalization + scoring.
LLM calls live in datasets/dynamicmem/workflow.py.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants (tce_core/prompts.py:16-33, tce_contracts.py:17-29,
# evaluation/eval_tce.py:45)
# ---------------------------------------------------------------------------

SCHEDULE_DATE_ENCODING_TEXT = (
    "`day_of_week` and `days_of_week` use zero-based weekday indexes: "
    "0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday. "
    "`day_of_month` and `days_of_month` use calendar day numbers 1-31; they are not zero-based. "
    "`week_of_month` uses ordinal week numbers within the month: "
    "1=first, 2=second, 3=third, 4=fourth, 5=fifth."
)

SCHEDULE_FORMAT_SUPPLEMENT = """- Habit schedule format vocabulary:
  - `daily`: every day. Format: `{"frequency_type": "daily"}`
  - `weekly`: every week on the listed weekday(s), such as every Tuesday and Thursday. Format: `{"frequency_type": "weekly", "days_of_week": [0-6 integers, 0=Monday ... 6=Sunday]}`
  - `biweekly`: every two weeks on one listed weekday; `start_date` is the first occurrence and anchors the alternating-week pattern. Format: `{"frequency_type": "biweekly", "days_of_week": [single integer 0-6], "start_date": "YYYY-MM-DD"}`
  - `monthly_by_date`: every month on specific calendar date(s), such as the 1st or 15th of each month. Format: `{"frequency_type": "monthly_by_date", "days_of_month": [1-28 integers]}`
  - `monthly_nth_weekday`: every month on an ordinal weekday, such as the first Monday, third Friday, or last Sunday of the month. Format: `{"frequency_type": "monthly_nth_weekday", "week_of_month": "1-4 or last", "day_of_week": "0-6 integer"}`
  Use these exact vocabulary values when the evidence supports them."""

TASK_A_EXCLUDED_VALUE_FIELDS_V2 = frozenset({"priority", "schedule_date", "schedule_dates"})
_TRANSITION_FIELDS = frozenset({"from", "to"})
POINT_ROLE_IDENTITY_GATE = "identity_gate"

SERVICE_FAMILIES = ("user_communication", "action_configuration", "information_request_construction")


# ---------------------------------------------------------------------------
# Memory section (tce_core/prompts.py:49-56) + memevol retrieved-dict adapter
# ---------------------------------------------------------------------------

def render_inline_memory_section(inline_memory_blocks: Sequence[str]) -> str:
    context = "\n<->\n".join(str(block) for block in inline_memory_blocks if str(block).strip())
    return f"""[Memory]
{context}
[/Memory]"""


def retrieved_to_memory_blocks(retrieved: Any) -> List[str]:
    """memevol adapter: convert a harness's `general_retrieve` dict into the
    official inline-memory block list.

    Convention: if the dict carries `inline_memory_blocks: List[str]`, those
    are used as-is (harnesses that want official-style per-log blocks emit
    this key). Any other non-empty dict is serialized as ONE JSON block.
    """
    if not isinstance(retrieved, dict) or not retrieved:
        return []
    blocks = retrieved.get("inline_memory_blocks")
    if isinstance(blocks, list) and blocks:
        return [str(b) for b in blocks if str(b).strip()]
    return [json.dumps(retrieved, ensure_ascii=False, indent=2, default=str)]


# ---------------------------------------------------------------------------
# Answer prompts
# ---------------------------------------------------------------------------

def _is_habit_state_key(state_key: str) -> bool:
    return str(state_key or "").strip().startswith("habits_state:")


def _state_completion_schedule_instruction(target_keys: List[str]) -> str:
    if not any(_is_habit_state_key(key) for key in target_keys):
        return ""
    return f"- Schedule date encoding: {SCHEDULE_DATE_ENCODING_TEXT}\n{SCHEDULE_FORMAT_SUPPLEMENT}\n"


def build_state_completion_prompt(
    *,
    task_query: str,
    state_key: str,
    answer_template: Any,
    memory_blocks: Sequence[str],
) -> str:
    """Task A answer prompt, single key (the official pipeline runs per-key).

    Port of tce_core/prompts.py::_build_state_completion_prompt (:92-147);
    `answer_template` is the pack item's `answer_template`
    (task_packs.py::build_state_completion_targets_from_pack :1692-1696).
    """
    target_keys = [state_key]
    target_value_templates = {state_key: answer_template}
    user_state_template = {k: target_value_templates.get(k, "<fill the blank>") for k in target_keys}
    evidence_template = {
        k: [{"app_log_id": "<app_log_id>", "evidence_content": "<supporting snippet>"}]
        for k in target_keys
    }
    fill_template = {
        "user_state": user_state_template,
        "evidence": evidence_template,
    }
    fill_template_block = json.dumps(fill_template, ensure_ascii=False, indent=2)
    schedule_instruction = _state_completion_schedule_instruction(target_keys)
    memory_section = render_inline_memory_section(memory_blocks)

    return f"""[Instructions]
- Answer the question below using only the system memory about the user's trajectory provided in [Memory]...[/Memory].
- **Make each answer value as detailed and accurate as the memory supports.** Preserve specific names, times, dates, places, labels, and constraints instead of giving vague summaries.
- Follow the template exactly, and fill every requested field with the most precise supported value.
{schedule_instruction}- `evidence` for each key must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only.
- No extra keys.

[Output format]
{{
  "user_state": {{
    "<key>": "<detailed, accurate value or nested object following the template>",
    "...": "<same structure as template>"
  }},
  "evidence": {{
    "<key>": [
      {{
        "app_log_id": "<app_log_id>",
        "evidence_content": "<supporting snippet from the same log>"
      }}
    ],
    "...": []
  }}
}}

[Question]
{task_query}
[/Question]

{memory_section}

Concrete JSON skeleton to fill:
{fill_template_block}
"""


def build_task_c_prompt(
    *,
    task_body: str,
    service_family: str,
    output_template: Any = None,
    memory_blocks: Sequence[str],
) -> str:
    """Task C answer prompt. Port of tce_core/prompts.py::
    _build_task_c_runtime_prompt (:1373-1434).

    `task_body` is the item's prebuilt `retrieval_query` string (the official
    default answer path uses it verbatim — pipeline.py:1723-1735 → :424).
    response_mode is structured for the two structured families, text for
    user_communication (pipeline.py::_build_apply_prompt_from_queryspec :410).
    """
    response_mode = "text" if str(service_family or "").strip() == "user_communication" else "structured"
    task_body = str(task_body or "").strip()
    memory_section = render_inline_memory_section(memory_blocks)
    if response_mode == "structured":
        instructions = """[Instructions]
- Use [Assistant Task] and the system-maintained memory of the user's trajectory in [Memory]...[/Memory] to complete the assistant task.
- Put the completed task result in `answer`.
- Put the memory evidence supporting that result in `evidence`.
- `evidence` must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only; do not write any text outside the JSON object.

[Output format]
Output JSON ONLY:
{{
  "answer": {output_template},
  "evidence": [
    {{
      "app_log_id": "<app_log_id>",
      "evidence_content": "<supporting snippet from the same log>"
    }}
  ]
}}""".format(
            output_template=json.dumps(output_template, ensure_ascii=False, indent=2),
        )
    else:
        # NOTE the doubled braces below are DELIBERATE and render literally:
        # upstream's text-mode instructions block escapes braces as if it
        # were headed for .format() but is never formatted (only the
        # structured branch is — tce_core/prompts.py:1373-1434), so the
        # official prompt the baselines saw literally shows "{{" / "}}".
        # Verbatim A/B comparability wins over fixing the upstream quirk
        # (protocol decision, 2026-07-08 audit M9).
        instructions = """[Instructions]
- Use [Assistant Task] and the system-maintained memory of the user's trajectory in [Memory]...[/Memory] to complete the assistant task.
- Put the completed task result in `answer`.
- Put the memory evidence supporting that result in `evidence`.
- `evidence` must be a list of objects with:
  - `app_log_id`: use the exact app log id when it can be identified; otherwise use "".
  - `evidence_content`: provide one short supporting snippet or close paraphrase. Keep it local and concise.
- If evidence is unknown, use [].
- Return JSON only; do not write any text outside the JSON object.

[Output format]
Output JSON ONLY:
{{
  "answer": "<specific and complete assistant message>",
  "evidence": [
    {{
      "app_log_id": "<app_log_id>",
      "evidence_content": "<supporting snippet from the same log>"
    }}
  ]
}}"""
    return "{instructions}\n\n[Assistant Task]\n{task_body}\n[/Assistant Task]\n\n{memory_section}\n".format(
        instructions=instructions,
        task_body=task_body,
        memory_section=memory_section,
    )


# ---------------------------------------------------------------------------
# Golden-state shaping (tce_contracts.py:64-131, tce_core/evaluation.py:114-146)
# ---------------------------------------------------------------------------

def has_materialized_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def drop_task_a_excluded_fields_v2(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if str(key).strip().lower() in TASK_A_EXCLUDED_VALUE_FIELDS_V2:
                continue
            out[key] = drop_task_a_excluded_fields_v2(child)
        return out
    if isinstance(value, list):
        return [drop_task_a_excluded_fields_v2(child) for child in value]
    return value


def _prune_empty_containers(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            cleaned_child = _prune_empty_containers(child)
            if has_materialized_value(cleaned_child):
                out[key] = cleaned_child
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            cleaned_child = _prune_empty_containers(child)
            if has_materialized_value(cleaned_child):
                out.append(cleaned_child)
        return out
    return value


def _task_contract_is_v2(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("taskabc_v2")


def normalize_task_a_current_value(value: Any, *, task_contract_version: Any) -> Any:
    """Port of tce_contracts.py::normalize_task_a_current_value (:114-131).
    Returns None when the golden value is empty after normalization — the
    official eval then EXCLUDES the key from the expected set."""
    candidate = copy.deepcopy(value)
    if isinstance(candidate, dict):
        normalized_keys = {str(key).strip().lower() for key in candidate.keys()}
        if normalized_keys and normalized_keys.issubset(_TRANSITION_FIELDS):
            candidate = copy.deepcopy(candidate.get("to"))
    if not has_materialized_value(candidate):
        return None
    if _task_contract_is_v2(task_contract_version):
        candidate = drop_task_a_excluded_fields_v2(candidate)
        candidate = _prune_empty_containers(candidate)
        if not has_materialized_value(candidate):
            return None
    return candidate


def flatten_snapshot(snapshot: Any) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    if any(isinstance(k, str) and ":" in k for k in snapshot.keys()):
        return {str(k): v for k, v in snapshot.items()}
    flat: Dict[str, Any] = {}
    for category, content in snapshot.items():
        if isinstance(content, dict):
            for state_name, value in content.items():
                flat[f"{category}:{state_name}"] = value
    return flat


def flatten_observability(observability: Any) -> Dict[str, Any]:
    return flatten_snapshot(observability)


def align_prediction_to_template(pred_value: Any, template_value: Any) -> Any:
    """Port of tce_core/pipeline.py::align_prediction_to_template (:292-304)."""
    if isinstance(template_value, dict):
        pred_dict = pred_value if isinstance(pred_value, dict) else {}
        return {k: align_prediction_to_template(pred_dict.get(k), v) for k, v in template_value.items()}
    if isinstance(template_value, list):
        pred_list = pred_value if isinstance(pred_value, list) else []
        if not template_value:
            return pred_list
        item_tmpl = template_value[0]
        return [align_prediction_to_template(v, item_tmpl) for v in pred_list]
    return pred_value


# ---------------------------------------------------------------------------
# Model-output parsing / normalization (pipeline.py:432-468, :540-566, :628-659;
# fence fallback per baseline_prediction/common/llm_client.py)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_output(text: Any) -> Optional[Dict[str, Any]]:
    """json.loads with the official ```-fenced-block fallback. Returns None
    when no JSON object can be recovered."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    for candidate in (raw,):
        try:
            out = json.loads(candidate)
            return out if isinstance(out, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
    match = _FENCE_RE.search(raw)
    if match:
        try:
            out = json.loads(match.group(1))
            return out if isinstance(out, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        try:
            out = json.loads(raw[start:end + 1])
            return out if isinstance(out, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def normalize_evidence_prediction(raw_evidence: Any, target_keys: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Port of pipeline.py::normalize_evidence_prediction (:432-468)."""
    def to_records(value: Any) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        out: List[Dict[str, str]] = []
        seen = set()
        for item in value:
            log_id = ""
            evidence_content = ""
            if isinstance(item, dict):
                candidate = item.get("app_log_id")
                if candidate is not None:
                    log_id = str(candidate).strip()
                evidence_content = str(item.get("evidence_content") or "").strip()
            elif isinstance(item, (str, int, float)):
                log_id = str(item).strip()
            dedupe_key = (log_id, evidence_content)
            if dedupe_key in seen:
                continue
            if log_id or evidence_content:
                seen.add(dedupe_key)
                out.append({"app_log_id": log_id, "evidence_content": evidence_content})
        return out

    evidence: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(raw_evidence, dict):
        for key in target_keys:
            evidence[key] = to_records(raw_evidence.get(key))
    else:
        for key in target_keys:
            evidence[key] = []
    return evidence


def normalize_generation_output(raw_out: Any, target_keys: List[str]) -> Dict[str, Any]:
    """Port of pipeline.py::normalize_generation_output (:628-659). Accepts
    `user_state` or `snapshot_state` as the state payload key."""
    if isinstance(raw_out, dict):
        if "user_state" in raw_out or "snapshot_state" in raw_out or "evidence" in raw_out:
            state_payload = raw_out.get("user_state") if "user_state" in raw_out else raw_out.get("snapshot_state", {})
            return {
                "snapshot_state": state_payload,
                "evidence": raw_out.get("evidence", {}),
            }
    return {"snapshot_state": {}, "evidence": {}}


def normalize_rq3_apply_answer_output(raw_out: Any, *, service_family: str = "") -> Dict[str, Any]:
    """Port of pipeline.py::_normalize_rq3_apply_answer_output (:540-566),
    v2 path only."""
    if not isinstance(raw_out, dict):
        return {"answer": "", "evidence": []}
    evidence = normalize_evidence_prediction({"_": raw_out.get("evidence")}, ["_"]).get("_", [])
    if str(service_family or "").strip() == "user_communication":
        answer = raw_out.get("answer")
        if answer is None:
            answer = raw_out.get("final_answer", "")
        return {"answer": str(answer or "").strip(), "evidence": evidence}
    return {"answer": raw_out.get("answer"), "evidence": evidence}


# ---------------------------------------------------------------------------
# Holistic judge — field generation (eval_tce.py:330-360, :793-841)
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _value_at_field_path(value: Any, field_path: str) -> Any:
    if field_path == "value":
        return value
    current = value
    for part in str(field_path or "").split("."):
        if isinstance(current, dict) and part in current:
            current = current.get(part)
            continue
        return None
    return current


def snapshot_holistic_fields(golden_state_value: Any, predicted_state_value: Any) -> List[Dict[str, Any]]:
    """Port of eval_tce.py::_snapshot_holistic_fields (:342-360): one field
    per golden leaf; a missing predicted subpath yields None (judged wrong)."""
    fields: List[Dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict) and value:
            for key in sorted(value.keys()):
                child_path = f"{path}.{key}" if path else str(key)
                visit(value.get(key), child_path)
            return
        fields.append(
            {
                "field_path": path or "value",
                "golden_field_value": _json_safe(value),
                "predicted_field_value_at_same_path": _json_safe(
                    _value_at_field_path(predicted_state_value, path or "value")
                ),
            }
        )

    visit(golden_state_value, "")
    return fields


def apply_holistic_reference_and_prediction(item: Dict[str, Any]) -> Tuple[Any, Any]:
    """Port of eval_tce.py::_apply_holistic_reference_and_prediction (:793-801)."""
    service_family = str((item or {}).get("service_family") or "")
    if service_family == "user_communication":
        return str((item or {}).get("reference_answer") or ""), str((item or {}).get("predicted_answer") or "")
    reference_output = (item or {}).get("reference_output")
    if reference_output is not None:
        return reference_output, (item or {}).get("predicted_output")
    return str((item or {}).get("reference_answer") or ""), str((item or {}).get("predicted_answer") or "")


def apply_holistic_fields(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Port of eval_tce.py::_apply_holistic_fields (:803-841).

    For user_communication the fields come from the item's
    `scoring_points` (= pack `answer_scoring_points`; official code goes
    through _materialize_slots first, but the slot attributes used here map
    1:1 onto the raw points, with predicted = the whole message). For
    structured families, fields are the golden-leaf walk of reference_output.
    """
    reference, predicted = apply_holistic_reference_and_prediction(item)
    service_family = str((item or {}).get("service_family") or "")
    if service_family == "user_communication":
        fields: List[Dict[str, Any]] = []
        for point in list((item or {}).get("scoring_points") or []):
            if not isinstance(point, dict):
                continue
            point_id = str(point.get("point_id") or "").strip()
            if not point_id:
                continue
            point_role = str(point.get("point_role") or "").strip().lower()
            source_field_path = str(point.get("source_field_path") or "").strip()
            target_path = str(point.get("target_path") or "").strip()
            if point_role == POINT_ROLE_IDENTITY_GATE:
                field_path = "identity_gate"
            else:
                field_path = source_field_path or target_path or point_id
            if not field_path:
                continue
            field: Dict[str, Any] = {
                "field_path": field_path,
                "golden_field_value": _json_safe(point.get("reference_value", point.get("point_text", ""))),
                "predicted_field_value_at_same_path": _json_safe(predicted),
            }
            criterion = str(point.get("point_text") or "").strip()
            if criterion:
                field["criterion"] = criterion
            if "reference_value" in point:
                field["reference_value"] = _json_safe(point.get("reference_value"))
            fields.append(field)
        return fields
    if not isinstance(reference, dict):
        return [
            {
                "field_path": "value",
                "golden_field_value": _json_safe(reference),
                "predicted_field_value_at_same_path": _json_safe(predicted),
            }
        ]
    return snapshot_holistic_fields(reference, predicted)


# ---------------------------------------------------------------------------
# Holistic judge — prompt builders (evaluation/prompts_tce.py)
# ---------------------------------------------------------------------------

def _snapshot_holistic_category(state_key: str) -> str:
    normalized = str(state_key or "").strip()
    if normalized.startswith("habits_state:"):
        return "habit"
    if normalized.startswith("preferences_state:"):
        return "preference"
    if normalized.startswith("user_attributes_state:"):
        return "attribute"
    return "profile"


def _snapshot_holistic_category_prompt_parts(category: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Verbatim port of evaluation/prompts_tce.py::_snapshot_holistic_category_prompt_parts (:239-398)."""
    if category == "habit":
        guidance = """- How to identify core vs detail for habit fields
  - Use `state_key` to identify the routine or recurring activity being evaluated, such as a morning walk, board meeting, medication routine, or commute habit
  - For `schedule.*` fields, core is the recurrence pattern or scheduled day/time concept; exact JSON encoding, wording, or weekday index format is detail
  - For `timing.*` fields, core is the intended time window or time point for the same routine; if the reference is an exact time, a prediction within 5 minutes can still count as the same timing core, while larger offsets usually change the practical timing
  - For exact `timing.*` details, use `detail_quality=2` for the exact time, `detail_quality=1` for a time within 10 minutes, and `detail_quality=0` for a larger offset unless the field is explicitly an approximate time window
  - For `location` fields, core is the same venue, route, or place category for the routine; compatible added specificity preserves core, such as "sofa in living room" for "sofa"
  - For `location` details, full address, room, landmark, branch-level specificity, or compatible added specificity can improve detail quality; missing required specificity lowers detail quality
  - For other habit fields, core is the routine component named by `field_path`; qualifiers, exact labels, and formatting are detail
- habit core_correct examples
  - `core_correct=true`: 06:33 for a 06:30 start time is within 5 minutes and still identifies the same routine-time field
  - `core_correct=true`: "every Monday, Wednesday, and Friday" identifies the same scheduled weekdays as [0, 2, 4]
  - `core_correct=true`: "sofa in the living room" preserves the same location core as "sofa" with compatible extra specificity
  - `core_correct=false`: 06:45 for a 06:30 exact start time is too far away for the same timing core
  - `core_correct=false`: "evening run" does not identify the same morning-walk routine field
- habit detail_quality examples
  - `detail_quality=2`: the requested field's exact days, exact time, or full compatible location specificity is present
  - `detail_quality=1`: timing is within 10 minutes but not exact, one scheduled weekday is missing, or the location has the right core but misses required specificity, such as "sofa" for "sofa in living room"
  - `detail_quality=0`: timing differs by more than 10 minutes, or the requested field's detail is missing, unusable, or contradictory"""
        example_input = {
            "state_key": "habits_state:morning_walk",
            "golden": {
                "schedule": {"frequency_type": "weekly", "days_of_week": [0, 2, 4]},
                "timing": {"start_time": "06:30"},
                "location": "lakefront trail",
            },
            "predicted": {
                "schedule": {"frequency_type": "weekly", "days_of_week": "Monday, Wednesday, and Friday"},
                "timing": {"start_time": "06:33"},
                "location": "lakefront trail entrance",
            },
            "fields_to_judge": [
                "schedule.frequency_type",
                "schedule.days_of_week",
                "timing.start_time",
                "location",
            ],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "schedule.frequency_type",
                    "analysis": "The prediction describes a weekly routine.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
                {
                    "field_path": "schedule.days_of_week",
                    "analysis": "Monday, Wednesday, and Friday exactly match weekday indexes 0, 2, and 4.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
                {
                    "field_path": "timing.start_time",
                    "analysis": "The prediction gives 06:33, which is within 5 minutes of 06:30 but not exact.",
                    "core_correct": True,
                    "detail_quality": 1,
                },
                {
                    "field_path": "location",
                    "analysis": "The prediction keeps the lakefront trail location and adds compatible entrance-level specificity.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
            ]
        }
        return guidance, example_input, example_output

    if category == "preference":
        guidance = """- How to identify core vs detail for preference fields
  - Use `state_key` to identify what preference is being evaluated, such as learning format, travel style, investment style, diet, communication, or service choices
  - For `statement`, core is the preference direction plus the main object or constraint: what the user primarily prefers, avoids, prioritizes, or requires
  - Secondary constraints, qualifiers, tradeoffs, scope limits, named examples, intensity, and explicit contrast options are details unless they define the preference itself
  - If the reference says "X over Y", preserving X is usually core; preserving the contrast with Y is usually detail unless Y is the main decision boundary
  - If the reference is mainly an avoidance preference, then the avoided option is core
  - If the prediction only mentions the broad topic but not the user's preference direction, the core is missing
- preference core_correct examples
  - `core_correct=true`: "prefers webinars and written materials" preserves the main preferred learning formats
  - `core_correct=true`: "avoids live conferences" preserves the core when the field is mainly about avoiding live conferences
  - `core_correct=false`: "prefers live conferences" reverses the preference
- preference detail_quality examples
  - `detail_quality=2`: all major qualifiers, constraints, scope limits, and avoided options from the reference are present
  - `detail_quality=1`: one or more details are missing, such as the self-paced qualifier or the avoidance of live conferences
  - `detail_quality=0`: the requested field's preference details are missing, unusable, or contradictory"""
        example_input = {
            "state_key": "preferences_state:learning_format",
            "golden": {
                "statement": "Prefers self-paced webinars and written guides over live conferences for professional learning"
            },
            "predicted": {
                "statement": "The user prefers webinars and written materials for professional learning."
            },
            "fields_to_judge": ["statement"],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "statement",
                    "analysis": "The prediction captures the preference for webinars and written materials, but omits the self-paced requirement and the avoidance of live conferences.",
                    "core_correct": True,
                    "detail_quality": 1,
                }
            ]
        }
        return guidance, example_input, example_output

    if category == "attribute":
        guidance = """- How to identify core vs detail for attribute fields
  - Use `state_key` to identify what stable personal fact is being evaluated, such as a subscription, relationship, role, membership, tool, organization, place, skill, or contribution
  - For `field_path = value`, judge the whole scalar or list attribute entry
  - Core is the main factual entity, relationship, role, object, organization, place, tool, or stable fact named by the requested field
  - Tiers, versions, branch names, addresses, departments, dates, scope, qualifiers, and extra specificity are details unless they change the identity of the fact
  - If the prediction gives a different entity, relationship, role, or object, the core is wrong even if the broad topic overlaps
- attribute core_correct examples
  - `core_correct=true`: "Spotify" identifies the same service as "Spotify Premium", though the tier is missing
  - `core_correct=true`: the same organization is named, even if the prediction omits a department, branch, or address
  - `core_correct=false`: a different service, organization, relationship, or role is given
- attribute detail_quality examples
  - `detail_quality=2`: important qualifiers such as tier, version, role, branch, address, scope, or plan type are present
  - `detail_quality=1`: one or more qualifiers are missing, such as Premium or family-plan details for a subscription
  - `detail_quality=0`: the requested field's attribute details are missing, unusable, or contradictory"""
        example_input = {
            "state_key": "user_attributes_state:music_subscription",
            "golden": "Spotify Premium family plan",
            "predicted": "The user has Spotify.",
            "fields_to_judge": ["value"],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "value",
                    "analysis": "The field path `value` refers to the whole scalar attribute entry. The prediction identifies Spotify as the same subscription service, but omits Premium and family-plan details.",
                    "core_correct": True,
                    "detail_quality": 1,
                }
            ]
        }
        return guidance, example_input, example_output

    guidance = """- How to identify core vs detail for profile fields
  - Use `state_key`, `field_path`, and `golden` to identify the requested field's main personal fact or pattern
  - Core is the requested field's central meaning; details are exact names, values, qualifiers, scope, and constraints"""
    example_input = {
        "state_key": "profile:example",
        "golden": {"value": "Uses a named service for a recurring personal activity"},
        "predicted": {"value": "Uses the same named service"},
        "fields_to_judge": ["value"],
    }
    example_output = {
        "field_judgments": [
            {
                "field_path": "value",
                "analysis": "The prediction captures the main value but omits the recurring personal-activity scope.",
                "core_correct": True,
                "detail_quality": 1,
            }
        ]
    }
    return guidance, example_input, example_output


def build_snapshot_holistic_judge_prompt(
    *,
    state_key: str,
    golden_state_value: Any,
    predicted_state_value: Any,
    fields_to_judge: List[Dict[str, Any]],
) -> str:
    """Verbatim port of evaluation/prompts_tce.py::build_snapshot_holistic_judge_prompt (:401-502)."""
    category = _snapshot_holistic_category(state_key)
    category_guidance, example_input, example_output = _snapshot_holistic_category_prompt_parts(category)
    return """[Task Instruction]
You are evaluating whether a predicted personal profile entry matches a reference profile entry.
Use a Core + Detail field evaluation method: judge each requested field separately, using the full prediction as context.
For every field, first identify the field's core meaning and supporting details, then score core correctness and detail quality separately.

[Definitions]
- state_key
  - a label for the profile entry being evaluated; use it to understand the entry type and topic before deciding what is core
- golden
  - the reference profile entry to evaluate against
- predicted
  - the model's predicted profile entry
- fields_to_judge
  - the exact field paths to judge; return one judgment for each field path
- field_path
  - the requested field identifier; copy it exactly into the output
- core_correct
  - boolean judgment for whether `predicted` captures the requested field's core meaning
- detail_quality
  - integer detail judgment: `2` complete/accurate, `1` partially complete or slightly imprecise, `0` mostly missing/wrong/contradictory
- semantic equivalent
  - the same practical meaning despite harmless wording or formatting differences, such as weekday names versus weekday indexes where 0=Monday and 6=Sunday
{category_guidance}

[Evaluation Method]
For each requested field, use a two-part Core + Detail judgment:
1. Identify the requested field's core meaning from `state_key`, `field_path`, and `golden`.
   - Core is the central value, relationship, direction, routine component, or attribute that must be present for the prediction to be practically the same profile information.
2. Decide `core_correct`.
   - Set `core_correct=true` when the predicted entry captures that core meaning, even if details are incomplete.
   - Set `core_correct=false` when the prediction omits the field, contradicts it, gives a different core value, or is too vague to identify the same field meaning.
3. Decide `detail_quality`.
   - Detail means supporting precision beyond the core, such as exact wording, exact time, exact encoding, qualifiers, constraints, tier/version, branch/address, examples, or scope.
   - Use `detail_quality=2` when key details are complete and accurate.
   - Use `detail_quality=1` when important details are missing, vague, or slightly imprecise.
   - Use `detail_quality=0` when details are mostly missing, wrong, contradictory, or unsupported.
   - If `core_correct=false`, `detail_quality` should normally be `0` unless the prediction contains accurate but non-core details.

[Constraints]
1. Judge only the requested `fields_to_judge`.
2. Return every requested `field_path` exactly once.
3. Judge each requested `field_path` independently, while using the full predicted profile entry as context.
4. Use `detail_quality` only for detail completeness and precision; do not use it to override `core_correct`.
5. Do not require exact wording, JSON field names, key order, or identical formatting when the meaning is semantically equivalent.
6. In every field judgment, write `analysis` before `core_correct` and `detail_quality`.

[Example]
[Example Input]
{example_input}

[Example Output]
{example_output}

[Input/Output Format]
Input:
- `state_key`
- `golden`
- `predicted`
- `fields_to_judge`

Output JSON ONLY:
{{
  "field_judgments": [
    {{
      "field_path": "<field_path>",
      "analysis": "<brief analysis before the labels>",
      "core_correct": "<bool>",
      "detail_quality": "<0|1|2 integer>"
    }}
  ]
}}

[Input]
{actual_input}
""".format(
        category_guidance=category_guidance,
        example_input=json.dumps(example_input, ensure_ascii=False, indent=2),
        example_output=json.dumps(example_output, ensure_ascii=False, indent=2),
        actual_input=json.dumps(
            {
                "state_key": str(state_key or ""),
                "golden": golden_state_value,
                "predicted": predicted_state_value,
                "fields_to_judge": [
                    str(field.get("field_path") or "")
                    for field in fields_to_judge
                    if isinstance(field, dict) and str(field.get("field_path") or "")
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def _apply_holistic_service_type(service_family: str) -> str:
    normalized = str(service_family or "").strip()
    if normalized == "user_communication":
        return "assistant_message"
    if normalized == "action_configuration":
        return "structured_action_setup"
    if normalized == "information_request_construction":
        return "structured_information_request"
    return "personalized_assistant_answer"


def _apply_holistic_prompt_parts(service_type: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Verbatim port of evaluation/prompts_tce.py::_apply_holistic_prompt_parts (:769-915)."""
    if service_type == "assistant_message":
        guidance = """- How to identify core vs detail for assistant-message checklist fields
  - Judge each requested field against the full reference message, full predicted message, and service context
  - Core is the checklist field's main practical requirement, such as the targeted routine, schedule, time, location, or action
  - Details are exact times, dates, locations, constraints, names, reminders, and other grounded specifics
  - If the requested field checks the targeted routine and the predicted message is about a different routine, task, person, or service moment, the core is wrong even if some details overlap
- assistant-message core_correct examples
  - `core_correct=true`: a reminder about the same morning walk preserves the core even if one route detail is missing
  - `core_correct=true`: a field asking for the start time is satisfied when the predicted message includes the correct start time
  - `core_correct=false`: a message about an evening run does not preserve the core of a morning-walk reminder
- assistant-message detail_quality examples
  - `detail_quality=2`: all important timing, location, and action details from the reference are present
  - `detail_quality=1`: one important detail is missing or slightly vague, such as omitting the route while keeping the correct routine and time
  - `detail_quality=0`: details are mostly missing, wrong, contradictory, or unsupported"""
        example_input = {
            "scenario": "It is Monday morning and the assistant is preparing a routine reminder.",
            "task_instruction": "Write the assistant message to send now.",
            "reference": "Your morning walk starts at 06:30 on the lakefront trail, so it is time to head out.",
            "predicted": "Your morning walk starts around 06:30. I am sending the reminder now.",
            "fields_to_judge": [
                {
                    "field_path": "timing.start_time",
                    "criterion": "The message should include the 06:30 start time.",
                    "reference_value": "06:30",
                },
                {
                    "field_path": "location",
                    "criterion": "The message should include the lakefront trail location.",
                    "reference_value": "lakefront trail",
                },
            ],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "timing.start_time",
                    "analysis": "The predicted message includes the 06:30 start time.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
                {
                    "field_path": "location",
                    "analysis": "The predicted message omits the lakefront trail location.",
                    "core_correct": False,
                    "detail_quality": 0,
                },
            ]
        }
        return guidance, example_input, example_output

    if service_type == "structured_action_setup":
        guidance = """- How to identify core vs detail for structured action setup fields
  - Use the scenario and task instruction to understand what real action or setup the assistant is completing
  - For each requested structured field, core is the correct action-relevant entity, choice, relationship, or value for that field
  - Details are tier, plan, address, branch, role, scope, version, notes, dates, and other precision inside the same action setup
  - A broadly useful but different setup value is not core-correct for the requested field
- structured action core_correct examples
  - `core_correct=true`: "Tawuniya" preserves the provider core for "Premium home insurance policy from Tawuniya"
  - `core_correct=true`: the same coverage type is selected even if the tier or scope detail is shorter
  - `core_correct=false`: a different bank, organization, role, or configuration choice is selected
- structured action detail_quality examples
  - `detail_quality=2`: all important qualifiers needed by the field are present
  - `detail_quality=1`: one or more qualifiers are missing, such as Premium tier, full coverage scope, branch, account purpose, or role scope
  - `detail_quality=0`: field details are missing, unusable, contradictory, or unsupported"""
        example_input = {
            "scenario": "The user is updating a property management dashboard. The assistant is filling the insurance section of the asset profile before saving the changes.",
            "task_instruction": "Help the user complete the setup or form fields in this scenario.",
            "reference": {
                "property_insurance_config": {
                    "policy_provider": "Premium home insurance policy from Tawuniya",
                    "coverage_scope": "comprehensive coverage for villa and contents",
                }
            },
            "predicted": {
                "property_insurance_config": {
                    "policy_provider": "Tawuniya",
                    "coverage_scope": "Comprehensive Villa Insurance",
                }
            },
            "fields_to_judge": ["property_insurance_config.policy_provider", "property_insurance_config.coverage_scope"],
        }
        example_output = {
            "field_judgments": [
                {
                    "field_path": "property_insurance_config.policy_provider",
                    "analysis": "The prediction preserves the Tawuniya provider core but omits the premium home-insurance policy qualifier.",
                    "core_correct": True,
                    "detail_quality": 1,
                },
                {
                    "field_path": "property_insurance_config.coverage_scope",
                    "analysis": "The prediction captures comprehensive villa insurance but is less explicit about contents coverage.",
                    "core_correct": True,
                    "detail_quality": 2,
                },
            ]
        }
        return guidance, example_input, example_output

    guidance = """- How to identify core vs detail for structured search/filter fields
  - Use the scenario and task instruction to understand what search, filter, comparison, or planning request the assistant is completing
  - For each requested structured field, core is the correct search/filter intent, preferred option, avoided option, constraint, or desired outcome for that field
  - Details are qualifiers, intensity, named examples, exclusions, scope limits, ordering, and format requirements inside the same information request
  - If the prediction gives the opposite preference direction or a different filter value, the core is wrong
- structured search/filter core_correct examples
  - `core_correct=true`: "home gym" preserves the indoor-environment core for climate-controlled exercise
  - `core_correct=true`: "avoid outdoor activities" preserves the core when the field is specifically an exclusion filter
  - `core_correct=false`: leaving an exclusion blank omits the excluded-environment core
- structured search/filter detail_quality examples
  - `detail_quality=2`: all important qualifiers, exclusions, and scope limits for the field are present
  - `detail_quality=1`: a qualifier or exclusion is missing, such as climate-controlled or outdoor-activity contrast
  - `detail_quality=0`: field details are missing, unusable, contradictory, or unsupported"""
    example_input = {
        "scenario": "The user is exploring local fitness facilities and workout classes. The assistant is configuring search parameters to narrow down the available options.",
        "task_instruction": "Help the user set the search filters in this scenario.",
        "reference": {
            "fitness_search_criteria": {
                "preferred_environment": "climate-controlled indoor exercise environments",
                "excluded_environment": "outdoor activities",
            }
        },
        "predicted": {
            "fitness_search_criteria": {
                "preferred_environment": "Home Gym",
                "excluded_environment": "",
            }
        },
        "fields_to_judge": ["fitness_search_criteria.preferred_environment", "fitness_search_criteria.excluded_environment"],
    }
    example_output = {
        "field_judgments": [
            {
                "field_path": "fitness_search_criteria.preferred_environment",
                "analysis": "The prediction keeps the indoor exercise-environment core, but narrows it to home gyms and omits the climate-controlled qualifier.",
                "core_correct": True,
                "detail_quality": 1,
            },
            {
                "field_path": "fitness_search_criteria.excluded_environment",
                "analysis": "The prediction leaves the excluded environment blank, so it does not exclude outdoor activities.",
                "core_correct": False,
                "detail_quality": 0,
            },
        ]
    }
    return guidance, example_input, example_output


def _apply_holistic_prompt_json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _apply_holistic_render_fields_for_prompt(fields_to_judge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Verbatim port of evaluation/prompts_tce.py::_apply_holistic_render_fields_for_prompt (:926-944)."""
    rendered: List[Dict[str, Any]] = []
    for field in fields_to_judge:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path") or "").strip()
        if not field_path:
            continue
        item: Dict[str, Any] = {"field_path": field_path}
        if field.get("criterion"):
            item["criterion"] = str(field.get("criterion") or "")
        if "reference_value" in field:
            item["reference_value"] = _apply_holistic_prompt_json_safe(field.get("reference_value"))
        elif "golden_field_value" in field:
            item["reference_value"] = _apply_holistic_prompt_json_safe(field.get("golden_field_value"))
        if "predicted_field_value_at_same_path" in field:
            item["predicted_value"] = _apply_holistic_prompt_json_safe(field.get("predicted_field_value_at_same_path"))
        rendered.append(item)
    return rendered


def build_apply_holistic_judge_prompt(
    *,
    item: Dict[str, Any],
    fields_to_judge: List[Dict[str, Any]],
) -> str:
    """Verbatim port of evaluation/prompts_tce.py::build_apply_holistic_judge_prompt (:947-1058).
    `item` carries service_family / scenario / task_instruction /
    reference_answer / predicted_answer / reference_output / predicted_output."""
    service_family = str((item or {}).get("service_family") or "")
    scenario = str((item or {}).get("scenario") or "")
    task_instruction = str((item or {}).get("task_instruction") or "")
    reference_answer = str((item or {}).get("reference_answer") or "")
    predicted_answer = str((item or {}).get("predicted_answer") or "")
    reference_output = (item or {}).get("reference_output")
    predicted_output = (item or {}).get("predicted_output")

    service_type = _apply_holistic_service_type(service_family)
    guidance, example_input, example_output = _apply_holistic_prompt_parts(service_type)
    reference = reference_answer if service_type == "assistant_message" or reference_output is None else reference_output
    predicted = predicted_answer if service_type == "assistant_message" or reference_output is None else predicted_output
    return """[Task Instruction]
You are evaluating whether a predicted assistant response matches the reference response for the same personalized service situation.
Use a Core + Detail field evaluation method for service fields: judge each requested assistant-message or structured-output field separately, using the full service context.
For every field, first identify the field's core service value and supporting service details, then score core correctness and detail quality separately.

[Definitions]
- scenario
  - the concrete service moment or user situation
- task_instruction
  - what the assistant is supposed to complete
- reference
  - the reference assistant message or structured service output
- predicted
  - the model's predicted assistant message or structured service output
- fields_to_judge
  - the exact field paths to judge; return one judgment for each field path
- field_path
  - the requested field identifier; copy it exactly into the output
- core_correct
  - boolean judgment for whether `predicted` captures the requested field's core service value
- detail_quality
  - integer detail judgment: `2` complete/accurate, `1` partially complete or slightly imprecise, `0` mostly missing/wrong/contradictory
- semantic equivalent
  - the same practical service meaning despite harmless wording, formatting, key naming, or ordering differences
{guidance}

[Evaluation Method]
For each requested field, use a two-part Core + Detail judgment:
1. Identify the requested field's core service value from `scenario`, `task_instruction`, `field_path`, `fields_to_judge`, and `reference`.
   - Core is the central service meaning that must be present for the predicted response to be practically useful for the same personalized service moment.
   - Core belongs to the service output field being judged, not to a raw source record.
2. Decide `core_correct`.
   - Set `core_correct=true` when the predicted response captures that core service value, even if details are incomplete.
   - Set `core_correct=false` when the prediction omits the field, contradicts it, gives a different core value, targets a different service moment, or is too vague to identify the same field meaning.
3. Decide `detail_quality`.
   - Detail means service-useful precision beyond the core, such as exact time, date, place, cadence, exact encoding, qualifiers, exclusions, constraints, tier/version, branch/address, examples, or scope.
   - Use `detail_quality=2` when key details are complete and accurate.
   - Use `detail_quality=1` when important details are missing, vague, or slightly imprecise.
   - Use `detail_quality=0` when details are mostly missing, wrong, contradictory, or unsupported.
   - If `core_correct=false`, `detail_quality` should normally be `0` unless the prediction contains accurate but non-core details.

[Constraints]
1. Judge only the requested `fields_to_judge`.
2. Return every requested `field_path` exactly once.
3. Judge each requested `field_path` independently, while using the full predicted response and service context.
4. Use `detail_quality` only for detail completeness and precision; do not use it to override `core_correct`.
5. Do not require exact wording, JSON field names, key order, or identical formatting when the meaning is semantically equivalent.
6. Do not penalize the prediction for not restating source records when the service output is correct, but do penalize missing details needed for the service response to be specific and actionable.
7. In every field judgment, write `analysis` before `core_correct` and `detail_quality`.

[Example]
[Example Input]
{example_input}

[Example Output]
{example_output}

[Input/Output Format]
Input:
- `scenario`
- `task_instruction`
- `reference`
- `predicted`
- `fields_to_judge`

Output JSON ONLY:
{{
  "field_judgments": [
    {{
      "field_path": "<field_path>",
      "analysis": "<brief analysis before the labels>",
      "core_correct": "<bool>",
      "detail_quality": "<0|1|2 integer>"
    }}
  ]
}}

[Input]
{actual_input}
""".format(
        guidance=guidance,
        example_input=json.dumps(example_input, ensure_ascii=False, indent=2),
        example_output=json.dumps(example_output, ensure_ascii=False, indent=2),
        actual_input=json.dumps(
            {
                "scenario": scenario,
                "task_instruction": task_instruction,
                "reference": reference,
                "predicted": predicted,
                "fields_to_judge": _apply_holistic_render_fields_for_prompt(fields_to_judge),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


# ---------------------------------------------------------------------------
# Holistic judge — judgment normalization + scoring
# (eval_tce.py:413-453, :711-732, :950-993)
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize_holistic_judgments(out: Any, fields: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Port of eval_tce.py::_normalize_snapshot_holistic_judgments (:413-453).
    Missing / malformed judgments default to core=False, detail=0, score 0.
    Per-field score = 0.8*core + 0.2*(detail/2)."""
    result: Dict[str, Dict[str, Any]] = {
        str(field.get("field_path") or ""): {
            "analysis": "missing from judge output",
            "core_correct": False,
            "detail_quality": 0,
            "score_0_1": 0.0,
        }
        for field in fields
        if str(field.get("field_path") or "")
    }
    if not isinstance(out, dict):
        return result
    data = out.get("field_judgments")
    if not isinstance(data, list):
        return result
    for item in data:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        if field_path not in result:
            continue
        raw_core = item.get("core_correct")
        if isinstance(raw_core, bool):
            core_correct = raw_core
        else:
            core_correct = str(raw_core).strip().lower() in {"true", "1", "yes"}
        try:
            detail_quality = int(item.get("detail_quality"))
        except Exception:
            detail_quality = 0
        detail_quality = max(0, min(2, detail_quality))
        score = 0.8 * (1.0 if core_correct else 0.0) + 0.2 * (float(detail_quality) / 2.0)
        result[field_path] = {
            "analysis": str(item.get("analysis") or "").strip(),
            "core_correct": core_correct,
            "detail_quality": detail_quality,
            "score_0_1": score,
        }
    return result


def ordered_holistic_judgments(
    fields: Sequence[Dict[str, Any]],
    judgments_by_field_path: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Port of eval_tce.py::_snapshot_holistic_ordered_judgments (:950-974)."""
    ordered: List[Dict[str, Any]] = []
    for field in fields:
        field_path = str(field.get("field_path") or "")
        judgment = judgments_by_field_path.get(field_path) or {
            "analysis": "missing from judge output",
            "core_correct": False,
            "detail_quality": 0,
            "score_0_1": 0.0,
        }
        ordered.append(
            {
                "field_path": field_path,
                "golden_field_value": _json_safe(field.get("golden_field_value")),
                "predicted_field_value_at_same_path": _json_safe(field.get("predicted_field_value_at_same_path")),
                "analysis": str(judgment.get("analysis") or ""),
                "core_correct": bool(judgment.get("core_correct")),
                "detail_quality": int(judgment.get("detail_quality") or 0),
                "score_0_1": float(judgment.get("score_0_1") or 0.0),
            }
        )
    return ordered


def snapshot_holistic_item_score(field_judgments: Sequence[Dict[str, Any]]) -> float:
    """Task A per-key score = mean of field scores (eval_tce.py:985-987)."""
    return _mean([float(j.get("score_0_1") or 0.0) for j in field_judgments])


def apply_holistic_score_from_judgments(
    item: Dict[str, Any],
    field_judgments: Sequence[Dict[str, Any]],
) -> float:
    """Port of eval_tce.py::_apply_holistic_score_from_judgments (:711-732).
    user_communication items with an identity_gate field score 0.0 unless
    every gate judgment has core_correct=True."""
    service_family = str((item or {}).get("service_family") or "")
    if service_family == "user_communication":
        identity_gate_judgments: List[Dict[str, Any]] = []
        regular_scores: List[float] = []
        for judgment in field_judgments:
            if not isinstance(judgment, dict):
                continue
            field_path = str(judgment.get("field_path") or "").strip().lower()
            if field_path == POINT_ROLE_IDENTITY_GATE:
                identity_gate_judgments.append(judgment)
            else:
                regular_scores.append(float(judgment.get("score_0_1") or 0.0))
        if identity_gate_judgments:
            gate_pass = all(bool(judgment.get("core_correct")) for judgment in identity_gate_judgments)
            if not gate_pass:
                return 0.0
            return _mean(regular_scores) if regular_scores else 1.0
    return _mean([float(judgment.get("score_0_1") or 0.0) for judgment in field_judgments])


# ---------------------------------------------------------------------------
# Evidence metric (tce_core/evaluation.py::score_evidence :690-723, per-key core)
# ---------------------------------------------------------------------------

def evidence_prf(expected_ids: Sequence[str], predicted_ids: Sequence[str]) -> Dict[str, float]:
    """Set-overlap P/R/F1 (+exact) for one key/item. Official conventions:
    empty expected -> recall 1.0 iff predicted also empty; empty predicted ->
    precision 1.0 iff expected also empty."""
    exp = set(str(i) for i in expected_ids if str(i).strip())
    pred = set(str(i) for i in predicted_ids if str(i).strip())

    if len(exp) == 0:
        recall = 1.0 if len(pred) == 0 else 0.0
    else:
        recall = len(exp & pred) / len(exp)

    if len(pred) == 0:
        precision = 1.0 if len(exp) == 0 else 0.0
    else:
        precision = len(exp & pred) / len(pred)

    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    exact = 1.0 if exp == pred else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "exact": exact}
