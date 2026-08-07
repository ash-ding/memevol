"""Tests for the official DynamicMem TCE protocol port.

Covers benchmarks/dynamicmem/tce_prompts.py (prompt builders + scoring math,
ported verbatim from /export/scratch_large/ding/code/DynamicMem) and the
env.py checkpoint loader against the real published data.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_tce_scoring.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

import benchmarks.dynamicmem.tce_prompts as tce


# ---------------- golden normalization (normalize_task_a_current_value) ----------------

def test_golden_normalization_drops_excluded_fields():
    value = {
        "schedule": {"frequency_type": "weekly", "days_of_week": [6], "priority": "high"},
        "timing": {"start_time": "09:30"},
        "schedule_dates": ["2024-01-01"],
    }
    out = tce.normalize_task_a_current_value(value, task_contract_version="taskabc_v2")
    assert out == {
        "schedule": {"frequency_type": "weekly", "days_of_week": [6]},
        "timing": {"start_time": "09:30"},
    }, f"got {out}"


def test_golden_normalization_transition_takes_to():
    value = {"from": {"x": 1}, "to": {"x": 2}}
    out = tce.normalize_task_a_current_value(value, task_contract_version="taskabc_v2")
    assert out == {"x": 2}, f"got {out}"


def test_golden_normalization_empty_returns_none():
    assert tce.normalize_task_a_current_value(None, task_contract_version="taskabc_v2") is None
    assert tce.normalize_task_a_current_value({}, task_contract_version="taskabc_v2") is None
    assert tce.normalize_task_a_current_value({"priority": "high"}, task_contract_version="taskabc_v2") is None


# ---------------- snapshot flatten ----------------

def test_flatten_snapshot_family_colon_key():
    snap = {"habits_state": {"budget_review": {"a": 1}}, "preferences_state": {"diet": {"b": 2}}}
    flat = tce.flatten_snapshot(snap)
    assert flat == {"habits_state:budget_review": {"a": 1}, "preferences_state:diet": {"b": 2}}
    # already-flat passes through
    assert tce.flatten_snapshot(flat) == flat


# ---------------- holistic field generation ----------------

def test_snapshot_holistic_fields_leaf_walk():
    golden = {"schedule": {"days_of_week": [0, 2], "frequency_type": "weekly"}, "location": "trail"}
    predicted = {"schedule": {"frequency_type": "weekly"}}
    fields = tce.snapshot_holistic_fields(golden, predicted)
    by_path = {f["field_path"]: f for f in fields}
    assert set(by_path) == {"schedule.days_of_week", "schedule.frequency_type", "location"}
    assert by_path["schedule.frequency_type"]["predicted_field_value_at_same_path"] == "weekly"
    # missing predicted subpath -> None
    assert by_path["location"]["predicted_field_value_at_same_path"] is None
    assert by_path["schedule.days_of_week"]["golden_field_value"] == [0, 2]


def test_snapshot_holistic_fields_scalar_golden():
    fields = tce.snapshot_holistic_fields("Spotify Premium", "Spotify")
    assert len(fields) == 1
    assert fields[0]["field_path"] == "value"
    assert fields[0]["predicted_field_value_at_same_path"] == "Spotify"


# ---------------- judgment normalization + 0.8/0.2 scoring ----------------

def _fields2():
    return [
        {"field_path": "a", "golden_field_value": 1, "predicted_field_value_at_same_path": 1},
        {"field_path": "b", "golden_field_value": 2, "predicted_field_value_at_same_path": None},
    ]


def test_normalize_holistic_judgments_score_formula():
    out = {"field_judgments": [
        {"field_path": "a", "analysis": "ok", "core_correct": True, "detail_quality": 2},
        {"field_path": "b", "analysis": "partial", "core_correct": True, "detail_quality": 1},
    ]}
    result = tce.normalize_holistic_judgments(out, _fields2())
    assert abs(result["a"]["score_0_1"] - 1.0) < 1e-9
    assert abs(result["b"]["score_0_1"] - 0.9) < 1e-9  # 0.8 + 0.2*(1/2)


def test_normalize_holistic_judgments_missing_field_scores_zero():
    out = {"field_judgments": [
        {"field_path": "a", "analysis": "ok", "core_correct": True, "detail_quality": 2},
    ]}
    result = tce.normalize_holistic_judgments(out, _fields2())
    assert result["b"]["score_0_1"] == 0.0
    assert result["b"]["core_correct"] is False


def test_normalize_holistic_judgments_clamps_and_coerces():
    out = {"field_judgments": [
        {"field_path": "a", "core_correct": "true", "detail_quality": 7},
        {"field_path": "b", "core_correct": False, "detail_quality": 2},
    ]}
    result = tce.normalize_holistic_judgments(out, _fields2())
    assert result["a"]["detail_quality"] == 2 and abs(result["a"]["score_0_1"] - 1.0) < 1e-9
    # core False + detail 2 -> 0.2
    assert abs(result["b"]["score_0_1"] - 0.2) < 1e-9


def test_normalize_holistic_judgments_garbage_output():
    result = tce.normalize_holistic_judgments("not a dict", _fields2())
    assert all(v["score_0_1"] == 0.0 for v in result.values())


# ---------------- apply item scoring + identity gate ----------------

def _judgment(fp, core, detail):
    score = 0.8 * (1.0 if core else 0.0) + 0.2 * (detail / 2.0)
    return {"field_path": fp, "core_correct": core, "detail_quality": detail, "score_0_1": score}


def test_apply_score_identity_gate_fail_zeroes_item():
    item = {"service_family": "user_communication"}
    judgments = [
        _judgment("identity_gate", False, 0),
        _judgment("timing.start_time", True, 2),
    ]
    assert tce.apply_holistic_score_from_judgments(item, judgments) == 0.0


def test_apply_score_identity_gate_pass_means_regulars():
    item = {"service_family": "user_communication"}
    judgments = [
        _judgment("identity_gate", True, 2),
        _judgment("timing.start_time", True, 2),   # 1.0
        _judgment("location", False, 0),           # 0.0
    ]
    assert abs(tce.apply_holistic_score_from_judgments(item, judgments) - 0.5) < 1e-9


def test_apply_score_gate_pass_no_regulars_is_one():
    item = {"service_family": "user_communication"}
    judgments = [_judgment("identity_gate", True, 2)]
    assert tce.apply_holistic_score_from_judgments(item, judgments) == 1.0


def test_apply_score_structured_family_plain_mean():
    item = {"service_family": "action_configuration"}
    judgments = [_judgment("a", True, 2), _judgment("b", False, 0)]
    assert abs(tce.apply_holistic_score_from_judgments(item, judgments) - 0.5) < 1e-9


# ---------------- apply holistic fields ----------------

def test_apply_holistic_fields_user_communication_identity_gate():
    item = {
        "service_family": "user_communication",
        "reference_answer": "ref msg",
        "predicted_answer": "pred msg",
        "scoring_points": [
            {"point_id": "p1", "point_type": "micro", "point_role": "identity_gate",
             "point_text": "The message is about the budget review routine."},
            {"point_id": "p2", "point_type": "micro", "point_text": "Mentions 09:30.",
             "source_field_path": "timing.start_time", "reference_value": "09:30"},
        ],
    }
    fields = tce.apply_holistic_fields(item)
    assert fields[0]["field_path"] == "identity_gate"
    assert fields[1]["field_path"] == "timing.start_time"
    assert fields[1]["criterion"] == "Mentions 09:30."
    assert fields[1]["reference_value"] == "09:30"
    # predicted value for user_communication fields = the whole predicted message
    assert fields[0]["predicted_field_value_at_same_path"] == "pred msg"


def test_apply_holistic_fields_structured_uses_leaf_walk():
    item = {
        "service_family": "action_configuration",
        "reference_output": {"cfg": {"provider": "Tawuniya", "scope": "villa"}},
        "predicted_output": {"cfg": {"provider": "Tawuniya"}},
        "scoring_points": [],
    }
    fields = tce.apply_holistic_fields(item)
    by_path = {f["field_path"]: f for f in fields}
    assert set(by_path) == {"cfg.provider", "cfg.scope"}
    assert by_path["cfg.scope"]["predicted_field_value_at_same_path"] is None


# ---------------- evidence P/R/F1 ----------------

def test_evidence_prf_basic():
    m = tce.evidence_prf(["a", "b", "c"], ["b", "c", "d"])
    assert abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert m["exact"] == 0.0


def test_evidence_prf_empty_conventions():
    both = tce.evidence_prf([], [])
    assert both["recall"] == 1.0 and both["precision"] == 1.0 and both["exact"] == 1.0
    exp_only = tce.evidence_prf(["a"], [])
    assert exp_only["recall"] == 0.0 and exp_only["precision"] == 0.0


# ---------------- JSON output parsing ----------------

def test_parse_json_output_plain_and_fenced():
    assert tce.parse_json_output('{"answer": "x"}') == {"answer": "x"}
    fenced = "```json\n{\"answer\": \"y\"}\n```"
    assert tce.parse_json_output(fenced) == {"answer": "y"}
    assert tce.parse_json_output("no json here") is None


def test_normalize_generation_output_user_state_alias():
    raw = {"user_state": {"k": {"v": 1}}, "evidence": {"k": [{"app_log_id": "log_1", "evidence_content": "s"}]}}
    out = tce.normalize_generation_output(raw, ["k"])
    assert out["snapshot_state"] == {"k": {"v": 1}}
    assert tce.normalize_generation_output("garbage", ["k"]) == {"snapshot_state": {}, "evidence": {}}


def test_align_prediction_to_template():
    template = {"a": {"b": "<fill>"}, "c": "<fill>"}
    pred = {"a": {"b": 1, "extra": 9}, "d": 5}
    aligned = tce.align_prediction_to_template(pred, template)
    assert aligned == {"a": {"b": 1}, "c": None}


# ---------------- prompt builders (structural checks) ----------------

def test_state_completion_prompt_structure():
    prompt = tce.build_state_completion_prompt(
        task_query="Infer the user's current state for habits budget review.",
        state_key="habits_state:budget_review",
        answer_template={"schedule": {"days_of_week": ["<fill the blank>"]}},
        memory_blocks=["{\"app_log_id\": \"log_1\"}", "{\"app_log_id\": \"log_2\"}"],
    )
    assert "[Memory]" in prompt and "[/Memory]" in prompt
    assert "\n<->\n" in prompt                       # official block separator
    assert "Schedule date encoding" in prompt        # habits_state key triggers schedule instruction
    assert '"user_state"' in prompt and '"evidence"' in prompt
    assert "[Question]" in prompt


def test_state_completion_prompt_no_schedule_for_non_habit():
    prompt = tce.build_state_completion_prompt(
        task_query="q", state_key="preferences_state:diet",
        answer_template={"statement": "<fill the blank>"}, memory_blocks=[],
    )
    assert "Schedule date encoding" not in prompt


def test_task_c_prompt_modes():
    structured = tce.build_task_c_prompt(
        task_body="[Scenario]\ns\n\n[Task Instruction]\nt",
        service_family="action_configuration",
        output_template={"cfg": {"x": "<fill>"}},
        memory_blocks=["m1"],
    )
    assert '"answer": {' in structured and "[Assistant Task]" in structured
    text = tce.build_task_c_prompt(
        task_body="[Scenario]\ns\n\n[Task Instruction]\nt",
        service_family="user_communication",
        output_template=None,
        memory_blocks=["m1"],
    )
    assert '"answer": "<specific and complete assistant message>"' in text


def test_snapshot_judge_prompt_structure():
    fields = tce.snapshot_holistic_fields({"timing": {"start_time": "06:30"}}, {"timing": {"start_time": "06:33"}})
    prompt = tce.build_snapshot_holistic_judge_prompt(
        state_key="habits_state:morning_walk",
        golden_state_value={"timing": {"start_time": "06:30"}},
        predicted_state_value={"timing": {"start_time": "06:33"}},
        fields_to_judge=fields,
    )
    assert "core_correct" in prompt and "detail_quality" in prompt
    assert "habit" in prompt.lower()          # habit category guidance selected
    assert "timing.start_time" in prompt


def test_apply_judge_prompt_structure():
    item = {
        "service_family": "user_communication",
        "scenario": "sc", "task_instruction": "ti",
        "reference_answer": "ref", "predicted_answer": "pred",
        "scoring_points": [
            {"point_id": "p1", "point_type": "micro", "point_role": "identity_gate", "point_text": "about routine"},
        ],
    }
    fields = tce.apply_holistic_fields(item)
    prompt = tce.build_apply_holistic_judge_prompt(item=item, fields_to_judge=fields)
    assert "assistant-message" in prompt        # assistant_message guidance selected
    assert "identity_gate" in prompt
    assert '"reference": "ref"' in prompt


# ---------------- memory blocks adapter ----------------

def test_retrieved_to_memory_blocks():
    # convention 1: harness returns explicit blocks
    blocks = tce.retrieved_to_memory_blocks({"inline_memory_blocks": ["b1", "b2"], "other": 1})
    assert blocks == ["b1", "b2"]
    # convention 2: arbitrary dict -> single JSON block
    blocks = tce.retrieved_to_memory_blocks({"facts": ["x"]})
    assert len(blocks) == 1 and '"facts"' in blocks[0]
    assert tce.retrieved_to_memory_blocks({}) == []


# ---------------- env loader on real data ----------------

def test_env_loader_counts_match_real_data():
    from benchmarks.dynamicmem.env import load_user_checkpoints
    user_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "benchmarks", "dynamicmem", "user_data", "001_user_001")
    app_logs, checkpoints = load_user_checkpoints(user_dir)
    assert len(app_logs) == 1455, f"app_logs {len(app_logs)}"
    assert len(checkpoints) == 5, f"checkpoints {len(checkpoints)}"
    # jq-verified: user001 has 189 Task C items total; cp0 has 30 Task A keys and 30 Task C items
    total_c = sum(len([i for i in cp["items"] if i["task_family"] == "apply_service"]) for cp in checkpoints)
    assert total_c == 189, f"task C items {total_c}"
    cp0 = checkpoints[0]
    a0 = [i for i in cp0["items"] if i["task_family"] == "state_completion"]
    c0 = [i for i in cp0["items"] if i["task_family"] == "apply_service"]
    assert len(a0) == 30 and len(c0) == 30, f"cp0 A={len(a0)} C={len(c0)}"
    # as_of prefix visibility fields
    assert cp0["as_of"]["log_index"] == 179
    assert cp0["checkpoint_id"] == "cal_quarterly_001"
    # item shape
    item = a0[0]
    assert item["state_key"] and item["query"] and item["reference"] is not None
    assert isinstance(item["evidence_ids"], list) and item["evidence_ids"]
    assert item["answer_template"] is not None
    c_item = c0[0]
    assert c_item["service_family"] in {"user_communication", "action_configuration", "information_request_construction"}
    assert isinstance(c_item["scoring_points"], list) and c_item["scoring_points"]
    # golden normalization applied to Task A references (no excluded fields)
    def has_excluded(v):
        if isinstance(v, dict):
            return any(str(k).lower() in {"priority", "schedule_date", "schedule_dates"} for k in v) or \
                   any(has_excluded(x) for x in v.values())
        if isinstance(v, list):
            return any(has_excluded(x) for x in v)
        return False
    assert not any(has_excluded(i["reference"]) for i in a0), "excluded fields leaked into Task A golden"


def test_env_sampling_stratified_and_deterministic():
    from benchmarks.dynamicmem.env import load_user_checkpoints, sample_items
    user_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "benchmarks", "dynamicmem", "user_data", "001_user_001")
    _, checkpoints = load_user_checkpoints(user_dir)
    s1 = sample_items(checkpoints, 20, seed="001_user_001")
    s2 = sample_items(checkpoints, 20, seed="001_user_001")
    assert [(i["checkpoint_id"], i["task_family"], i["state_key"], i["qa_id"]) for i in s1] == \
           [(i["checkpoint_id"], i["task_family"], i["state_key"], i["qa_id"]) for i in s2], "not deterministic"
    assert len(s1) == 20
    # stratified: all 5 checkpoints and both families present
    assert len({i["checkpoint_id"] for i in s1}) == 5
    assert {i["task_family"] for i in s1} == {"state_completion", "apply_service"}
    # tiny n must still spread across ALL checkpoints and BOTH families
    s5 = sample_items(checkpoints, 5, seed="001_user_001")
    assert len({i["checkpoint_id"] for i in s5}) == 5, "n=5 must cover 5 checkpoints"
    assert {i["task_family"] for i in s5} == {"state_completion", "apply_service"}
    # None -> all items
    s_all = sample_items(checkpoints, None, seed="x")
    assert len(s_all) == sum(len(cp["items"]) for cp in checkpoints)


def test_env_compat_shim_last_checkpoint_only():
    from benchmarks.dynamicmem.env import load_user_data
    user_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "benchmarks", "dynamicmem", "user_data", "001_user_001")
    app_logs, profile, qa_pairs = load_user_data(user_dir, None)
    assert len(app_logs) == 1455
    assert profile == {}
    # last checkpoint of user001 has 42 Task A keys + 42 Task C items? -> just sanity: flat {query, reference, metadata}
    assert qa_pairs and all(("query" in q and "reference" in q and "metadata" in q) for q in qa_pairs)
    assert all(isinstance(q["reference"], str) for q in qa_pairs), "shim must normalize reference to str"
    # temporal consistency: single (last) checkpoint only
    cps = {q["metadata"]["checkpoint_id"] for q in qa_pairs}
    assert len(cps) == 1


def test_task_c_text_prompt_keeps_upstream_brace_quirk():
    """Audit M9 (user decision: align to official): upstream's text-mode
    Task C instructions block escapes braces for .format() but is never
    formatted, so the OFFICIAL prompt literally shows '{{' / '}}'. Our
    rendered text must match byte-for-byte — verbatim A/B comparability
    wins over fixing the quirk."""
    prompt = tce.build_task_c_prompt(
        task_body="Please draft a message",
        service_family="user_communication",
        output_template=None,
        memory_blocks=["log_00001 | note"],
    )
    # literal doubled braces around the JSON skeleton (the quirk)
    assert "{{\n  \"answer\": \"<specific and complete assistant message>\"" in prompt, (
        "text-mode Task C prompt lost the upstream literal-brace quirk"
    )
    assert "}}" in prompt
    # structured mode: braces are single (that branch IS .format()ed upstream)
    sprompt = tce.build_task_c_prompt(
        task_body="Remind me",
        service_family="schedule_reminder",
        output_template={"title": "<t>"},
        memory_blocks=["log_00001 | note"],
    )
    assert "{{" not in sprompt, "structured Task C prompt must render single braces"


# ---------------- runner ----------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
