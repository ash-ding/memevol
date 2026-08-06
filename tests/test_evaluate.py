"""Tests for staged evaluation: nested sampling + stages config schema.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_evaluate.py          # repo-root uv project

Covers:
  - datasets/dynamicmem/env.py::sample_items_staged (per-checkpoint A/C
    counts, nesting across stages, determinism)
  - datasets/locomo/env.py QA sampling (prefix nesting)
  - datasets/longmemeval/env.py 300/200 stratified split
  - forge/orchestrator.py stages config schema (defaults, validation,
    old-field migration error, wire-spec normalization)
"""
import argparse
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DM_USER = os.path.join(REPO, "datasets", "dynamicmem", "user_data", "001_user_001")


def _item_key(i):
    return (i["checkpoint_id"], i["task_family"], i["state_key"], i["qa_id"])


# ---------------- DynamicMem: sample_items_staged ----------------

def test_dm_staged_counts():
    from datasets.dynamicmem.env import load_user_checkpoints, sample_items_staged
    _, cps = load_user_checkpoints(DM_USER)
    out = sample_items_staged(cps, n_checkpoints=3, n_task_a=5, n_task_c=5, seed=DM_USER)
    assert len(out) == 30, f"expected 3cp*10, got {len(out)}"
    by_cp = {}
    for i in out:
        by_cp.setdefault(i["checkpoint_id"], {"state_completion": 0, "apply_service": 0})
        by_cp[i["checkpoint_id"]][i["task_family"]] += 1
    assert len(by_cp) == 3
    # first 3 checkpoints in order
    assert list(by_cp) == [cp["checkpoint_id"] for cp in cps[:3]]
    for cid, fam_counts in by_cp.items():
        assert fam_counts == {"state_completion": 5, "apply_service": 5}, (cid, fam_counts)


def test_dm_staged_caps_at_available():
    """Requesting more than a bucket holds returns what exists (no crash)."""
    from datasets.dynamicmem.env import load_user_checkpoints, sample_items_staged
    _, cps = load_user_checkpoints(DM_USER)
    out = sample_items_staged(cps, n_checkpoints=1, n_task_a=999, n_task_c=999, seed=DM_USER)
    a = sum(1 for i in out if i["task_family"] == "state_completion")
    c = sum(1 for i in out if i["task_family"] == "apply_service")
    assert a == 30 and c == 30, (a, c)  # user001 cp1 has 30/30 (jq-verified)


def test_dm_staged_nesting():
    """sanity ⊂ stage1 ⊂ stage2 ⊂ stage3 (same seed)."""
    from datasets.dynamicmem.env import load_user_checkpoints, sample_items_staged
    _, cps = load_user_checkpoints(DM_USER)
    sanity = sample_items_staged(cps, n_checkpoints=1, n_task_a=1, n_task_c=1, seed=DM_USER)
    s1 = sample_items_staged(cps, n_checkpoints=1, n_task_a=5, n_task_c=5, seed=DM_USER)
    s2 = sample_items_staged(cps, n_checkpoints=3, n_task_a=5, n_task_c=5, seed=DM_USER)
    s3 = sample_items_staged(cps, n_checkpoints=5, n_task_a=5, n_task_c=5, seed=DM_USER)
    k = lambda items: {_item_key(i) for i in items}
    assert len(sanity) == 2
    assert k(sanity) <= k(s1), "sanity not subset of stage1"
    assert k(s1) <= k(s2), "stage1 not subset of stage2"
    assert k(s2) <= k(s3), "stage2 not subset of stage3"
    assert len(k(s3)) == 50


def test_dm_staged_deterministic():
    from datasets.dynamicmem.env import load_user_checkpoints, sample_items_staged
    _, cps = load_user_checkpoints(DM_USER)
    a = sample_items_staged(cps, n_checkpoints=2, n_task_a=3, n_task_c=3, seed=DM_USER)
    b = sample_items_staged(cps, n_checkpoints=2, n_task_a=3, n_task_c=3, seed=DM_USER)
    assert [_item_key(i) for i in a] == [_item_key(i) for i in b]


# ---------------- LoCoMo: prefix-nested QA sampling ----------------

def test_locomo_qa_nesting():
    from datasets.locomo.env import load_user_data, get_task_list
    conv = get_task_list("search", 1)[0]
    _, _, qa20 = load_user_data(conv, 20)
    _, _, qa40 = load_user_data(conv, 40)
    assert len(qa20) == 20 and len(qa40) == 40
    q = lambda qs: [x["query"] for x in qs]
    assert q(qa20) == q(qa40)[:20], "LoCoMo sampling is not prefix-nested"
    # determinism
    _, _, qa20b = load_user_data(conv, 20)
    assert q(qa20) == q(qa20b)


def test_test_split_honours_sample_cap():
    """Audit M7: get_task_list(status='test') ignored eval_n_samples for
    dynamicmem/locomo, so mode=test stage sizing silently ran the full
    held-out split at every gauntlet tier."""
    from datasets.locomo.env import get_task_list as locomo_tasks
    assert len(locomo_tasks("test", 1)) == 1
    assert len(locomo_tasks("test", 2)) == 2
    assert locomo_tasks("test", 1) == locomo_tasks("test", 2)[:1], "prefix nesting"
    assert len(locomo_tasks("test", 99)) == 4  # cap at available

    from datasets.longmemeval.env import get_task_list as lme_tasks
    assert len(lme_tasks("test", 5)) == 5

    # dynamicmem user_data/ is gitignored — only assert when data is present
    from datasets.dynamicmem.env import get_task_list as dm_tasks
    try:
        full = dm_tasks("test", 99)
    except Exception:
        return  # no local data; the code path is identical to locomo's
    if full:
        assert len(dm_tasks("test", 1)) == 1
        assert dm_tasks("test", 1) == full[:1]


def test_locomo_cat5_excluded():
    """Category-5 adversarial QAs (no gold answer in locomo10.json) are
    excluded at load time — full pool and sampled prefixes alike."""
    from datasets.locomo.env import load_user_data, get_task_list
    for conv in get_task_list("search", 6) + get_task_list("test", 4):
        _, _, qa_all = load_user_data(conv, None)
        cats = {qa["metadata"]["category"] for qa in qa_all}
        assert 5 not in cats, f"{conv}: cat-5 leaked into QA pool"
        assert qa_all, f"{conv}: QA pool empty after filtering"
        # every remaining QA must have a non-empty gold reference
        empty_refs = [qa["query"] for qa in qa_all if not qa["reference"].strip()]
        assert not empty_refs, f"{conv}: {len(empty_refs)} QAs with empty gold"
        # sampled prefix inherits the filter
        _, _, qa20 = load_user_data(conv, 20)
        assert all(qa["metadata"]["category"] != 5 for qa in qa20)


# ---------------- LongMemEval: 300/200 split ----------------

def test_longmemeval_split_300_200():
    from datasets.longmemeval.env import _compute_split, get_task_list
    search_qids, test_qids = _compute_split()
    assert len(search_qids) == 300, f"search={len(search_qids)}"
    assert len(test_qids) == 200, f"test={len(test_qids)}"
    assert not (set(search_qids) & set(test_qids))
    # prefix nesting of the task list
    t20 = get_task_list("search", 20)
    t50 = get_task_list("search", 50)
    assert t20 == t50[:20]


def test_longmemeval_split_stratified():
    import json
    from datasets.longmemeval.env import _compute_split
    data = json.load(open(os.path.join(REPO, "datasets", "longmemeval", "longmemeval_s_cleaned.json")))
    by_type_total = {}
    type_of = {}
    for d in data:
        by_type_total[d["question_type"]] = by_type_total.get(d["question_type"], 0) + 1
        type_of[d["question_id"]] = d["question_type"]
    search_qids, _ = _compute_split()
    by_type_search = {}
    for qid in search_qids:
        by_type_search[type_of[qid]] = by_type_search.get(type_of[qid], 0) + 1
    for t, total in by_type_total.items():
        expected = total * 300 / 500
        got = by_type_search.get(t, 0)
        assert abs(got - expected) <= 2, f"{t}: got {got}, expected ~{expected:.0f}"


# ---------------- Config schema ----------------

def _resolve(yaml_text):
    """Run _resolve_config against a temp YAML with an all-defaults Namespace.

    Always passes --no-strict-config: this file's fixtures are deliberately
    partial (only the stage-schema-relevant keys) — this file tests the
    staged-evaluation config schema, not strict-config completeness (that's
    tests/test_strict_config_forge.py's job)."""
    from forge.orchestrator import _resolve_config, build_arg_parser
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        args = build_arg_parser().parse_args(["--config", path, "--no-strict-config"])
        return _resolve_config(args)
    finally:
        os.unlink(path)


def test_config_defaults_fill():
    cfg = _resolve("datasets:\n  dynamicmem: {}\n  locomo: {}\n  longmemeval_s: {}\n")
    dm = cfg["datasets"]["dynamicmem"]["stages"]
    assert dm["sanity_check"] == {"n_users": 1, "n_checkpoints": 1, "n_task_a": 1, "n_task_c": 1}
    assert dm["stage1"]["n_users"] == 2 and dm["stage1"]["n_checkpoints"] == 1
    assert dm["stage2"]["n_users"] == 4 and dm["stage2"]["n_checkpoints"] == 3
    assert dm["stage3"]["n_users"] == 6 and dm["stage3"]["n_checkpoints"] == 5
    assert "threshold" in dm["stage1"] and "threshold" in dm["stage2"]
    assert "threshold" not in dm["stage3"]
    lc = cfg["datasets"]["locomo"]["stages"]
    assert lc["stage1"] == {"n_conversations": 2, "n_qa": 20, "threshold": lc["stage1"]["threshold"]}
    lme = cfg["datasets"]["longmemeval_s"]["stages"]
    assert lme["stage3"] == {"n_questions": 100}
    # per-dataset judge_model default still applied
    assert cfg["datasets"]["locomo"]["judge_model"] == cfg["judge_model"]


def test_config_family_defaults_for_longmemeval_m():
    cfg = _resolve("datasets:\n  longmemeval_m: {}\n")
    assert cfg["datasets"]["longmemeval_m"]["stages"]["stage1"]["n_questions"] == 20


def test_config_partial_override():
    cfg = _resolve(
        "datasets:\n"
        "  dynamicmem:\n"
        "    stages:\n"
        "      stage1: {threshold: 0.42}\n"
    )
    s1 = cfg["datasets"]["dynamicmem"]["stages"]["stage1"]
    assert s1["threshold"] == 0.42
    assert s1["n_users"] == 2 and s1["n_task_a"] == 5, "defaults not preserved under partial override"


def test_config_old_fields_error():
    try:
        _resolve("datasets:\n  dynamicmem: {eval_n_samples: 6, eval_n_qa: 20}\n")
        assert False, "old fields must raise"
    except ValueError as e:
        assert "stages" in str(e), f"migration hint missing: {e}"


def test_config_bad_threshold_error():
    try:
        _resolve(
            "datasets:\n  dynamicmem:\n    stages:\n      stage1: {threshold: 1.5}\n"
        )
        assert False, "threshold > 1 must raise"
    except ValueError:
        pass


def test_config_non_monotonic_error():
    try:
        _resolve(
            "datasets:\n  dynamicmem:\n    stages:\n"
            "      stage1: {n_users: 5}\n      stage2: {n_users: 2}\n"
        )
        assert False, "shrinking sizes must raise"
    except ValueError:
        pass


def test_config_unknown_stage_field_error():
    try:
        _resolve(
            "datasets:\n  locomo:\n    stages:\n      stage1: {n_users: 3}\n"
        )
        assert False, "n_users on locomo must raise (wrong family field)"
    except ValueError:
        pass


def test_wire_spec_normalization():
    from forge.orchestrator import stage_wire_spec
    dm = stage_wire_spec("dynamicmem", {"n_users": 4, "n_checkpoints": 3, "n_task_a": 5, "n_task_c": 5, "threshold": 0.1})
    assert dm == {"n_samples": 4, "n_checkpoints": 3, "n_task_a": 5, "n_task_c": 5}
    lc = stage_wire_spec("locomo", {"n_conversations": 2, "n_qa": 20, "threshold": 0.3})
    assert lc == {"n_samples": 2, "n_qa": 20}
    lme = stage_wire_spec("longmemeval_m", {"n_questions": 50})
    assert lme == {"n_samples": 50}


def test_config_mode_removed():
    # `mode:` was removed 2026-07-14 — any value must hit the migration guard.
    try:
        _resolve("mode: dev\ndatasets:\n  dynamicmem: {}\n")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("mode: dev accepted")
    # default behavior needs no mode key at all
    cfg = _resolve("datasets:\n  dynamicmem: {}\n")
    assert "mode" not in cfg and cfg["smoke_test"] is False


def test_config_old_status_key_error():
    try:
        _resolve("status: search\ndatasets:\n  dynamicmem: {}\n")
        assert False, "old top-level `status:` must raise a migration error"
    except ValueError as e:
        assert "mode" in str(e), f"migration hint missing: {e}"


def test_stage_plan_order_and_thresholds():
    from forge.orchestrator import stage_plan
    cfg = _resolve("datasets:\n  dynamicmem: {}\n")
    plan = stage_plan("dynamicmem", cfg["datasets"]["dynamicmem"])
    names = [p[0] for p in plan]
    assert names == ["stage1", "stage2", "stage3"]
    # (name, wire_spec, threshold); stage3 threshold None
    assert plan[0][2] is not None and plan[1][2] is not None and plan[2][2] is None


# ---------------- progressive=false: uncapped wire specs ----------------

def test_full_wire_spec_shapes():
    from forge.orchestrator import full_wire_spec
    assert full_wire_spec("dynamicmem") == {
        "n_samples": None, "n_checkpoints": None, "n_task_a": None, "n_task_c": None,
    }
    assert full_wire_spec("locomo") == {"n_samples": None, "n_qa": None}
    assert full_wire_spec("longmemeval_s") == {"n_samples": None}


def test_gauntlet_single_stage_from_single_stage_block():
    """run_gauntlet(progressive=False) now resolves its plan via
    resolve_sampling_plan: progressive=False (progressive=False) sizes ONE
    pass from the required `single_stage` block — replacing the old automatic
    full_wire_spec whole-split."""
    import asyncio
    from common.evaluate import run_gauntlet
    seen = []

    async def run_stage(ds, stage, spec):
        seen.append((stage, spec)); return None

    def read_metrics(ds, stage):
        return {"raw_score": 1.0, "score_max": 1, "tokens": {}}

    cfg = {"locomo": {"single_stage": {"n_conversations": 2, "n_qa": 20}}}
    out = asyncio.run(run_gauntlet(
        datasets_config=cfg, progressive=False, smoke=False,
        sample_seed_for=lambda ds: None, run_stage_fn=run_stage, read_metrics_fn=read_metrics))
    assert [s for s, _ in seen] == ["single"]                       # one pass, no gauntlet
    assert seen[0][1] == {"n_samples": 2, "n_qa": 20}               # sized by single_stage
    assert out["locomo"]["eliminated"] is False
    assert out["locomo"]["stage"] == 4.0                            # "single" -> FULL_STAGE


def test_gauntlet_progressive_false_missing_single_stage_raises():
    """progressive=false (progressive=false) with no `single_stage` block must
    raise — no silent automatic whole-split anymore."""
    import asyncio
    from common.evaluate import run_gauntlet

    async def run_stage(ds, stage, spec): return None

    def read_metrics(ds, stage): return {"raw_score": 1.0, "score_max": 1, "tokens": {}}

    raised = False
    try:
        asyncio.run(run_gauntlet(datasets_config={"locomo": {"stages": {}}}, progressive=False,
            smoke=False, sample_seed_for=lambda ds: None, run_stage_fn=run_stage, read_metrics_fn=read_metrics))
    except ValueError as e:
        raised = "single_stage" in str(e)
    assert raised


def test_get_task_list_none_means_whole_split():
    from datasets.locomo.env import get_task_list as locomo_list
    from datasets.longmemeval.env import get_task_list as lme_list
    assert len(locomo_list("search", None)) == 6
    assert len(locomo_list("test", None)) == 4
    assert len(lme_list("search", None)) == 300
    assert len(lme_list("test", None)) == 200
    # capped behavior unchanged
    assert len(locomo_list("test", 2)) == 2
    assert len(lme_list("test", 50)) == 50
    from datasets.dynamicmem.env import get_task_list as dm_list
    if os.path.isdir(os.path.join(REPO, "datasets", "dynamicmem", "user_data")):
        assert len(dm_list("test", None)) == 4
        assert len(dm_list("search", None)) == 6


def test_locomo_full_qa_is_all_cat14():
    import json as _json
    from datasets.locomo.env import load_user_data, get_task_list
    sid = get_task_list("test", 1)[0]
    _, _, qa = load_user_data(sid, eval_n_qa=None)
    raw = _json.load(open(os.path.join(REPO, "datasets", "locomo", "locomo10.json")))
    sample = next(s for s in raw if s["sample_id"] == sid)
    expected = sum(1 for q in sample["qa"] if q.get("category") != 5)
    assert len(qa) == expected, (len(qa), expected)


def test_dm_full_sampling_and_nesting():
    if not os.path.isdir(DM_USER):
        print("  (dynamicmem user_data missing — skipped)")
        return
    from datasets.dynamicmem.env import load_user_checkpoints, sample_items_staged
    _, cps = load_user_checkpoints(DM_USER)
    full = sample_items_staged(cps, n_checkpoints=None, n_task_a=None,
                               n_task_c=None, seed=DM_USER)
    total = sum(len(cp.get("items", [])) for cp in cps)
    assert len(full) == total, (len(full), total)
    stage3 = sample_items_staged(cps, n_checkpoints=5, n_task_a=5, n_task_c=5,
                                 seed=DM_USER)
    full_keys = {_item_key(i) for i in full}
    assert all(_item_key(i) in full_keys for i in stage3)  # stage3 ⊂ full


# ---------------- mode removal: migration guards + smoke_test ----------------

def test_mode_migration_guards():
    import yaml
    from forge.orchestrator import build_arg_parser, _resolve_config
    hints = {"search": "delete", "dev": "smoke_test", "test": "forge.heldout"}
    with tempfile.TemporaryDirectory() as td:
        for val, hint in hints.items():
            p = os.path.join(td, f"{val}.yaml")
            with open(p, "w") as f:
                yaml.safe_dump({"datasets": {"locomo": {}}, "mode": val}, f)
            try:
                _resolve_config(build_arg_parser().parse_args(["--config", p]))
            except ValueError as exc:
                assert "mode" in str(exc) and hint in str(exc), (val, str(exc))
            else:
                raise AssertionError(f"mode: {val} accepted")


def test_smoke_test_flag():
    # --no-strict-config: fixture is deliberately partial (only `datasets`) —
    # this test is about smoke_test resolution, not strict-config completeness.
    import yaml
    from forge.orchestrator import build_arg_parser, _resolve_config
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.yaml")
        with open(p, "w") as f:
            yaml.safe_dump({"datasets": {"locomo": {}}}, f)
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", p, "--no-strict-config"]))
        assert cfg["smoke_test"] is False
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", p, "--smoke-test", "--no-strict-config"]))
        assert cfg["smoke_test"] is True


def test_smoke_test_forces_single_step():
    """smoke_test (2026-07-16): steps/k_per_step forced to 1 unless the CLI
    explicitly overrides — YAML values must not multiply a smoke run.

    --no-strict-config throughout: fixtures are deliberately partial (only
    the steps/smoke_test-relevant keys) — this test is about smoke_test's
    single-step forcing, not strict-config completeness."""
    import yaml
    from forge.orchestrator import build_arg_parser, _resolve_config
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.yaml")
        with open(p, "w") as f:
            yaml.safe_dump({"steps": 5, "propose": {"k_per_step": 2},
                            "datasets": {"locomo": {}}}, f)
        # --smoke-test overrides YAML steps/k
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", p, "--smoke-test", "--no-strict-config"]))
        assert cfg["steps"] == 1 and cfg["propose"]["k_per_step"] == 1
        # explicit CLI values are respected (each independently)
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", p, "--smoke-test", "--steps", "3", "--no-strict-config"]))
        assert cfg["steps"] == 3 and cfg["propose"]["k_per_step"] == 1
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", p, "--smoke-test", "--k-per-step", "2", "--no-strict-config"]))
        assert cfg["steps"] == 1 and cfg["propose"]["k_per_step"] == 2
        # smoke_test set via YAML forces the same way
        p2 = os.path.join(td, "c2.yaml")
        with open(p2, "w") as f:
            yaml.safe_dump({"steps": 5, "smoke_test": True,
                            "datasets": {"locomo": {}}}, f)
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", p2, "--no-strict-config"]))
        assert cfg["steps"] == 1 and cfg["propose"]["k_per_step"] == 1
        # non-smoke runs keep YAML values untouched
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", p, "--no-strict-config"]))
        assert cfg["steps"] == 5 and cfg["propose"]["k_per_step"] == 2


def test_search_configs_default_smoke_test_false():
    from forge.orchestrator import build_arg_parser, _resolve_config
    for name in ("search_example.yaml",):
        cfg = _resolve_config(build_arg_parser().parse_args(
            ["--config", os.path.join(REPO, "configs", name)]))
        assert cfg["smoke_test"] is False and "mode" not in cfg


# ---------------- stage sizes: null = whole split ----------------

def test_stage_null_wire_equals_full():
    from forge.orchestrator import stage_wire_spec, full_wire_spec, _resolve_dataset_stages
    # null on every stage3 field → wire spec identical to full_wire_spec
    for ds in ("dynamicmem", "locomo", "longmemeval_s"):
        p = {"stages": {"stage3": {}}}
        # seed stage3 with the family's fields set to None
        _resolve_dataset_stages(ds, p)  # fill defaults first
        for f in list(p["stages"]["stage3"]):
            if f != "threshold":
                p["stages"]["stage3"][f] = None
        assert stage_wire_spec(ds, p["stages"]["stage3"]) == full_wire_spec(ds), ds


def test_stage_full_all_aliases_normalize_to_none():
    from forge.orchestrator import _resolve_dataset_stages
    p = {"stages": {"stage3": {"n_conversations": "full", "n_qa": "ALL"}}}
    _resolve_dataset_stages("locomo", p)
    assert p["stages"]["stage3"]["n_conversations"] is None
    assert p["stages"]["stage3"]["n_qa"] is None


def test_stage_null_stage3_is_valid_and_monotonic():
    from forge.orchestrator import _resolve_dataset_stages
    p = {"stages": {
        "stage1": {"n_conversations": 2, "n_qa": 20, "threshold": 0.3},
        "stage2": {"n_conversations": 4, "n_qa": 40, "threshold": 0.35},
        "stage3": {"n_conversations": None, "n_qa": None},
    }}
    _resolve_dataset_stages("locomo", p)  # must not raise
    assert p["stages"]["stage3"]["n_qa"] is None


def test_stage_null_before_concrete_rejected():
    from forge.orchestrator import _resolve_dataset_stages
    # null at stage1 (= full = +inf) followed by a concrete stage2 → decreasing
    p = {"stages": {
        "stage1": {"n_conversations": None, "n_qa": None},
        "stage2": {"n_conversations": 4, "n_qa": 40, "threshold": 0.35},
        "stage3": {"n_conversations": 6, "n_qa": 60},
    }}
    try:
        _resolve_dataset_stages("locomo", p)
    except ValueError as exc:
        assert "non-decreasing" in str(exc)
    else:
        raise AssertionError("null-before-concrete stage sizes accepted")


def test_stage_all_null_valid():
    from forge.orchestrator import _resolve_dataset_stages
    p = {"stages": {
        "stage1": {"n_conversations": None, "n_qa": None, "threshold": 0.3},
        "stage2": {"n_conversations": None, "n_qa": None, "threshold": 0.35},
        "stage3": {"n_conversations": None, "n_qa": None},
    }}
    _resolve_dataset_stages("locomo", p)  # all-full is trivially non-decreasing


def test_stage3_null_sampling_equals_full_dynamicmem():
    """The core guarantee: a stage3-null wire spec samples the SAME item set
    as progressive=false for a real DynamicMem user."""
    if not os.path.isdir(DM_USER):
        print("  (dynamicmem user_data missing — skipped)")
        return
    from datasets.dynamicmem.env import load_user_checkpoints, sample_items_staged
    _, cps = load_user_checkpoints(DM_USER)
    # full via None (what stage3-null produces on the wire)
    staged_full = sample_items_staged(
        cps, n_checkpoints=None, n_task_a=None, n_task_c=None, seed=DM_USER)
    all_items = sample_items_staged(
        cps, n_checkpoints=99999, n_task_a=99999, n_task_c=99999, seed=DM_USER)
    assert {_item_key(i) for i in staged_full} == {_item_key(i) for i in all_items}
    # and stage1 sample ⊂ stage3-null (nesting preserved)
    stage1 = sample_items_staged(cps, n_checkpoints=1, n_task_a=5, n_task_c=5, seed=DM_USER)
    full_keys = {_item_key(i) for i in staged_full}
    assert all(_item_key(i) in full_keys for i in stage1)


def test_build_objectives_no_mean():
    """No cross-benchmark `accuracy` mean; per-dataset axes recorded independently."""
    from forge.orchestrator import _build_objectives
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        hd = os.path.join(td, "h"); os.makedirs(hd)
        open(os.path.join(hd, "harness.py"), "w").write("x = 1\n")
        per_ds = {
            "locomo": {"raw_score": 0.42, "score_max": 1, "stage": 3.0, "tokens": 100},
            "dynamicmem": {"raw_score": 0.31, "score_max": 1, "stage": 3.0, "tokens": 50},
        }
        obj = _build_objectives(per_ds, __import__("pathlib").Path(hd))
    assert "accuracy" not in obj            # NO mean
    assert obj["accuracy_locomo"] == 0.42
    assert obj["accuracy_dynamicmem"] == 0.31
    assert obj["stage_locomo"] == 3.0
    assert obj["tokens_total"] == 150


def test_frontier_is_pure_record_store():
    """Frontier no longer exposes selection methods (proposer self-selects)."""
    from forge.selection import Frontier, Entry
    f = Frontier([Entry(id="1_a", objectives={"accuracy_locomo": 0.5})])
    assert not hasattr(f, "pareto_ids")
    assert not hasattr(f, "sample_parent")
    assert not hasattr(Frontier, "OBJECTIVES")
    # to_dict has no OBJECTIVES header, entries round-trip
    d = f.to_dict()
    assert "objectives" not in d and len(d["entries"]) == 1
    assert Frontier.from_dict(d).get("1_a").objectives == {"accuracy_locomo": 0.5}


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
