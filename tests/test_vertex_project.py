"""Vertex project inference, through every gate that guards it.

Written after a null project_id was rejected by THREE separate checks in turn
— orchestrator config resolution, the proposer preflight, and (correctly) the
resolver itself. Unit-testing the resolver alone had missed all three, because
each gate carried its own copy of the rule.

Run:  uv run python tests/test_vertex_project.py
"""
import json, os, sys, tempfile, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.proposer import (  # noqa: E402
    ProposerLaunchError, _check_environment,
    _resolve_vertex_project, _resolve_vertex_region,
)


def _creds(**fields):
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(fields, fh); fh.close()
    return fh.name


def _clear():
    for k in ("ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION",
              "GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ.pop(k, None)


def test_explicit_project_wins():
    _clear()
    assert _resolve_vertex_project({"project_id": "explicit"}) == "explicit"


def test_env_beats_credentials():
    _clear()
    os.environ["ANTHROPIC_VERTEX_PROJECT_ID"] = "from-env"
    sa = _creds(type="service_account", project_id="from-creds")
    assert _resolve_vertex_project({"project_id": None, "credentials": sa}) == "from-env"


def test_service_account_key_names_the_project():
    _clear()
    sa = _creds(type="service_account", project_id="lightwell-devel")
    assert _resolve_vertex_project({"project_id": None, "credentials": sa}) == "lightwell-devel"


def test_adc_file_names_it_under_quota_project_id():
    _clear()
    adc = _creds(type="authorized_user", quota_project_id="adc-project")
    assert _resolve_vertex_project({"project_id": None, "credentials": adc}) == "adc-project"


def test_credentials_env_var_is_honoured():
    _clear()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds(
        type="service_account", project_id="from-gac")
    assert _resolve_vertex_project({"project_id": None}) == "from-gac"


def test_region_explicit_then_env_then_error():
    _clear()
    assert _resolve_vertex_region({"region": "us-east5"}) == "us-east5"
    os.environ["CLOUD_ML_REGION"] = "europe-west1"
    assert _resolve_vertex_region({"region": None}) == "europe-west1"
    _clear()
    try:
        _resolve_vertex_region({"region": None})
    except ProposerLaunchError as exc:
        assert "CLOUD_ML_REGION" in str(exc), exc
    else:
        raise AssertionError("an unset region must raise")


def test_credentials_naming_no_project_raise_actionably():
    _clear()
    bad = _creds(type="service_account")          # no project of any kind
    try:
        _resolve_vertex_project({"project_id": None, "credentials": bad})
    except ProposerLaunchError as exc:
        assert "names no project" in str(exc), exc
    else:
        raise AssertionError("credentials naming no project must raise")


def test_preflight_accepts_a_null_project():
    """The gate that shipped broken: _check_environment carried its own copy of
    the rule and rejected exactly the config the inference exists to serve."""
    _clear()
    sa = _creds(type="service_account", project_id="inferred-here")
    vc = {"project_id": None, "region": "us-east5", "credentials": sa}
    try:
        _check_environment("claude_code", claude_auth="vertex", vertex_cfg=vc)
    except ProposerLaunchError as exc:
        # A missing claude CLI is this environment's business, not the gate's.
        if "project_id" in str(exc) or "region" in str(exc):
            raise AssertionError(f"preflight still demands a project: {exc}")


def test_preflight_still_rejects_a_missing_region():
    _clear()
    sa = _creds(type="service_account", project_id="p")
    vc = {"project_id": None, "region": None, "credentials": sa}
    try:
        _check_environment("claude_code", claude_auth="vertex", vertex_cfg=vc)
    except ProposerLaunchError as exc:
        assert "region" in str(exc), exc
    else:
        raise AssertionError("preflight must still require a region")


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}"); traceback.print_exc(); failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
