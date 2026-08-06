"""Tests for claude_code proposer auth modes (subscription | api_key | vertex).

Zero-dependency runner (no pytest in the venvs):

    venv/bin/python tests/test_proposer_auth.py          # root venv

Covers forge/proposer.py:
  - _resolve_anthropic_api_key (env precedence, .env fallback)
  - _resolve_gcp_credentials (explicit > $GOOGLE_APPLICATION_CREDENTIALS > ADC)
  - _agent_auth_extra_env (per-mode env injection; codex unaffected)
  - _check_environment (actionable errors per mode)
  - _prepare_claude_home (creds copy only for subscription; stale copy removed)
  - _prepare_agent_auth / _build_singularity_cmd (vertex bind; subscription
    regression — no /gcp bind, cmd shape unchanged)
and forge/orchestrator.py:
  - _resolve_claude_auth + config validation of proposer.claude_code.auth
"""
import contextlib
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

from forge import proposer as P  # noqa: E402
from forge.proposer import ProposerLaunchError  # noqa: E402

_TEST_RUN_ID = "_test_proposer_auth"


@contextlib.contextmanager
def _patched(obj, **attrs):
    """Temporarily set attributes on `obj`, restoring afterwards."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def _patched_env(**env):
    """Temporarily set/unset os.environ keys (None = ensure absent)."""
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _tmpfile(dir, name, content="x"):
    p = Path(dir) / name
    p.write_text(content, encoding="utf-8")
    return p


VC = {"project_id": "itpc-gcp-ai-eng-claude", "region": "us-east5", "credentials": None}


# ---------------- _resolve_anthropic_api_key ----------------

def test_api_key_env_wins():
    with _patched_env(ANTHROPIC_API_KEY="sk-env"):
        assert P._resolve_anthropic_api_key() == "sk-env"


def test_api_key_dotenv_fallback():
    with _patched_env(ANTHROPIC_API_KEY=None):
        with _patched(P, dotenv_values=lambda _p: {"ANTHROPIC_API_KEY": "sk-dotenv"}):
            assert P._resolve_anthropic_api_key() == "sk-dotenv"


def test_api_key_absent():
    with _patched_env(ANTHROPIC_API_KEY=None):
        with _patched(P, dotenv_values=lambda _p: {}):
            assert P._resolve_anthropic_api_key() == ""


# ---------------- _resolve_gcp_credentials ----------------

def test_gcp_explicit_beats_env_and_adc():
    with tempfile.TemporaryDirectory() as td:
        explicit = _tmpfile(td, "sa.json")
        env_file = _tmpfile(td, "env.json")
        adc = _tmpfile(td, "adc.json")
        with _patched_env(GOOGLE_APPLICATION_CREDENTIALS=str(env_file)):
            with _patched(P, _GCP_ADC_DEFAULT=adc):
                got = P._resolve_gcp_credentials({**VC, "credentials": str(explicit)})
        assert got == explicit, got


def test_gcp_env_beats_adc():
    with tempfile.TemporaryDirectory() as td:
        env_file = _tmpfile(td, "env.json")
        adc = _tmpfile(td, "adc.json")
        with _patched_env(GOOGLE_APPLICATION_CREDENTIALS=str(env_file)):
            with _patched(P, _GCP_ADC_DEFAULT=adc):
                assert P._resolve_gcp_credentials(VC) == env_file


def test_gcp_adc_default():
    with tempfile.TemporaryDirectory() as td:
        adc = _tmpfile(td, "adc.json")
        with _patched_env(GOOGLE_APPLICATION_CREDENTIALS=None):
            with _patched(P, _GCP_ADC_DEFAULT=adc):
                assert P._resolve_gcp_credentials(VC) == adc


def test_gcp_missing_is_actionable():
    with tempfile.TemporaryDirectory() as td:
        with _patched_env(GOOGLE_APPLICATION_CREDENTIALS=None):
            with _patched(P, _GCP_ADC_DEFAULT=Path(td) / "nope.json"):
                try:
                    P._resolve_gcp_credentials(VC)
                except ProposerLaunchError as exc:
                    msg = str(exc)
                    assert "gcloud auth application-default login" in msg
                    assert "GOOGLE_APPLICATION_CREDENTIALS" in msg
                else:
                    raise AssertionError("expected ProposerLaunchError")


# ---------------- _agent_auth_extra_env ----------------

def test_extra_env_api_key():
    with _patched_env(ANTHROPIC_API_KEY="sk-test-123"):
        env = P._agent_auth_extra_env("claude_code", claude_auth="api_key")
    assert env == {"SINGULARITYENV_ANTHROPIC_API_KEY": "sk-test-123"}


def test_extra_env_api_key_missing_raises():
    with _patched_env(ANTHROPIC_API_KEY=None):
        with _patched(P, dotenv_values=lambda _p: {}):
            try:
                P._agent_auth_extra_env("claude_code", claude_auth="api_key")
            except ProposerLaunchError:
                pass
            else:
                raise AssertionError("expected ProposerLaunchError")


def test_extra_env_vertex():
    env = P._agent_auth_extra_env("claude_code", claude_auth="vertex", vertex_cfg=VC)
    assert env == {
        "SINGULARITYENV_CLAUDE_CODE_USE_VERTEX": "1",
        "SINGULARITYENV_ANTHROPIC_VERTEX_PROJECT_ID": "itpc-gcp-ai-eng-claude",
        "SINGULARITYENV_CLOUD_ML_REGION": "us-east5",
        "SINGULARITYENV_GOOGLE_APPLICATION_CREDENTIALS": "/gcp/credentials.json",
    }, env


def test_extra_env_subscription_token():
    with tempfile.TemporaryDirectory() as td:
        tok = _tmpfile(td, "oauth_token", "sk-ant-oat01-abc\n")
        with _patched(P, _HOST_OAUTH_TOKEN_FILE=tok):
            env = P._agent_auth_extra_env("claude_code", claude_auth="subscription")
    assert env == {"SINGULARITYENV_CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-abc"}


def test_extra_env_subscription_no_token_falls_back():
    with tempfile.TemporaryDirectory() as td:
        with _patched(P, _HOST_OAUTH_TOKEN_FILE=Path(td) / "missing"):
            env = P._agent_auth_extra_env("claude_code", claude_auth="subscription")
    assert env == {}  # container falls back to credentials.json copy


def test_extra_env_codex_ignores_claude_auth():
    with _patched_env(OPENAI_API_KEY="sk-openai"):
        env = P._agent_auth_extra_env("codex", claude_auth="vertex", vertex_cfg=VC)
    assert env == {"SINGULARITYENV_OPENAI_API_KEY": "sk-openai"}


# ---------------- _check_environment ----------------

@contextlib.contextmanager
def _cc_env_ready(td):
    """SIF + claude binary present; credential state left to each test."""
    sif = _tmpfile(td, "proposer-base.sif")
    cc_bin = _tmpfile(td, "claude")
    with _patched(P, PROPOSER_BASE_SIF=sif, _HOST_CLAUDE_BIN=cc_bin):
        yield


def _expect_launch_error(fn, *needles):
    try:
        fn()
    except ProposerLaunchError as exc:
        for n in needles:
            assert n in str(exc), f"error lacks {n!r}: {exc}"
    else:
        raise AssertionError("expected ProposerLaunchError")


def test_check_env_bad_auth_mode():
    with tempfile.TemporaryDirectory() as td:
        with _cc_env_ready(td):
            _expect_launch_error(
                lambda: P._check_environment("claude_code", claude_auth="oauth2"),
                "unknown claude_auth", "subscription",
            )


def test_check_env_api_key_missing():
    with tempfile.TemporaryDirectory() as td:
        with _cc_env_ready(td):
            with _patched_env(ANTHROPIC_API_KEY=None):
                with _patched(P, dotenv_values=lambda _p: {}):
                    _expect_launch_error(
                        lambda: P._check_environment("claude_code", claude_auth="api_key"),
                        "ANTHROPIC_API_KEY", ".env",
                    )


def test_check_env_vertex_incomplete_cfg():
    with tempfile.TemporaryDirectory() as td:
        with _cc_env_ready(td):
            _expect_launch_error(
                lambda: P._check_environment(
                    "claude_code", claude_auth="vertex",
                    vertex_cfg={"project_id": "", "region": "us-east5"},
                ),
                "project_id", "region",
            )


def test_check_env_vertex_ok_with_adc():
    with tempfile.TemporaryDirectory() as td:
        with _cc_env_ready(td):
            adc = _tmpfile(td, "adc.json")
            with _patched_env(GOOGLE_APPLICATION_CREDENTIALS=None):
                with _patched(P, _GCP_ADC_DEFAULT=adc):
                    P._check_environment("claude_code", claude_auth="vertex", vertex_cfg=VC)


def test_check_env_subscription_no_creds():
    with tempfile.TemporaryDirectory() as td:
        with _cc_env_ready(td):
            with _patched(
                P,
                _HOST_CLAUDE_CREDS=Path(td) / "no_creds.json",
                _HOST_OAUTH_TOKEN_FILE=Path(td) / "no_token",
            ):
                _expect_launch_error(
                    lambda: P._check_environment("claude_code", claude_auth="subscription"),
                    "claude login", "setup-token",
                )


# ---------------- _prepare_claude_home / _prepare_agent_auth / cmd ----------------

@contextlib.contextmanager
def _test_workspace():
    P.paths.set_run_id(_TEST_RUN_ID)
    try:
        yield P.paths.workspace
    finally:
        shutil.rmtree(P.paths.workspace, ignore_errors=True)
        P.paths._run_id = None


def test_prepare_home_subscription_copies_creds():
    with tempfile.TemporaryDirectory() as td, _test_workspace() as ws:
        creds = _tmpfile(td, "creds.json", '{"tok": 1}')
        with _patched(P, _HOST_CLAUDE_CREDS=creds):
            home = P._prepare_claude_home("subscription")
        staged = home / ".claude" / ".credentials.json"
        assert staged.exists() and staged.read_text() == '{"tok": 1}'


def test_prepare_home_api_key_removes_stale_copy():
    with tempfile.TemporaryDirectory() as td, _test_workspace() as ws:
        creds = _tmpfile(td, "creds.json", "{}")
        with _patched(P, _HOST_CLAUDE_CREDS=creds):
            P._prepare_claude_home("subscription")   # stage a copy first
            home = P._prepare_claude_home("api_key")  # then switch modes
        assert not (home / ".claude" / ".credentials.json").exists()


def test_prepare_agent_auth_vertex_bind_and_cmd():
    with tempfile.TemporaryDirectory() as td, _test_workspace() as ws:
        adc = _tmpfile(td, "adc.json")
        with _patched_env(GOOGLE_APPLICATION_CREDENTIALS=None):
            with _patched(P, _GCP_ADC_DEFAULT=adc):
                home, binds, env = P._prepare_agent_auth(
                    "claude_code", claude_auth="vertex", vertex_cfg=VC,
                )
        assert binds == [f"{adc}:/gcp/credentials.json:ro"], binds
        assert env["SINGULARITYENV_CLAUDE_CODE_USE_VERTEX"] == "1"

        cmd = P._build_singularity_cmd(
            propose_args=["--x"], proposer_home=home, agent="claude_code",
            extra_binds=binds,
        )
        joined = " ".join(cmd)
        assert f"{adc}:/gcp/credentials.json:ro" in joined


def test_cmd_subscription_regression_no_gcp():
    """Subscription path must be bit-identical to the pre-feature cmd shape:
    no /gcp bind, extra_binds omitted == extra_binds None == empty list."""
    with _test_workspace() as ws:
        home = ws / ".proposer_home"
        home.mkdir(parents=True, exist_ok=True)
        base = P._build_singularity_cmd(
            propose_args=["--x"], proposer_home=home, agent="claude_code",
        )
        none_ = P._build_singularity_cmd(
            propose_args=["--x"], proposer_home=home, agent="claude_code",
            extra_binds=None,
        )
        empty = P._build_singularity_cmd(
            propose_args=["--x"], proposer_home=home, agent="claude_code",
            extra_binds=[],
        )
        assert base == none_ == empty
        assert "/gcp/" not in " ".join(base)


# ---------------- orchestrator config plumbing ----------------

def test_orchestrator_resolve_claude_auth():
    from forge.orchestrator import DEFAULT_CONFIG, _resolve_claude_auth
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    auth, vc = _resolve_claude_auth(cfg)
    assert auth == "subscription"
    assert vc["project_id"] == "itpc-gcp-ai-eng-claude" and vc["region"] == "us-east5"

    cfg["proposer"]["claude_code"]["auth"] = "vertex"
    auth, vc = _resolve_claude_auth(cfg)
    assert auth == "vertex"


def test_orchestrator_validates_auth_value():
    """A bad proposer.claude_code.auth must abort config resolution."""
    import argparse
    import copy
    import yaml
    from forge.orchestrator import build_arg_parser, _resolve_config

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "bad.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "datasets": {"locomo": {}},
            "proposer": {"claude_code": {"auth": "oauth2"}},
        }))
        args = build_arg_parser().parse_args(["--config", str(cfg_path)])
        try:
            _resolve_config(args)
        except ValueError as exc:
            assert "proposer.claude_code.auth" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_orchestrator_cli_claude_auth_override():
    import yaml
    from forge.orchestrator import build_arg_parser, _resolve_config

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "ok.yaml"
        # strict_config: False — this test isolates the --claude-auth CLI override;
        # it deliberately ships a minimal config and must not be gated by the
        # strict-completeness check (added 2026-07-26, after this test).
        cfg_path.write_text(yaml.safe_dump(
            {"datasets": {"locomo": {}}, "strict_config": False}))
        args = build_arg_parser().parse_args(
            ["--config", str(cfg_path), "--claude-auth", "api_key"]
        )
        cfg = _resolve_config(args)
        assert cfg["proposer"]["claude_code"]["auth"] == "api_key"


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
