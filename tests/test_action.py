"""Tests for the GitHub Action entrypoint (token-gated, driven fully offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extract_regress.action import entrypoint

# A hook returning canned JSON per source: matches the goldens below → PASS.
PASS_CONFTEST = """
from extract_regress import ERConfig
from extract_regress.tolerances import ToleranceConfig, ToleranceRule

RESPONSES = {
    "doc-a": {"total": 100, "vendor": "ACME"},
    "doc-b": {"total": 200, "vendor": "Globex"},
}

def _extract(source):
    text = source if isinstance(source, str) else source.read_text()
    return RESPONSES[text]

def extract_regress_config():
    return ERConfig(
        extract_fn=_extract,
        fixtures_dir="fixtures",
        tolerances=ToleranceConfig(rules=(ToleranceRule(path="vendor", ignore_case=True),)),
    )
"""

# A hook that always drifts every field → FAIL.
DRIFT_CONFTEST = """
from extract_regress import ERConfig

def _extract(source):
    return {"total": 999, "vendor": "WRONG"}

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""

_GITHUB_ENV = (
    "GITHUB_TOKEN",
    "GITHUB_OUTPUT",
    "GITHUB_EVENT_PATH",
    "GITHUB_REPOSITORY",
    "GITHUB_API_URL",
    "ER_FIXTURES_DIR",
    "ER_BUDGET",
    "ER_COMMENT",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited GitHub/action env so each test drives from a blank slate."""
    for var in _GITHUB_ENV:
        monkeypatch.delenv(var, raising=False)


def _setup_project(tmp_path: Path, conftest: str) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "conftest.py").write_text(conftest, encoding="utf-8")
    fixtures = project / "fixtures"
    fixtures.mkdir()
    for name, text, expected in [
        ("invoice_a", "doc-a", {"total": 100, "vendor": "ACME"}),
        ("invoice_b", "doc-b", {"total": 200, "vendor": "Globex"}),
    ]:
        payload = {"version": 1, "name": name, "source_inline": text, "expected": expected}
        (fixtures / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return project


@pytest.mark.usefixtures("clean_env")
def test_action_passing_run_renders_pass_and_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _setup_project(tmp_path, PASS_CONFTEST)
    monkeypatch.setenv("ER_PROJECT_DIR", str(project))

    code = entrypoint.main()
    captured = capsys.readouterr()

    assert code == 0
    # The PASS form is rendered to stdout and the comment step is skipped.
    assert "PASS" in captured.out
    assert "<details>" not in captured.out
    assert "PR comment skipped" in captured.err


@pytest.mark.usefixtures("clean_env")
def test_action_failing_run_lists_failing_paths_and_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _setup_project(tmp_path, DRIFT_CONFTEST)
    monkeypatch.setenv("ER_PROJECT_DIR", str(project))

    code = entrypoint.main()
    captured = capsys.readouterr()

    assert code == entrypoint.EXIT_REGRESSION
    assert "FAIL" in captured.out
    # Every drifted field path appears in the collapsed diff table.
    assert "`vendor`" in captured.out
    assert "`total`" in captured.out


@pytest.mark.usefixtures("clean_env")
def test_action_writes_github_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _setup_project(tmp_path, PASS_CONFTEST)
    out_file = tmp_path / "gh_output"
    monkeypatch.setenv("ER_PROJECT_DIR", str(project))
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    assert entrypoint.main() == 0
    assert "passed=true" in out_file.read_text(encoding="utf-8")


@pytest.mark.usefixtures("clean_env")
def test_action_reports_error_on_missing_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "conftest.py").write_text("x = 1\n", encoding="utf-8")  # no hook
    (project / "fixtures").mkdir()
    monkeypatch.setenv("ER_PROJECT_DIR", str(project))

    code = entrypoint.main()
    assert code == entrypoint.EXIT_ERROR
    assert "action error" in capsys.readouterr().err


@pytest.mark.usefixtures("clean_env")
def test_post_comment_skipped_without_token() -> None:
    assert entrypoint._post_comment("body") is False


@pytest.mark.usefixtures("clean_env")
def test_post_comment_posts_when_token_and_pr_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7}}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "tkn")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    calls: dict[str, str] = {}

    def _fake_post(url: str, token: str, body: str) -> bool:
        calls.update(url=url, token=token, body=body)
        return True

    monkeypatch.setattr(entrypoint, "_http_post_comment", _fake_post)

    assert entrypoint._post_comment("hello") is True
    assert calls["url"].endswith("/repos/owner/repo/issues/7/comments")
    assert calls["token"] == "tkn"
    assert calls["body"] == "hello"


def test_action_yaml_ships_composite() -> None:
    """The composite action manifest ships inside the package."""
    action_yml = Path(entrypoint.__file__).resolve().parent / "action.yml"
    assert action_yml.exists()
    body = action_yml.read_text(encoding="utf-8")
    assert 'using: "composite"' in body
    assert "github-token" in body
