"""CLI tests: record → run happy path and exit codes (§3.9, §5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from extract_regress.cli import _import_module, app

runner = CliRunner()


CONFTEST = """
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

DRIFT_CONFTEST = """
from extract_regress import ERConfig

def _extract(source):
    return {"total": 999, "vendor": "WRONG"}

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""

ERROR_CONFTEST = """
from extract_regress import ERConfig
from extract_regress.types import ExtractionResult

def _extract(source):
    return ExtractionResult(error="provider down")

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""


def _setup_project(tmp_path: Path, conftest: str, *, with_goldens: bool) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "conftest.py").write_text(conftest, encoding="utf-8")
    fixtures = project / "fixtures"
    fixtures.mkdir()
    for name, text, expected in [
        ("invoice_a", "doc-a", {"total": 100, "vendor": "ACME"}),
        ("invoice_b", "doc-b", {"total": 200, "vendor": "Globex"}),
    ]:
        payload: dict[str, object] = {
            "version": 1,
            "name": name,
            "source_inline": text,
        }
        # An un-recorded fixture OMITS the ``expected`` key entirely; an empty
        # ``{}`` would now (correctly) count as a recorded golden.
        if with_goldens:
            payload["expected"] = expected
        (fixtures / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return project


def test_record_then_run_happy_path(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=False)

    rec = runner.invoke(app, ["record", "--project-dir", str(project)])
    assert rec.exit_code == 0, rec.output
    assert "recorded 2 fixture(s)" in rec.output

    # Goldens were written.
    golden = json.loads((project / "fixtures" / "invoice_a.json").read_text())
    assert golden["expected"] == {"total": 100, "vendor": "ACME"}

    # And the coverage snapshot exists.
    assert (project / "fixtures" / "coverage_baseline.json").exists()

    run = runner.invoke(app, ["run", "--project-dir", str(project)])
    assert run.exit_code == 0, run.output
    assert "PASS" in run.output


def test_run_fails_on_regression_with_nonzero_exit(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, DRIFT_CONFTEST, with_goldens=True)
    run = runner.invoke(app, ["run", "--project-dir", str(project)])
    assert run.exit_code == 1, run.output
    assert "FAIL" in run.output


def test_run_markdown_format(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    run = runner.invoke(app, ["run", "--project-dir", str(project), "--format", "md"])
    assert run.exit_code == 0, run.output
    assert "## extract-regress" in run.output


def test_run_json_format_parses(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    run = runner.invoke(app, ["run", "--project-dir", str(project), "--format", "json"])
    assert run.exit_code == 0, run.output
    payload = json.loads(run.output)
    assert payload["status"] == "PASS"


def test_run_json_format_still_exits_one_on_regression(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, DRIFT_CONFTEST, with_goldens=True)
    run = runner.invoke(app, ["run", "--project-dir", str(project), "--format", "json"])
    assert run.exit_code == 1, run.output
    payload = json.loads(run.output)
    assert payload["status"] == "FAIL"


def test_run_rejects_unknown_format(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    run = runner.invoke(app, ["run", "--project-dir", str(project), "--format", "bogus"])
    assert run.exit_code == 2, run.output
    assert "unknown --format" in run.output


def test_report_json_format(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    rep = runner.invoke(app, ["report", "--project-dir", str(project), "--format", "json"])
    assert rep.exit_code == 0, rep.output
    # ``report`` renders the last run (which may be an in-memory one from a
    # prior invocation in this process), so assert the stable JSON shape rather
    # than a specific pass/fail verdict.
    payload = json.loads(rep.output)
    assert payload["status"] in {"PASS", "FAIL"}
    assert "fixtures" in payload and "budget" in payload


def test_report_rejects_unknown_format(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    rep = runner.invoke(app, ["report", "--project-dir", str(project), "--format", "bogus"])
    assert rep.exit_code == 2, rep.output
    assert "unknown --format" in rep.output


def test_update_overwrites_goldens(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    # Corrupt a golden, then update from current (correct) outputs.
    path = project / "fixtures" / "invoice_a.json"
    payload = json.loads(path.read_text())
    payload["expected"] = {"total": -1, "vendor": "STALE"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    upd = runner.invoke(app, ["update", "--project-dir", str(project)])
    assert upd.exit_code == 0, upd.output

    restored = json.loads(path.read_text())
    assert restored["expected"] == {"total": 100, "vendor": "ACME"}


def test_report_command_runs(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    rep = runner.invoke(app, ["report", "--project-dir", str(project)])
    assert rep.exit_code == 0, rep.output


def test_no_args_shows_help() -> None:
    # ``no_args_is_help`` prints usage and exits with click's "no command" code.
    result = runner.invoke(app, [])
    assert result.exit_code != 0  # usage shown, no command executed
    assert "record" in result.output and "run" in result.output


def test_explicit_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("record", "run", "update", "report"):
        assert command in result.output


def test_missing_hook_is_a_clear_error(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "conftest.py").write_text("x = 1\n", encoding="utf-8")  # no hook
    (project / "fixtures").mkdir()
    result = runner.invoke(app, ["run", "--project-dir", str(project)])
    assert result.exit_code != 0
    assert "extract_regress_config" in result.output


def test_hook_wrong_return_type_is_a_clear_error(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "conftest.py").write_text(
        "def extract_regress_config():\n    return 42\n", encoding="utf-8"
    )
    (project / "fixtures").mkdir()
    result = runner.invoke(app, ["run", "--project-dir", str(project)])
    assert result.exit_code != 0
    assert "ERConfig" in result.output


def test_fixtures_dir_override(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    # Move fixtures into a differently-named directory and point the CLI at it.
    (project / "fixtures").rename(project / "goldens")
    run = runner.invoke(
        app,
        ["run", "--project-dir", str(project), "--fixtures-dir", str(project / "goldens")],
    )
    assert run.exit_code == 0, run.output


def test_run_with_no_budget_flag(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    run = runner.invoke(app, ["run", "--project-dir", str(project), "--no-budget"])
    assert run.exit_code == 0, run.output


def test_record_skips_errored_fixture_and_warns(tmp_path: Path) -> None:
    # A CLI record against a failing extractor must not write {} goldens and
    # must report the skipped fixtures.
    project = _setup_project(tmp_path, ERROR_CONFTEST, with_goldens=False)
    rec = runner.invoke(app, ["record", "--project-dir", str(project)])
    assert rec.exit_code == 0, rec.output
    assert "skipped 2 errored fixture(s)" in rec.output

    # Both fixtures remain un-recorded on disk (no golden written): the errored
    # extraction was never pinned, so the file still carries no ``expected`` key.
    for name in ("invoice_a", "invoice_b"):
        golden = json.loads((project / "fixtures" / f"{name}.json").read_text())
        assert "expected" not in golden
    # No coverage snapshot field was populated from an errored {} sample.
    baseline = json.loads((project / "fixtures" / "coverage_baseline.json").read_text())
    assert baseline == {}


def _write_fixture(fixtures: Path, name: str, payload: dict[str, object]) -> None:
    (fixtures / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# validate (#6)
# ---------------------------------------------------------------------------


def test_validate_ok_exits_zero(tmp_path: Path) -> None:
    # A directory of well-formed inline fixtures validates with no extraction.
    project = _setup_project(tmp_path, CONFTEST, with_goldens=False)
    result = runner.invoke(app, ["validate", "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert "invoice_a: OK" in result.output
    assert "invoice_b: OK" in result.output
    assert "all 2 fixture(s) valid" in result.output


def test_validate_dual_source_exits_two(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=False)
    _write_fixture(
        project / "fixtures",
        "dual",
        {"version": 1, "name": "dual", "source_ref": "a.txt", "source_inline": "x"},
    )
    result = runner.invoke(app, ["validate", "--project-dir", str(project)])
    assert result.exit_code == 2, result.output
    assert "exactly one of" in result.output


def test_validate_escaping_ref_exits_two(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=False)
    _write_fixture(
        project / "fixtures",
        "evil",
        {"version": 1, "name": "evil", "source_ref": "../secret.txt"},
    )
    result = runner.invoke(app, ["validate", "--project-dir", str(project)])
    assert result.exit_code == 2, result.output
    assert "escapes the fixture directory" in result.output


def test_validate_bad_json_exits_two(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=False)
    (project / "fixtures" / "broken.json").write_text("{ not json", encoding="utf-8")
    result = runner.invoke(app, ["validate", "--project-dir", str(project)])
    assert result.exit_code == 2, result.output
    assert "invalid JSON" in result.output


# ---------------------------------------------------------------------------
# coverage (#7)
# ---------------------------------------------------------------------------


def test_coverage_command_term(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    result = runner.invoke(app, ["coverage", "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert "coverage" in result.output
    assert "total" in result.output


def test_coverage_command_flags_drop_in_json(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    # A baseline field that the recorded goldens no longer fill drops to 0.
    (project / "fixtures" / "coverage_baseline.json").write_text(
        json.dumps({"total": 1.0, "vendor": 1.0, "tax_id": 1.0}), encoding="utf-8"
    )
    result = runner.invoke(app, ["coverage", "--project-dir", str(project), "--format", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    entries = {e["path"]: e for e in payload["coverage"]}
    assert entries["tax_id"]["dropped"] is True
    assert entries["total"]["dropped"] is False
    assert payload["drops"] == 1

    # Read-only: the baseline snapshot on disk is untouched.
    baseline = json.loads((project / "fixtures" / "coverage_baseline.json").read_text())
    assert baseline == {"total": 1.0, "vendor": 1.0, "tax_id": 1.0}


def test_coverage_command_markdown(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    result = runner.invoke(app, ["coverage", "--project-dir", str(project), "--format", "md"])
    assert result.exit_code == 0, result.output
    assert "## extract-regress coverage" in result.output


def test_coverage_rejects_unknown_format(tmp_path: Path) -> None:
    project = _setup_project(tmp_path, CONFTEST, with_goldens=True)
    result = runner.invoke(app, ["coverage", "--project-dir", str(project), "--format", "bogus"])
    assert result.exit_code == 2, result.output
    assert "unknown --format" in result.output


def test_import_module_uses_unique_name_and_cleans_syspath(tmp_path: Path) -> None:
    # Two different projects each have a ``conftest.py`` defining MARK; importing
    # both must not clobber a shared ``sys.modules['conftest']`` key, and the
    # transient sys.path entries must not leak.
    proj_a = tmp_path / "a"
    proj_b = tmp_path / "b"
    for proj, mark in ((proj_a, "AAA"), (proj_b, "BBB")):
        proj.mkdir()
        (proj / "conftest.py").write_text(f"MARK = {mark!r}\n", encoding="utf-8")

    syspath_before = list(sys.path)
    mod_a = _import_module("conftest.py", proj_a)
    mod_b = _import_module("conftest.py", proj_b)

    # Distinct modules, each with its own MARK (no last-writer-wins clobber).
    assert mod_a.MARK == "AAA"  # type: ignore[attr-defined]
    assert mod_b.MARK == "BBB"  # type: ignore[attr-defined]
    # The plain "conftest" key was never hijacked in the global module table.
    assert "conftest" not in sys.modules
    # sys.path is restored: no leaked project directories.
    assert sys.path == syspath_before
    assert str(proj_a) not in sys.path
    assert str(proj_b) not in sys.path


def test_import_module_does_not_leak_unique_name_in_sys_modules(tmp_path: Path) -> None:
    # The per-call unique module name (a path digest) must be popped from
    # sys.modules after execution: leaving it would grow the table unboundedly
    # and risk serving a stale module on a later import of the same path.
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "conftest.py").write_text("VALUE = 1\n", encoding="utf-8")

    modules_before = set(sys.modules)
    module = _import_module("conftest.py", proj)
    assert module.VALUE == 1  # type: ignore[attr-defined]

    # No new ``_extract_regress_cfg_*`` entries linger in sys.modules.
    prefix = "_extract_regress_cfg_"
    leaked = {name for name in set(sys.modules) - modules_before if name.startswith(prefix)}
    assert leaked == set(), f"leaked module names: {leaked}"
