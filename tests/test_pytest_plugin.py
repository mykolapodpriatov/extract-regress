"""Plugin tests via ``pytester`` covering the core user loop (§3.8, §5)."""

from __future__ import annotations

import json

import pytest

from extract_regress.config import ERConfig
from extract_regress.pytest_plugin import _CASE_REGISTRY, _resolve_fixtures_dirs, case
from tests._fakes import FakeExtractor

pytest_plugins = ["pytester"]


# A conftest the inner pytest run will use: registers a FakeExtractor through
# the canonical ``extract_regress_config()`` hook.
PASSING_CONFTEST = """
from extract_regress import ERConfig
from extract_regress.types import ExtractionResult

RESPONSES = {"hello invoice": {"total": 100, "vendor": "ACME"}}

def _extract(source):
    text = source if isinstance(source, str) else source.read_text()
    return RESPONSES[text]

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""

DRIFTED_CONFTEST = """
from extract_regress import ERConfig

def _extract(source):
    text = source if isinstance(source, str) else source.read_text()
    # Drift: vendor changed, total dropped a field.
    return {"total": 100, "vendor": "Globex"}

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""

ERRORING_CONFTEST = """
from extract_regress import ERConfig
from extract_regress.types import ExtractionResult

def _extract(source):
    return ExtractionResult(error="provider exploded")

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""


TWO_FIXTURE_CONFTEST = """
from extract_regress import ERConfig

RESPONSES = {
    "doc one": {"total": 100, "vendor": "ACME"},
    "doc two": {"total": 200, "vendor": "Globex"},
}

def _extract(source):
    text = source if isinstance(source, str) else source.read_text()
    return RESPONSES[text]

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""


def _make_fixture(
    pytester: pytest.Pytester,
    expected: dict | None,
    *,
    name: str = "invoice_basic",
    source_inline: str = "hello invoice",
) -> None:
    fixtures = pytester.path / "fixtures"
    fixtures.mkdir(exist_ok=True)
    payload: dict = {
        "version": 1,
        "name": name,
        "source_inline": source_inline,
    }
    # An un-recorded fixture OMITS the ``expected`` key; an empty ``{}`` would
    # now correctly count as a recorded (empty) golden.
    if expected is not None:
        payload["expected"] = expected
    (fixtures / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_passing_fixture(pytester: pytest.Pytester) -> None:
    # No per-fixture test file is needed: the plugin auto-collects one item
    # per fixture from the conftest hook.
    pytester.makeconftest(PASSING_CONFTEST)
    _make_fixture(pytester, {"total": 100, "vendor": "ACME"})
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)


def test_regressed_fixture_fails_with_diff_output(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(DRIFTED_CONFTEST)
    _make_fixture(pytester, {"total": 100, "vendor": "ACME"})
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    # The FieldDiff table is rendered in the failure output.
    result.stdout.fnmatch_lines(["*non-tolerated diff*"])


def test_record_then_replay_roundtrip(pytester: pytest.Pytester) -> None:
    """The core loop: --er-record writes goldens, a plain run then replays."""
    pytester.makeconftest(PASSING_CONFTEST)
    _make_fixture(pytester, expected=None)  # no golden yet

    # Record: passes (assertions skipped) and writes the golden.
    rec = pytester.runpytest("--er-record", "-p", "no:cacheprovider")
    rec.assert_outcomes(passed=1)

    golden = json.loads(
        (pytester.path / "fixtures" / "invoice_basic.json").read_text(encoding="utf-8")
    )
    assert golden["expected"] == {"total": 100, "vendor": "ACME"}

    # Replay: a plain run now finds a matching golden and passes.
    replay = pytester.runpytest("-p", "no:cacheprovider")
    replay.assert_outcomes(passed=1)


def test_update_overwrites_stale_golden(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(PASSING_CONFTEST)
    _make_fixture(pytester, {"total": 1, "vendor": "STALE"})  # stale golden

    upd = pytester.runpytest("--er-update", "-p", "no:cacheprovider")
    upd.assert_outcomes(passed=1)

    golden = json.loads(
        (pytester.path / "fixtures" / "invoice_basic.json").read_text(encoding="utf-8")
    )
    assert golden["expected"] == {"total": 100, "vendor": "ACME"}


def test_record_with_erroring_extractor_does_not_write_empty_golden(
    pytester: pytest.Pytester,
) -> None:
    # --er-record against a failing extractor must refuse to pin {} as golden.
    pytester.makeconftest(ERRORING_CONFTEST)
    _make_fixture(pytester, expected=None)  # no golden yet

    rec = pytester.runpytest("--er-record", "-p", "no:cacheprovider")
    rec.assert_outcomes(failed=1)  # the record refuses and fails the item

    golden = json.loads(
        (pytester.path / "fixtures" / "invoice_basic.json").read_text(encoding="utf-8")
    )
    # The fixture was NOT written with an empty golden: it stays un-recorded.
    assert "expected" not in golden


def test_update_with_erroring_extractor_does_not_clobber_golden(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(ERRORING_CONFTEST)
    _make_fixture(pytester, {"total": 100, "vendor": "ACME"})  # good golden

    upd = pytester.runpytest("--er-update", "-p", "no:cacheprovider")
    upd.assert_outcomes(failed=1)

    golden = json.loads(
        (pytester.path / "fixtures" / "invoice_basic.json").read_text(encoding="utf-8")
    )
    # The good golden survives the errored update.
    assert golden["expected"] == {"total": 100, "vendor": "ACME"}


def test_markdown_report_written_once_for_whole_run(pytester: pytest.Pytester) -> None:
    # With two fixtures, the report must contain exactly one header and one
    # combined diff table — not one duplicated stanza per fixture (the old bug,
    # which appended the whole report once per fixture).
    pytester.makeconftest(TWO_FIXTURE_CONFTEST)
    # Both fixtures drift so the report renders a diff row for each, in one table.
    _make_fixture(
        pytester,
        {"total": 100, "vendor": "STALE_ONE"},
        name="invoice_one",
        source_inline="doc one",
    )
    _make_fixture(
        pytester,
        {"total": 200, "vendor": "STALE_TWO"},
        name="invoice_two",
        source_inline="doc two",
    )
    report_path = pytester.path / "report.md"
    result = pytester.runpytest("--er-report", f"md:{report_path}", "-p", "no:cacheprovider")
    result.assert_outcomes(failed=2)

    text = report_path.read_text(encoding="utf-8")
    # Exactly one document header and one table header for the entire run.
    assert text.count("## extract-regress") == 1
    assert text.count("| Fixture | Path | Kind |") == 1
    # Both fixtures' rows live in that single combined table.
    assert "invoice_one" in text
    assert "invoice_two" in text


ONE_ERROR_ONE_OK_CONFTEST = """
from extract_regress import ERConfig
from extract_regress.types import ExtractionResult

def _extract(source):
    text = source if isinstance(source, str) else source.read_text()
    if text == "boom":
        return ExtractionResult(error="provider exploded")
    return {"total": 100, "vendor": "ACME"}

def extract_regress_config():
    return ERConfig(extract_fn=_extract, fixtures_dir="fixtures")
"""


def test_empty_session_writes_valid_empty_report(pytester: pytest.Pytester) -> None:
    # No fixtures are collected at all, yet --er-report is set. The plugin must
    # write a valid, empty RunReport (not silently skip it, and never crash).
    pytester.makeconftest(PASSING_CONFTEST)
    # Deliberately create NO fixtures: the fixtures dir does not exist.
    report_path = pytester.path / "report.md"
    result = pytester.runpytest("--er-report", f"md:{report_path}", "-p", "no:cacheprovider")
    # Zero extract-regress items; the session is otherwise clean.
    result.assert_outcomes(passed=0, failed=0, errors=0)

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert text.count("## extract-regress") == 1
    assert "0 fixtures" in text
    # No diff table is rendered for an empty run.
    assert "| Fixture | Path | Kind |" not in text


def test_empty_session_without_report_option_does_not_crash(pytester: pytest.Pytester) -> None:
    # The empty-session path must be a no-op (and never raise) when no report
    # was requested.
    pytester.makeconftest(PASSING_CONFTEST)
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=0, failed=0, errors=0)


def test_record_error_fixture_appears_in_report(pytester: pytest.Pytester) -> None:
    # An errored fixture during --er-record must be carried into the session
    # report as a FixtureResult (counted), not silently dropped. Previously the
    # record path raised before appending, so the report showed 0 fixtures.
    pytester.makeconftest(ERRORING_CONFTEST)
    _make_fixture(pytester, expected=None)  # no golden yet
    report_path = pytester.path / "report.md"

    rec = pytester.runpytest(
        "--er-record", "--er-report", f"md:{report_path}", "-p", "no:cacheprovider"
    )
    rec.assert_outcomes(failed=1)  # the record refuses and fails the item

    text = report_path.read_text(encoding="utf-8")
    # The errored fixture is now part of the run: the summary reflects one
    # fixture and an overall FAIL (it was invisible / "0 fixtures" before).
    assert "## extract-regress: FAIL" in text
    assert "1 fixtures" in text


def test_record_snapshot_excludes_errored_fixture_with_stale_golden(
    pytester: pytest.Pytester,
) -> None:
    # Two fixtures: one extracts cleanly (gets recorded), one errors but still
    # has a STALE golden on disk. The refreshed coverage snapshot must exclude
    # the errored fixture's stale golden, counting only the reproducible sample.
    pytester.makeconftest(ONE_ERROR_ONE_OK_CONFTEST)
    # ok_fixture has no golden yet -> it will be recorded with {total, vendor}.
    _make_fixture(pytester, expected=None, name="ok_fixture", source_inline="hello invoice")
    # bad_fixture errors, but a stale golden exists carrying a field ("ghost")
    # that the clean fixture does not have.
    _make_fixture(
        pytester,
        {"ghost": "stale-only-field"},
        name="bad_fixture",
        source_inline="boom",
    )

    rec = pytester.runpytest("--er-record", "-p", "no:cacheprovider")
    # ok_fixture records (passes); bad_fixture errors (fails).
    rec.assert_outcomes(passed=1, failed=1)

    baseline = json.loads(
        (pytester.path / "fixtures" / "coverage_baseline.json").read_text(encoding="utf-8")
    )
    # The snapshot reflects ONLY the clean fixture; the errored fixture's stale
    # "ghost" field must not leak into the corpus fill-rates.
    assert "ghost" not in baseline
    assert baseline.get("total") == 1.0
    assert baseline.get("vendor") == 1.0


def test_duplicate_case_registration_with_conflicting_config_raises() -> None:
    # Two @case decorators for the same fixture with different configs must not
    # silently last-writer-win; the conflict is raised eagerly.
    cfg_a = ERConfig(extract_fn=FakeExtractor({}), fixtures_dir="fixtures")
    cfg_b = ERConfig(extract_fn=FakeExtractor({}), fixtures_dir="other_fixtures")

    name = "dup_case_fixture"
    try:
        case(name, config=cfg_a)(lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            case(name, config=cfg_b)(lambda: None)
        # Re-registering the SAME config is idempotent and must not raise.
        case(name, config=cfg_a)(lambda: None)
    finally:
        _CASE_REGISTRY.pop(name, None)


def test_resolve_fixtures_dirs_scans_all_registered_case_dirs() -> None:
    # With only @case overrides (no conftest hook), every DISTINCT fixtures_dir
    # in the registry must be scanned, not just the first one.
    cfg_x = ERConfig(extract_fn=FakeExtractor({}), fixtures_dir="dir_x")
    cfg_y = ERConfig(extract_fn=FakeExtractor({}), fixtures_dir="dir_y")
    cfg_x_again = ERConfig(extract_fn=FakeExtractor({}), fixtures_dir="dir_x")

    names = ("case_in_x", "case_in_y", "case_in_x_again")
    try:
        case("case_in_x", config=cfg_x)(lambda: None)
        case("case_in_y", config=cfg_y)(lambda: None)
        case("case_in_x_again", config=cfg_x_again)(lambda: None)

        dirs = _resolve_fixtures_dirs(None)
        # Both directories are present; duplicates are collapsed.
        assert set(dirs) == {"dir_x", "dir_y"}
        assert len(dirs) == 2
    finally:
        for name in names:
            _CASE_REGISTRY.pop(name, None)


def test_resolve_fixtures_dirs_prefers_hook_then_case_dirs() -> None:
    # The conftest hook's directory and the @case directories all contribute.
    hook_cfg = ERConfig(extract_fn=FakeExtractor({}), fixtures_dir="hook_dir")
    case_cfg = ERConfig(extract_fn=FakeExtractor({}), fixtures_dir="case_dir")
    try:
        case("hooked_case", config=case_cfg)(lambda: None)
        dirs = _resolve_fixtures_dirs(hook_cfg)
        assert dirs[0] == "hook_dir"
        assert "case_dir" in dirs
    finally:
        _CASE_REGISTRY.pop("hooked_case", None)
