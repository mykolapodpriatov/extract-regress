"""Tests for deferring a known-broken fixture.

The rules that give quarantine teeth are what these assert: a passing
quarantined fixture fails the build, an expired entry fails the build, and a
live entry absorbs a failure without hiding it from the report.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from extract_regress.config import ERConfig, ProjectConfig
from extract_regress.fixtures import Fixture, FixtureStore
from extract_regress.quarantine import QuarantineConfig, QuarantineRule
from extract_regress.report import (
    render_json,
    render_junit,
    render_markdown,
    render_terminal,
    summary_line,
)
from extract_regress.runner import Runner
from extract_regress.types import ExtractionResult
from tests._fakes import DriftedExtractor, FakeExtractor

TODAY = date(2026, 6, 1)
FUTURE = TODAY + timedelta(days=30)
PAST = TODAY - timedelta(days=1)


def _seed(fixtures_dir: Path, name: str, source: str, expected: dict) -> None:
    FixtureStore(fixtures_dir).save(Fixture(name=name, source_inline=source, expected=expected))


def _drifting_config(fixtures_dir: Path, rules: tuple[QuarantineRule, ...]) -> ERConfig:
    """A config whose single fixture always drifts, so it always fails."""
    _seed(fixtures_dir, "broken", "doc", {"vendor": "ACME"})
    extractor = DriftedExtractor(
        {"doc": {"vendor": "ACME"}},
        mutate=lambda v: {**v, "vendor": "Globex"},
    )
    return ERConfig(
        extract_fn=extractor,
        fixtures_dir=str(fixtures_dir),
        quarantine=QuarantineConfig(rules=rules),
    )


def _passing_config(fixtures_dir: Path, rules: tuple[QuarantineRule, ...]) -> ERConfig:
    _seed(fixtures_dir, "broken", "doc", {"vendor": "ACME"})
    return ERConfig(
        extract_fn=FakeExtractor({"doc": {"vendor": "ACME"}}),
        fixtures_dir=str(fixtures_dir),
        quarantine=QuarantineConfig(rules=rules),
    )


def _rule(until: date = FUTURE, name: str = "broken") -> QuarantineRule:
    return QuarantineRule(
        name=name, reason="vendor switched layout; new prompt in #41", until=until
    )


# ---------------------------------------------------------------------------
# the rule model
# ---------------------------------------------------------------------------


def test_reason_and_until_are_required() -> None:
    # An entry with no reason is a mystery to whoever finds it in six months,
    # and a deferral with no end is a deletion nobody wrote down.
    with pytest.raises(ValidationError):
        QuarantineRule(name="f", until=FUTURE)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        QuarantineRule(name="f", reason="because")  # type: ignore[call-arg]


def test_a_blank_reason_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QuarantineRule(name="f", reason="   ", until=FUTURE)


def test_two_entries_for_one_fixture_are_rejected() -> None:
    # Two entries means two stated reasons and two expiry dates, and no way to
    # say which one is the decision.
    with pytest.raises(ValidationError):
        QuarantineConfig(rules=(_rule(), _rule(until=PAST)))


def test_expiry_is_inclusive_of_the_named_day() -> None:
    rule = _rule(until=TODAY)
    assert not rule.expired(TODAY)
    assert rule.expired(TODAY + timedelta(days=1))


def test_lookup_is_by_exact_name() -> None:
    config = QuarantineConfig(rules=(_rule(name="invoice_lumen"),))
    assert config.rule_for("invoice_lumen") is not None
    # Not a glob: a pattern would quietly cover fixtures added later.
    assert config.rule_for("invoice_lumen_2") is None
    assert config.rule_for("invoice_*") is None


# ---------------------------------------------------------------------------
# what quarantine does to the run
# ---------------------------------------------------------------------------


def test_a_live_entry_absorbs_the_failure(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    assert report.passed
    assert report.failing_results, "the fixture still ran and still failed"
    assert len(report.quarantined_results) == 1
    assert not report.blocking_results


def test_without_an_entry_the_same_failure_blocks(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, ())).run(today=TODAY)

    assert not report.passed
    assert len(report.blocking_results) == 1


def test_an_expired_entry_fails_the_build(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(until=PAST),))).run(today=TODAY)

    assert not report.passed
    assert len(report.expired_quarantines) == 1
    assert not report.quarantined_results, "an expired entry absorbs nothing"


def test_a_quarantined_fixture_that_passes_fails_the_build(fixtures_dir: Path) -> None:
    # A stale entry silently swallows the regression when it comes back.
    report = Runner(_passing_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    assert not report.passed
    assert len(report.stale_quarantines) == 1


def test_an_entry_for_an_absent_fixture_is_inert(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(name="not_here"),))).run(today=TODAY)

    assert not report.passed, "the real failure is not covered"
    assert not report.quarantined_results


def test_quarantine_covers_an_extraction_error_too(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "broken", "doc", {"vendor": "ACME"})

    def failing(source: object) -> ExtractionResult:
        return ExtractionResult(error="provider exploded")

    config = ERConfig(
        extract_fn=failing,
        fixtures_dir=str(fixtures_dir),
        quarantine=QuarantineConfig(rules=(_rule(),)),
    )
    report = Runner(config).run(today=TODAY)

    assert report.passed
    assert report.results[0].error is not None


def test_expiry_is_judged_against_the_run_date(fixtures_dir: Path) -> None:
    """Resolved by the runner, so a saved report renders the same tomorrow."""
    config = _drifting_config(fixtures_dir, (_rule(until=TODAY),))

    assert Runner(config).run(today=TODAY).passed
    assert not Runner(config).run(today=TODAY + timedelta(days=1)).passed


def test_quarantine_does_not_touch_coverage(fixtures_dir: Path) -> None:
    """A quarantined fixture is still a source format the suite covers."""
    report = Runner(_drifting_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    assert report.coverage_deltas is not None
    assert report.passed


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


def test_quarantine_loads_from_a_toml_table() -> None:
    project = ProjectConfig.from_mapping(
        {
            "fixtures_dir": "tests/fx",
            "quarantine": [
                {
                    "name": "invoice_lumen_2024",
                    "reason": "vendor switched to a two-column layout",
                    "until": "2026-10-01",
                }
            ],
        }
    )

    rule = project.quarantine.rule_for("invoice_lumen_2024")
    assert rule is not None
    assert rule.until == date(2026, 10, 1)


def test_a_toml_entry_without_a_reason_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.from_mapping({"quarantine": [{"name": "f", "until": "2026-10-01"}]})


def test_er_config_inherits_the_project_quarantine() -> None:
    project = ProjectConfig.from_mapping(
        {"quarantine": [{"name": "f", "reason": "r", "until": "2026-10-01"}]}
    )
    config = ERConfig.from_project(FakeExtractor({}), project)

    assert config.quarantine.rule_for("f") is not None


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_the_summary_line_counts_quarantined_fixtures(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    line = summary_line(report)
    assert "PASS" in line
    assert "1 quarantined" in line


def test_the_summary_line_calls_out_a_stale_entry(fixtures_dir: Path) -> None:
    report = Runner(_passing_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    assert "stale quarantine" in summary_line(report)


def test_terminal_output_names_the_reason_and_the_date(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    out = render_terminal(report, color=False)
    assert "quarantined" in out
    assert "vendor switched layout" in out
    assert str(FUTURE) in out


def test_markdown_has_its_own_quarantine_section(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    md = render_markdown(report)
    assert "### Quarantine" in md
    assert "deferred" in md


def test_markdown_marks_an_expired_entry(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(until=PAST),))).run(today=TODAY)

    assert "**EXPIRED**" in render_markdown(report)


def test_json_carries_the_entry_and_the_blocking_flag(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    payload = json.loads(render_json(report))
    assert payload["status"] == "PASS"
    assert payload["summary"]["quarantined"] == 1
    fixture = payload["fixtures"][0]
    assert fixture["passed"] is False
    assert fixture["blocking"] is False
    assert fixture["quarantine"]["until"] == FUTURE.isoformat()
    assert fixture["quarantine"]["expired"] is False


def test_json_leaves_quarantine_null_for_an_uncovered_fixture(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, ())).run(today=TODAY)

    payload = json.loads(render_json(report))
    assert payload["fixtures"][0]["quarantine"] is None
    assert payload["fixtures"][0]["blocking"] is True


def test_junit_marks_a_quarantined_fixture_skipped(fixtures_dir: Path) -> None:
    # A CI test UI already knows how to render a skip with a reason.
    report = Runner(_drifting_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    xml = render_junit(report)
    assert "<skipped" in xml
    assert "vendor switched layout" in xml
    assert 'failures="0"' in xml


def test_junit_fails_a_stale_entry(fixtures_dir: Path) -> None:
    report = Runner(_passing_config(fixtures_dir, (_rule(),))).run(today=TODAY)

    xml = render_junit(report)
    assert "<failure" in xml
    assert "stale" in xml


def test_junit_fails_an_expired_entry(fixtures_dir: Path) -> None:
    report = Runner(_drifting_config(fixtures_dir, (_rule(until=PAST),))).run(today=TODAY)

    xml = render_junit(report)
    assert "<failure" in xml
    assert "expired" in xml
