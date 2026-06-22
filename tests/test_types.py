"""Tests for the core type aggregation properties."""

from __future__ import annotations

from extract_regress.types import (
    BudgetOutcome,
    CoverageDelta,
    FieldDiff,
    FixtureResult,
    RunReport,
    Usage,
)


def _diff(tolerated: bool) -> FieldDiff:
    return FieldDiff(path="p", kind="changed", tolerated=tolerated)


def test_usage_has_cost_and_latency_flags() -> None:
    assert not Usage().has_cost
    assert not Usage().has_latency
    assert Usage(cost_usd=0.0).has_cost
    assert Usage(latency_ms=0.0).has_latency


def test_field_diff_failing() -> None:
    assert _diff(tolerated=False).failing
    assert not _diff(tolerated=True).failing


def test_fixture_result_pass_and_failing_diffs() -> None:
    result = FixtureResult(fixture_name="f", diffs=(_diff(True), _diff(False)))
    assert len(result.failing_diffs) == 1
    assert not result.passed

    clean = FixtureResult(fixture_name="g", diffs=(_diff(True),))
    assert clean.passed


def test_fixture_result_error_fails() -> None:
    assert not FixtureResult(fixture_name="f", error="boom").passed


def test_run_report_aggregates() -> None:
    report = RunReport(
        results=(
            FixtureResult(fixture_name="a", diffs=(_diff(True),)),
            FixtureResult(fixture_name="b", diffs=(_diff(False),)),
        )
    )
    assert len(report.all_diffs) == 2
    assert [r.fixture_name for r in report.failing_results] == ["b"]
    assert not report.passed


def test_run_report_passes_when_clean() -> None:
    report = RunReport(results=(FixtureResult(fixture_name="a", diffs=(_diff(True),)),))
    assert report.passed


def test_coverage_delta_drop_fails_report() -> None:
    report = RunReport(
        coverage_deltas=(
            CoverageDelta(path="x", baseline_fill_rate=1.0, current_fill_rate=0.0, dropped=True),
        )
    )
    assert report.dropped_coverage
    assert not report.passed
    assert report.coverage_deltas[0].delta == -1.0


def test_budget_outcome_failing_only_when_checked() -> None:
    assert not BudgetOutcome(checked=False, passed=False).failing
    assert BudgetOutcome(checked=True, passed=False).failing


def test_budget_failure_fails_report() -> None:
    report = RunReport(budget=BudgetOutcome(checked=True, passed=False))
    assert not report.passed
