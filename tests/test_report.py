"""Tests for the terminal and Markdown report renderers."""

from __future__ import annotations

import json

from extract_regress.report import (
    render_json,
    render_markdown,
    render_terminal,
    summary_line,
)
from extract_regress.types import (
    BudgetOutcome,
    CoverageDelta,
    FieldDiff,
    FixtureResult,
    RunReport,
)


def _passing_report() -> RunReport:
    return RunReport(
        results=(
            FixtureResult(
                fixture_name="ok",
                diffs=(
                    FieldDiff(
                        path="total",
                        kind="changed",
                        golden=100,
                        actual=100.0,
                        tolerated=True,
                        reason="within numeric tolerance",
                    ),
                ),
            ),
        )
    )


def _failing_report() -> RunReport:
    return RunReport(
        results=(
            FixtureResult(
                fixture_name="bad",
                diffs=(
                    FieldDiff(
                        path="vendor",
                        kind="changed",
                        golden="ACME",
                        actual="Globex",
                        tolerated=False,
                        reason="value differs",
                    ),
                ),
            ),
        ),
        coverage_deltas=(
            CoverageDelta(
                path="tax_id",
                baseline_fill_rate=1.0,
                current_fill_rate=0.5,
                dropped=True,
            ),
        ),
        budget=BudgetOutcome(
            checked=True,
            passed=False,
            total_cost_usd=5.0,
            max_cost_usd=1.0,
            messages=("cost $5.0000 exceeds budget $1.0000",),
        ),
    )


def test_summary_line_pass_and_fail() -> None:
    assert summary_line(_passing_report()).startswith("PASS")
    assert summary_line(_failing_report()).startswith("FAIL")


def test_terminal_render_contains_paths() -> None:
    out = render_terminal(_failing_report(), color=False)
    assert "vendor" in out
    assert "FAIL" in out
    assert "coverage drop" in out
    assert "budget" in out


def test_terminal_render_no_diffs() -> None:
    out = render_terminal(RunReport(), color=False)
    assert "No field diffs" in out


def test_markdown_render_table_and_sections() -> None:
    md = render_markdown(_failing_report())
    assert md.startswith("## extract-regress: FAIL")
    assert "| Fixture | Path |" in md
    assert "`vendor`" in md
    assert "### Coverage drops" in md
    assert "### Budget" in md


def test_markdown_escapes_pipes() -> None:
    report = RunReport(
        results=(
            FixtureResult(
                fixture_name="f",
                diffs=(
                    FieldDiff(
                        path="x",
                        kind="changed",
                        golden="a|b",
                        actual="c",
                        tolerated=False,
                        reason="value differs",
                    ),
                ),
            ),
        )
    )
    md = render_markdown(report)
    assert "a\\|b" in md


def test_markdown_passing_has_no_failure_marker_in_status() -> None:
    md = render_markdown(_passing_report())
    assert md.startswith("## extract-regress: PASS")


def test_render_json_parses_and_round_trips_failing_diff() -> None:
    payload = json.loads(render_json(_failing_report()))

    assert payload["status"] == "FAIL"
    assert payload["summary"]["failing_diffs"] == 1
    assert payload["summary"]["coverage_drops"] == 1

    fixtures = {f["fixture"]: f for f in payload["fixtures"]}
    # The failing diff round-trips every identifying field verbatim.
    assert fixtures["bad"]["diffs"][0] == {
        "path": "vendor",
        "kind": "changed",
        "golden": "ACME",
        "actual": "Globex",
        "tolerated": False,
        "reason": "value differs",
    }
    assert fixtures["bad"]["passed"] is False

    # Coverage drops and budget carry through the stable schema.
    assert payload["coverage_drops"][0]["path"] == "tax_id"
    assert payload["coverage_drops"][0]["baseline_fill_rate"] == 1.0
    assert payload["budget"]["passed"] is False
    assert payload["budget"]["max_cost_usd"] == 1.0


def test_render_json_passing_status_and_sorted_keys() -> None:
    text = render_json(_passing_report())
    payload = json.loads(text)

    assert payload["status"] == "PASS"
    assert payload["summary"]["failing_diffs"] == 0

    # sort_keys=True: the top-level keys appear in sorted order in the output.
    top_keys = ["budget", "coverage_drops", "fixtures", "status", "summary"]
    positions = [text.index(f'"{key}"') for key in top_keys]
    assert positions == sorted(positions)
