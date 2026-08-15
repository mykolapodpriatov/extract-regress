"""Terminal (rich), Markdown, JSON, and PR-comment renderers for a :class:`RunReport`."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from io import StringIO
from typing import Any

from rich.console import Console
from rich.table import Table

from .types import CoverageDelta, FieldDiff, FixtureResult, RunReport

__all__ = [
    "render_coverage",
    "render_json",
    "render_junit",
    "render_markdown",
    "render_pr_comment",
    "render_terminal",
    "summary_line",
]


def _truncate(value: object, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summary_line(report: RunReport) -> str:
    """One-line human summary of the run outcome."""
    status = "PASS" if report.passed else "FAIL"
    failing = sum(len(r.failing_diffs) for r in report.results)
    dropped = len(report.dropped_coverage)
    parts = [
        f"{status}",
        f"{len(report.results)} fixtures",
        f"{failing} failing diff(s)",
        f"{dropped} coverage drop(s)",
    ]
    if report.budget.checked:
        parts.append("budget " + ("ok" if report.budget.passed else "exceeded"))
    return " | ".join(parts)


def _diff_rows() -> tuple[str, ...]:
    return ("fixture", "path", "kind", "golden", "actual", "status", "reason")


def render_terminal(report: RunReport, *, color: bool = True) -> str:
    """Render the report as a rich table string for the terminal."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=color,
        width=120,
        color_system="truecolor" if color else None,
    )

    table = Table(title="extract-regress field diffs", show_lines=False)
    for column in _diff_rows():
        table.add_column(column, overflow="fold")

    any_rows = False
    for result in report.results:
        for diff in result.diffs:
            any_rows = True
            status = "[green]ok[/]" if diff.tolerated else "[red]FAIL[/]"
            table.add_row(
                result.fixture_name,
                diff.path,
                diff.kind,
                _truncate(diff.golden),
                _truncate(diff.actual),
                status,
                diff.reason,
            )
    if any_rows:
        console.print(table)
    else:
        console.print("[green]No field diffs.[/]")

    for delta in report.dropped_coverage:
        console.print(
            f"[yellow]coverage drop[/] {delta.path}: "
            f"{delta.baseline_fill_rate:.2f} -> {delta.current_fill_rate:.2f}"
        )

    if report.budget.checked:
        for message in report.budget.messages:
            console.print(f"[blue]budget[/] {message}")

    console.print(summary_line(report))
    return buffer.getvalue()


def _md_cell(value: object) -> str:
    return _truncate(value).replace("|", "\\|")


def render_markdown(report: RunReport) -> str:
    """Render the report as a GitHub-flavored Markdown document."""
    lines: list[str] = []
    status = "PASS" if report.passed else "FAIL"
    lines.append(f"## extract-regress: {status}")
    lines.append("")
    lines.append(summary_line(report))
    lines.append("")

    diffs: list[tuple[str, FieldDiff]] = [
        (r.fixture_name, d) for r in report.results for d in r.diffs
    ]
    if diffs:
        lines.append("| Fixture | Path | Kind | Golden | Actual | Status | Reason |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for fixture_name, diff in diffs:
            cell_status = "ok" if diff.tolerated else "**FAIL**"
            lines.append(
                "| "
                + " | ".join(
                    [
                        fixture_name,
                        f"`{diff.path}`",
                        diff.kind,
                        _md_cell(diff.golden),
                        _md_cell(diff.actual),
                        cell_status,
                        diff.reason,
                    ]
                )
                + " |"
            )
        lines.append("")

    if report.dropped_coverage:
        lines.append("### Coverage drops")
        lines.append("")
        lines.append("| Field | Baseline | Current |")
        lines.append("| --- | --- | --- |")
        for delta in report.dropped_coverage:
            lines.append(
                f"| `{delta.path}` | {delta.baseline_fill_rate:.2f} | "
                f"{delta.current_fill_rate:.2f} |"
            )
        lines.append("")

    if report.budget.checked:
        lines.append("### Budget")
        lines.append("")
        for message in report.budget.messages:
            lines.append(f"- {message}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _diff_payload(diff: FieldDiff) -> dict[str, Any]:
    """Serialize a single field diff to a stable, JSON-friendly mapping."""
    return {
        "path": diff.path,
        "kind": diff.kind,
        "golden": diff.golden,
        "actual": diff.actual,
        "tolerated": diff.tolerated,
        "reason": diff.reason,
    }


def render_json(report: RunReport) -> str:
    """Render the report as a machine-readable JSON document.

    The schema is stable and sorted: a top-level ``status`` plus ``summary``
    counts, a ``fixtures`` array carrying every per-fixture diff (with
    ``path``/``kind``/``golden``/``actual``/``tolerated``/``reason``), the
    flagged ``coverage_drops``, and the ``budget`` outcome. ``default=str``
    guards against any non-JSON-native golden/actual value so serialization
    never raises.
    """
    payload: dict[str, Any] = {
        "status": "PASS" if report.passed else "FAIL",
        "summary": {
            "fixtures": len(report.results),
            "failing_diffs": sum(len(r.failing_diffs) for r in report.results),
            "coverage_drops": len(report.dropped_coverage),
            "budget_checked": report.budget.checked,
            "budget_passed": report.budget.passed,
        },
        "fixtures": [
            {
                "fixture": result.fixture_name,
                "error": result.error,
                "passed": result.passed,
                "diffs": [_diff_payload(d) for d in result.diffs],
            }
            for result in report.results
        ],
        "coverage_drops": [
            {
                "path": delta.path,
                "baseline_fill_rate": delta.baseline_fill_rate,
                "current_fill_rate": delta.current_fill_rate,
            }
            for delta in report.dropped_coverage
        ],
        "budget": {
            "checked": report.budget.checked,
            "passed": report.budget.passed,
            "total_cost_usd": report.budget.total_cost_usd,
            "p95_latency_ms": report.budget.p95_latency_ms,
            "max_cost_usd": report.budget.max_cost_usd,
            "max_p95_latency_ms": report.budget.max_p95_latency_ms,
            "messages": list(report.budget.messages),
        },
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=str)


def _junit_failure_message(result: FixtureResult) -> str:
    """One-line field-level mismatch (or extraction-error) summary."""
    if result.error is not None:
        return result.error
    parts: list[str] = []
    for diff in result.failing_diffs:
        detail = (
            f"{diff.path}: {diff.kind} golden={_truncate(diff.golden)} "
            f"actual={_truncate(diff.actual)}"
        )
        if diff.reason:
            detail = f"{detail} ({diff.reason})"
        parts.append(detail)
    return "; ".join(parts) or f"{result.fixture_name} failed"


def render_junit(report: RunReport) -> str:
    """Render the run as a JUnit XML test report.

    One ``<testcase>`` is emitted per fixture (``name`` is the fixture name).
    A field mismatch or extraction error becomes a ``<failure>`` on that case
    whose message is the field-level summary. A budget breach is a suite-level
    ``<failure>`` on the ``<testsuite>`` so CI still flags the run when every
    fixture passed its field checks.
    """
    failures = 0
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        name="extract-regress",
        tests=str(len(report.results)),
        failures="0",
    )
    for result in report.results:
        case = ET.SubElement(suite, "testcase", classname="fixture", name=result.fixture_name)
        if not result.passed:
            failures += 1
            message = _junit_failure_message(result)
            failure = ET.SubElement(case, "failure", type="regression", message=message)
            failure.text = message
    if report.budget.failing:
        failures += 1
        message = "; ".join(report.budget.messages) or "budget exceeded"
        failure = ET.SubElement(suite, "failure", type="budget", message=message)
        failure.text = message
    suite.set("failures", str(failures))
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="utf-8"?>\n{body}\n'


def _coverage_terminal(deltas: Sequence[CoverageDelta]) -> str:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120, color_system=None)

    table = Table(title="extract-regress coverage", show_lines=False)
    for column in ("field", "baseline", "current", "status"):
        table.add_column(column, overflow="fold")
    for delta in deltas:
        status = "[red]drop[/]" if delta.dropped else "[green]ok[/]"
        table.add_row(
            delta.path,
            f"{delta.baseline_fill_rate:.2f}",
            f"{delta.current_fill_rate:.2f}",
            status,
        )
    console.print(table)

    dropped = sum(1 for delta in deltas if delta.dropped)
    console.print(f"{len(deltas)} field(s) | {dropped} drop(s)")
    return buffer.getvalue()


def _coverage_markdown(deltas: Sequence[CoverageDelta]) -> str:
    dropped = sum(1 for delta in deltas if delta.dropped)
    lines: list[str] = [
        "## extract-regress coverage",
        "",
        f"{len(deltas)} field(s), {dropped} drop(s)",
        "",
    ]
    if deltas:
        lines.append("| Field | Baseline | Current | Status |")
        lines.append("| --- | --- | --- | --- |")
        for delta in deltas:
            status = "**drop**" if delta.dropped else "ok"
            lines.append(
                f"| `{delta.path}` | {delta.baseline_fill_rate:.2f} | "
                f"{delta.current_fill_rate:.2f} | {status} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _coverage_json(deltas: Sequence[CoverageDelta]) -> str:
    payload: dict[str, Any] = {
        "fields": len(deltas),
        "drops": sum(1 for delta in deltas if delta.dropped),
        "coverage": [
            {
                "path": delta.path,
                "baseline_fill_rate": delta.baseline_fill_rate,
                "current_fill_rate": delta.current_fill_rate,
                "dropped": delta.dropped,
            }
            for delta in deltas
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=str)


def render_coverage(deltas: Sequence[CoverageDelta], report_format: str = "term") -> str:
    """Render per-field coverage deltas in term, Markdown, or JSON form.

    A read-only view of fill-rate drift versus the committed baseline: each row
    is one field path with its baseline and current fill-rate and whether the
    field dropped beyond the configured threshold. Reuses :class:`CoverageDelta`
    and writes nothing to disk.
    """
    if report_format == "md":
        return _coverage_markdown(deltas)
    if report_format == "json":
        return _coverage_json(deltas)
    return _coverage_terminal(deltas)


def render_pr_comment(report: RunReport) -> str:
    """Render a compact, collapsible Markdown summary for a PR comment.

    Leads with a pass/fail line and the one-line summary, then a collapsed
    ``<details>`` table listing only the *failing* field diffs, followed by
    coverage-drop and budget sections. A passing run renders the short PASS
    form with no diff table, keeping green PRs quiet.
    """
    status = "PASS" if report.passed else "FAIL"
    emoji = "✅" if report.passed else "❌"
    lines: list[str] = [f"### extract-regress: {emoji} {status}", "", summary_line(report), ""]

    failing: list[tuple[str, FieldDiff]] = [
        (r.fixture_name, d) for r in report.results for d in r.failing_diffs
    ]
    if failing:
        lines.append("<details><summary>Failing field diffs</summary>")
        lines.append("")
        lines.append("| Fixture | Path | Kind | Golden | Actual | Reason |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for fixture_name, diff in failing:
            lines.append(
                "| "
                + " | ".join(
                    [
                        fixture_name,
                        f"`{diff.path}`",
                        diff.kind,
                        _md_cell(diff.golden),
                        _md_cell(diff.actual),
                        diff.reason,
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if report.dropped_coverage:
        lines.append("**Coverage drops**")
        lines.append("")
        for delta in report.dropped_coverage:
            lines.append(
                f"- `{delta.path}`: {delta.baseline_fill_rate:.2f} -> {delta.current_fill_rate:.2f}"
            )
        lines.append("")

    if report.budget.checked:
        lines.append("**Budget**")
        lines.append("")
        for message in report.budget.messages:
            lines.append(f"- {message}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
