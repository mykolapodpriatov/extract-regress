"""Terminal (rich), Markdown, JSON, and PR-comment renderers for a :class:`RunReport`."""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

from rich.console import Console
from rich.table import Table

from .types import FieldDiff, RunReport

__all__ = [
    "render_json",
    "render_markdown",
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
