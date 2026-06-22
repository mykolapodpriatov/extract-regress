"""Core types shared across :mod:`extract_regress`.

These are the load-bearing data structures: the extraction contract
(:data:`ExtractFn` / :class:`ExtractionResult`), the per-field diff
(:class:`FieldDiff`), and the aggregate :class:`RunReport`. Everything
downstream of the runner only ever sees :class:`ExtractionResult`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# A source can be raw text, raw bytes, or a path to a file on disk.
ExtractInput = str | bytes | Path

DiffKind = Literal["changed", "added", "removed", "type_changed"]
"""Structural classification of a field-level change."""


class Usage(BaseModel):
    """Per-call provider usage.

    Every field is optional: a user who does not wrap a provider simply
    returns a bare ``dict`` from their :data:`ExtractFn`, and budgets are
    skipped for that call. ``latency_ms`` and ``cost_usd`` feed the budget
    engine (:mod:`extract_regress.budget`).
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None

    @property
    def has_cost(self) -> bool:
        """Whether this record contributes to the cost budget."""
        return self.cost_usd is not None

    @property
    def has_latency(self) -> bool:
        """Whether this record contributes to the latency budget."""
        return self.latency_ms is not None


class ExtractionResult(BaseModel):
    """The normalized result of a single extraction call.

    The runner coerces a bare ``dict`` return into this shape with an empty
    :class:`Usage`, so every downstream stage can rely on a uniform type.
    """

    model_config = ConfigDict(frozen=True)

    value: dict[str, Any] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


@runtime_checkable
class ExtractFn(Protocol):
    """The extraction callable contract.

    An implementation accepts an :data:`ExtractInput` and returns either a
    plain ``dict`` (the extracted JSON; usage unknown) or a fully populated
    :class:`ExtractionResult` (value plus usage/error). The runner normalizes
    both into :class:`ExtractionResult`.
    """

    def __call__(self, source: ExtractInput) -> dict[str, Any] | ExtractionResult:
        """Extract structured data from ``source``."""
        ...


# A judge callable returns ``(verdict, resolved_model_id)``; see
# :mod:`extract_regress.judge` for the bootstrapping contract.
JudgeFn = Callable[[str, str, str], tuple[bool, str]]


class FieldDiff(BaseModel):
    """A single resolved field-level difference between golden and actual."""

    model_config = ConfigDict(frozen=True)

    path: str
    kind: DiffKind
    golden: Any = None
    actual: Any = None
    tolerated: bool = False
    reason: str = ""

    @property
    def failing(self) -> bool:
        """A diff fails the run iff it is not tolerated."""
        return not self.tolerated


class CoverageDelta(BaseModel):
    """Per-field fill-rate change between the baseline snapshot and this run."""

    model_config = ConfigDict(frozen=True)

    path: str
    baseline_fill_rate: float
    current_fill_rate: float
    dropped: bool

    @property
    def delta(self) -> float:
        """Signed change in fill-rate (current minus baseline)."""
        return self.current_fill_rate - self.baseline_fill_rate


class BudgetOutcome(BaseModel):
    """Result of evaluating cost/latency thresholds for a run."""

    model_config = ConfigDict(frozen=True)

    checked: bool = False
    passed: bool = True
    total_cost_usd: float | None = None
    p95_latency_ms: float | None = None
    max_cost_usd: float | None = None
    max_p95_latency_ms: float | None = None
    messages: tuple[str, ...] = ()

    @property
    def failing(self) -> bool:
        """Whether the budget check ran and failed."""
        return self.checked and not self.passed


class FixtureResult(BaseModel):
    """Per-fixture outcome: the diffs found and any extraction error."""

    model_config = ConfigDict(frozen=True)

    fixture_name: str
    diffs: tuple[FieldDiff, ...] = ()
    error: str | None = None

    @property
    def failing_diffs(self) -> tuple[FieldDiff, ...]:
        """The non-tolerated diffs for this fixture."""
        return tuple(d for d in self.diffs if d.failing)

    @property
    def passed(self) -> bool:
        """A fixture passes iff it had no error and no failing diffs."""
        return self.error is None and not self.failing_diffs


class RunReport(BaseModel):
    """Aggregate report for a full run across all fixtures."""

    model_config = ConfigDict(frozen=True)

    results: tuple[FixtureResult, ...] = ()
    coverage_deltas: tuple[CoverageDelta, ...] = ()
    budget: BudgetOutcome = Field(default_factory=BudgetOutcome)

    @property
    def all_diffs(self) -> list[FieldDiff]:
        """Flattened list of every field diff across all fixtures."""
        return [d for r in self.results for d in r.diffs]

    @property
    def failing_results(self) -> list[FixtureResult]:
        """Fixtures that failed (error or non-tolerated diff)."""
        return [r for r in self.results if not r.passed]

    @property
    def dropped_coverage(self) -> list[CoverageDelta]:
        """Coverage deltas flagged as a meaningful fill-rate drop."""
        return [c for c in self.coverage_deltas if c.dropped]

    @property
    def passed(self) -> bool:
        """Overall pass/fail for the run.

        Fails if any fixture failed, any coverage fill-rate dropped beyond
        threshold, or the budget check failed.
        """
        return not self.failing_results and not self.dropped_coverage and not self.budget.failing
