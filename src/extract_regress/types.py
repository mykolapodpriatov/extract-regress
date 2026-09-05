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

from .quarantine import QuarantineRule

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
    #: The quarantine entry covering this fixture, if any. A quarantined
    #: fixture still runs and still diffs; it just does not fail the build.
    quarantine: QuarantineRule | None = None
    #: Whether that entry had run out when the run happened. Resolved by the
    #: runner against the run date, so the report stays a pure value.
    quarantine_expired: bool = False

    @property
    def failing_diffs(self) -> tuple[FieldDiff, ...]:
        """The non-tolerated diffs for this fixture."""
        return tuple(d for d in self.diffs if d.failing)

    @property
    def passed(self) -> bool:
        """A fixture passes iff it had no error and no failing diffs.

        Unchanged by quarantine: this says what the fixture did, not what the
        build should do about it.
        """
        return self.error is None and not self.failing_diffs

    @property
    def quarantined(self) -> bool:
        """Whether a live quarantine entry covers this fixture."""
        return self.quarantine is not None and not self.quarantine_expired

    @property
    def stale_quarantine(self) -> bool:
        """A quarantined fixture that passed, so the entry should be removed.

        This fails the build. A stale entry silently swallows the regression
        when it comes back, which is worse than having no quarantine at all.
        """
        return self.quarantine is not None and self.passed

    @property
    def blocking(self) -> bool:
        """Whether this result should fail the build.

        A failing fixture blocks unless a live quarantine covers it. A passing
        one blocks only when it is still quarantined, and an expired entry
        blocks whatever the fixture did.
        """
        if self.quarantine is None:
            return not self.passed
        if self.quarantine_expired:
            return True
        return self.stale_quarantine


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
        """Fixtures that failed (error or non-tolerated diff).

        Includes quarantined ones: this reports what happened, and
        :attr:`blocking_results` reports what it costs.
        """
        return [r for r in self.results if not r.passed]

    @property
    def quarantined_results(self) -> list[FixtureResult]:
        """Failing fixtures a live quarantine entry is absorbing."""
        return [r for r in self.results if r.quarantined and not r.passed]

    @property
    def stale_quarantines(self) -> list[FixtureResult]:
        """Quarantined fixtures that passed, so their entries should go."""
        return [r for r in self.results if r.stale_quarantine]

    @property
    def expired_quarantines(self) -> list[FixtureResult]:
        """Fixtures whose quarantine entry has run out."""
        return [r for r in self.results if r.quarantine is not None and r.quarantine_expired]

    @property
    def blocking_results(self) -> list[FixtureResult]:
        """Results that should fail the build."""
        return [r for r in self.results if r.blocking]

    @property
    def dropped_coverage(self) -> list[CoverageDelta]:
        """Coverage deltas flagged as a meaningful fill-rate drop."""
        return [c for c in self.coverage_deltas if c.dropped]

    @property
    def passed(self) -> bool:
        """Overall pass/fail for the run.

        Fails if any fixture blocks, any coverage fill-rate dropped beyond
        threshold, or the budget check failed. A failure a live quarantine
        entry covers does not block; a quarantine that has expired, or that
        covers a fixture which now passes, does.
        """
        return not self.blocking_results and not self.dropped_coverage and not self.budget.failing
