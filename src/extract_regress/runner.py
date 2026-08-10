"""Run orchestration: load fixtures → call ExtractFn → collect results.

The runner is the one place that normalizes a :data:`ExtractFn` return (bare
``dict`` or :class:`ExtractionResult`) into a uniform :class:`ExtractionResult`,
so every downstream stage (diff, coverage, budget) sees a single type. It also
supports the ``record`` / ``update`` write modes and the read-only ``run`` mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from . import coverage as coverage_mod
from .budget import evaluate_budget
from .config import ERConfig
from .diff import diff_extraction
from .fixtures import Fixture, FixtureError, FixtureStore
from .types import (
    BudgetOutcome,
    ExtractionResult,
    FixtureResult,
    RunReport,
    Usage,
)


class Mode(StrEnum):
    """Execution mode for a run."""

    RUN = "run"
    """Read-only replay + checks (the CI path)."""
    RECORD = "record"
    """Fill goldens for fixtures lacking them; refresh the snapshot."""
    UPDATE = "update"
    """Accept current outputs as new goldens; refresh the snapshot."""


def normalize_result(raw: dict[str, Any] | ExtractionResult) -> ExtractionResult:
    """Coerce an :data:`ExtractFn` return into an :class:`ExtractionResult`.

    A bare ``dict`` is wrapped with an empty :class:`Usage` (so budgets are
    skipped for that call); an :class:`ExtractionResult` is passed through.
    """
    if isinstance(raw, ExtractionResult):
        return raw
    if isinstance(raw, Mapping):
        return ExtractionResult(value=dict(raw), usage=Usage())
    raise TypeError(f"extract_fn must return dict or ExtractionResult, got {type(raw).__name__}")


class Runner:
    """Drives a run over all fixtures for a given :class:`ERConfig`."""

    def __init__(self, config: ERConfig) -> None:
        self.config = config
        self.store = FixtureStore(config.fixtures_dir)
        # Names of fixtures whose golden write was skipped (last record/update).
        self.last_skipped: tuple[str, ...] = ()

    # -- extraction --------------------------------------------------------

    def _extract(self, fixture: Fixture) -> ExtractionResult:
        source = fixture.resolve_source()
        raw = self.config.extract_fn(source)
        return normalize_result(raw)

    # -- public API --------------------------------------------------------

    def run(self, *, check_budget: bool = True, names: Sequence[str] | None = None) -> RunReport:
        """Replay every fixture and produce a :class:`RunReport` (read-only).

        ``names``, if given, limits the replay to fixtures with an exact name
        match (see :meth:`_select`); every other fixture on disk is left alone.
        """
        fixtures = self._select(self.store.load_all(), names)
        results: list[FixtureResult] = []
        usages: list[Usage] = []
        current_extractions: list[dict[str, Any]] = []

        for fixture in fixtures:
            extraction = self._extract(fixture)
            usages.append(extraction.usage)

            if extraction.error is not None:
                # An errored extraction yields ``{}``; counting it in the
                # coverage sample would skew every field toward 0 and raise
                # spurious coverage-drop alerts, so it is excluded.
                results.append(FixtureResult(fixture_name=fixture.name, error=extraction.error))
                continue

            current_extractions.append(extraction.value)

            diffs = diff_extraction(
                fixture.expected,
                extraction.value,
                self.config.tolerances,
                judge_fn=self.config.judge_fn,
            )
            results.append(FixtureResult(fixture_name=fixture.name, diffs=tuple(diffs)))

        coverage_deltas = self._coverage_deltas(current_extractions)
        budget = self._budget(usages) if check_budget else BudgetOutcome(checked=False)

        return RunReport(
            results=tuple(results),
            coverage_deltas=tuple(coverage_deltas),
            budget=budget,
        )

    def record(self, *, overwrite: bool = False, names: Sequence[str] | None = None) -> list[str]:
        """Fill goldens (idempotent) and rewrite the coverage snapshot.

        With ``overwrite`` false (the ``record`` semantics), only fixtures
        lacking a golden are populated; with it true the call behaves like
        ``update``. Returns the names of fixtures whose golden was written.

        An extraction that reports an error is never written as a golden (that
        would pin an empty ``{}`` and produce false FAILs forever) and never
        contributes to the coverage sample; such fixtures are skipped and their
        names collected in ``skipped`` for the caller to surface.

        ``names``, if given, limits *extraction* (and therefore writing) to
        fixtures with an exact name match (see :meth:`_select`). Fixtures
        outside the filter are never extracted, but a fixture that already has
        a golden still contributes it to the coverage sample, so the refreshed
        snapshot reflects the full on-disk golden set rather than just the
        filtered subset.
        """
        fixtures = self.store.load_all()
        selected_names = {f.name for f in self._select(fixtures, names)}
        written: list[str] = []
        skipped: list[str] = []
        sampled: list[dict[str, Any]] = []

        for fixture in fixtures:
            if fixture.name not in selected_names:
                if fixture.has_golden():
                    sampled.append(fixture.expected)
                continue

            extraction = self._extract(fixture)

            if extraction.error is not None:
                # Do not corrupt the golden or skew coverage with an errored run.
                skipped.append(fixture.name)
                continue

            if overwrite or not fixture.has_golden():
                updated = fixture.model_copy(update={"expected": extraction.value})
                self.store.save(updated)
                written.append(fixture.name)
                sampled.append(extraction.value)
            else:
                # An already-recorded golden still contributes to coverage.
                sampled.append(fixture.expected)

        self.last_skipped = tuple(skipped)
        self._refresh_baseline(sampled)
        return written

    def update(self, *, names: Sequence[str] | None = None) -> list[str]:
        """Accept current outputs as the new goldens; refresh the snapshot."""
        return self.record(overwrite=True, names=names)

    # -- helpers -----------------------------------------------------------

    def _select(self, fixtures: list[Fixture], names: Sequence[str] | None) -> list[Fixture]:
        """Filter ``fixtures`` down to an exact-name match against ``names``.

        ``names`` is ``None`` or empty returns ``fixtures`` unchanged. Every
        given name must match at least one loaded fixture; an unmatched name
        raises :class:`FixtureError` instead of silently running zero
        fixtures.
        """
        if not names:
            return fixtures
        by_name = {fixture.name: fixture for fixture in fixtures}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise FixtureError(f"no fixture(s) matching name(s): {', '.join(missing)}")
        wanted = set(names)
        return [fixture for fixture in fixtures if fixture.name in wanted]

    def _coverage_deltas(self, extractions: list[dict[str, Any]]) -> list[Any]:
        baseline = coverage_mod.load_baseline(self.config.fixtures_dir)
        if not baseline:
            return []
        current = coverage_mod.compute_fill_rates(extractions)
        return coverage_mod.diff_coverage(
            baseline,
            current,
            drop_threshold=self.config.coverage_drop_threshold,
        )

    def _refresh_baseline(self, extractions: list[dict[str, Any]]) -> None:
        fill_rates = coverage_mod.compute_fill_rates(extractions)
        coverage_mod.write_baseline(self.config.fixtures_dir, fill_rates)

    def _budget(self, usages: list[Usage]) -> BudgetOutcome:
        return evaluate_budget(usages, self.config.budget)
