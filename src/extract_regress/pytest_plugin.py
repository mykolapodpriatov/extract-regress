"""pytest plugin: hooks, fixtures, options, and registration (plan §3.8).

Registration happens through an ``extract_regress_config()`` hook in the user's
``conftest.py`` returning an :class:`ERConfig`. The plugin discovers it and
**auto-collects one test item per fixture** (no per-fixture boilerplate), runs
the runner, and asserts no non-tolerated diffs. The ``@extract_regress.case(...)``
decorator is offered as sugar and takes precedence over the conftest hook for the
fixtures it annotates; the two coexist.

Options:
    ``--er-record``          write/refresh goldens, then skip assertions
    ``--er-update``          update goldens (overwrite), then skip assertions
    ``--er-budget`` / ``--no-er-budget``   toggle budget enforcement
    ``--er-report=md:PATH``  also write a Markdown report to PATH
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from . import coverage as coverage_mod
from .config import ERConfig
from .diff import diff_extraction
from .fixtures import Fixture, FixtureStore
from .report import render_markdown, render_terminal
from .runner import Mode, Runner, normalize_result
from .types import FixtureResult, RunReport

CONFIG_HOOK = "extract_regress_config"


# ---------------------------------------------------------------------------
# @extract_regress.case decorator (sugar; precedence over the conftest hook)
# ---------------------------------------------------------------------------


@dataclass
class _CaseOverride:
    """A per-fixture override registered via :func:`case`."""

    fixture_name: str
    config: ERConfig


_CASE_REGISTRY: dict[str, _CaseOverride] = {}


def case(
    fixture_name: str, *, config: ERConfig
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bind an :class:`ERConfig` to a single fixture, overriding the hook.

    Usage in a test module::

        import extract_regress

        @extract_regress.case("invoice_basic", config=my_config)
        def test_invoice_basic() -> None:
            ...

    The decorated function is left intact; the binding is consulted by the
    plugin when it builds the per-fixture items.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        existing = _CASE_REGISTRY.get(fixture_name)
        if existing is not None and existing.config is not config:
            raise ValueError(
                f"fixture {fixture_name!r} is already registered with a different "
                "ERConfig via @extract_regress.case; a fixture may bind to at most "
                "one config. Remove the duplicate decorator or reuse the same config."
            )
        _CASE_REGISTRY[fixture_name] = _CaseOverride(fixture_name, config)
        return func

    return decorator


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--er-*`` command-line options."""
    group = parser.getgroup("extract_regress")
    group.addoption(
        "--er-record",
        action="store_true",
        default=False,
        help="Write/refresh goldens for fixtures lacking them, then skip assertions.",
    )
    group.addoption(
        "--er-update",
        action="store_true",
        default=False,
        help="Update (overwrite) goldens from current outputs, then skip assertions.",
    )
    group.addoption(
        "--er-budget",
        action="store_true",
        default=True,
        dest="er_budget",
        help="Enforce cost/latency budgets (default).",
    )
    group.addoption(
        "--no-er-budget",
        action="store_false",
        dest="er_budget",
        help="Disable cost/latency budget enforcement.",
    )
    group.addoption(
        "--er-report",
        action="store",
        default=None,
        metavar="md:PATH",
        help="Write a report (currently 'md:PATH') in addition to terminal output.",
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_hook_config(config: pytest.Config) -> ERConfig | None:
    """Resolve the conftest ``extract_regress_config()`` hook, if present."""
    for plugin in config.pluginmanager.get_plugins():
        candidate = getattr(plugin, CONFIG_HOOK, None)
        if callable(candidate):
            result = candidate()
            if isinstance(result, ERConfig):
                return result
    return None


def _mode(config: pytest.Config) -> Mode:
    if config.getoption("er_update"):
        return Mode.UPDATE
    if config.getoption("er_record"):
        return Mode.RECORD
    return Mode.RUN


def _config_for(fixture_name: str, hook_config: ERConfig | None) -> ERConfig | None:
    """Decorator override wins over the conftest hook for annotated fixtures."""
    override = _CASE_REGISTRY.get(fixture_name)
    if override is not None:
        return override.config
    return hook_config


# ---------------------------------------------------------------------------
# Auto-collection: one item per fixture, no user boilerplate
# ---------------------------------------------------------------------------


def _resolve_fixtures_dirs(hook_config: ERConfig | None) -> list[str]:
    """All distinct ``fixtures_dir`` values to scan for auto-collected fixtures.

    The conftest hook contributes its directory, and every ``@case``-registered
    override contributes its own. Scanning *all* of them (deduplicated, in a
    stable order) ensures fixtures living under a non-default directory are never
    silently skipped just because another directory was registered first.
    """
    dirs: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            dirs.append(value)

    if hook_config is not None:
        add(hook_config.fixtures_dir)
    for override in _CASE_REGISTRY.values():
        add(override.config.fixtures_dir)
    return dirs


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Inject one :class:`ExtractRegressItem` per fixture into the test run."""
    hook_config = _discover_hook_config(config)
    fixtures_dirs = _resolve_fixtures_dirs(hook_config)
    if not fixtures_dirs:
        return

    parent = ExtractRegressCollector.from_parent(session, name="extract_regress")
    seen_names: set[str] = set()
    for fixtures_dir in fixtures_dirs:
        store = FixtureStore(fixtures_dir)
        for fixture in store.load_all():
            er_config = _config_for(fixture.name, hook_config)
            if er_config is None:
                continue
            # A fixture name reachable from several registered dirs is collected
            # once, against the first directory that yields it.
            if fixture.name in seen_names:
                continue
            seen_names.add(fixture.name)
            items.append(
                ExtractRegressItem.from_parent(
                    parent,
                    name=fixture.name,
                    fixture=fixture,
                    er_config=er_config,
                )
            )


class ExtractRegressCollector(pytest.Collector):
    """Virtual parent node grouping the auto-generated fixture items."""

    def collect(self) -> list[pytest.Item]:  # pragma: no cover - not used directly
        return []


class ExtractRegressItem(pytest.Item):
    """A single fixture's replay-and-assert test item."""

    def __init__(
        self,
        *,
        name: str,
        parent: pytest.Collector,
        fixture: Fixture,
        er_config: ERConfig,
    ) -> None:
        super().__init__(name, parent)
        self._fixture = fixture
        self._er_config = er_config

    def runtest(self) -> None:
        run_fixture(self.config, self._fixture, self._er_config)

    def reportinfo(self) -> tuple[str | Path, int | None, str]:
        return self._er_config.fixtures_dir, 0, f"extract-regress: {self.name}"

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException], style: Any = None) -> Any:
        # Render our own assertion text plainly; defer everything else to pytest.
        if isinstance(excinfo.value, ExtractRegressFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)


class ExtractRegressFailure(AssertionError):
    """Raised when a fixture replay produces a non-tolerated diff or error."""


# ---------------------------------------------------------------------------
# The per-fixture check (also callable directly, used by the CLI/tests)
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    """Per-session accumulator for the report and snapshot side-effects.

    Collecting results here lets us write the Markdown report exactly once at
    session end (instead of re-appending it per fixture, which duplicated the
    header/table N times) and refresh the coverage snapshot once per run rather
    than once per fixture (O(N) instead of O(N²) disk I/O).
    """

    results: list[FixtureResult] = field(default_factory=list)
    er_configs: dict[str, ERConfig] = field(default_factory=dict)
    did_write: bool = False
    # Fixtures whose extraction errored during record/update. They were never
    # written, so the refreshed coverage snapshot must exclude them even though a
    # stale golden may still exist on disk (consistent with run-mode exclusion).
    skipped: set[str] = field(default_factory=set)


_SESSION_KEY = pytest.StashKey[_SessionState]()


def _session_state(config: pytest.Config) -> _SessionState:
    state = config.stash.get(_SESSION_KEY, None)
    if state is None:
        state = _SessionState()
        config.stash[_SESSION_KEY] = state
    return state


def run_fixture(config: pytest.Config, fixture: Fixture, er_config: ERConfig) -> None:
    """Replay a single fixture and assert no failing diffs (or record/update).

    Per-fixture results are accumulated on the session; the aggregate report and
    coverage-snapshot refresh are emitted once at :func:`pytest_sessionfinish`.
    """
    store = FixtureStore(er_config.fixtures_dir)
    fresh = store.load(fixture.name)
    state = _session_state(config)
    # Remember a config per fixtures_dir so the snapshot can be refreshed once.
    state.er_configs.setdefault(er_config.fixtures_dir, er_config)

    raw = er_config.extract_fn(fresh.resolve_source())
    result = normalize_result(raw)

    mode = _mode(config)
    if mode in (Mode.RECORD, Mode.UPDATE):
        if result.error is not None:
            # Never pin an errored extraction (an empty ``{}``) as the golden:
            # that would permanently corrupt the fixture and FAIL forever.
            # Record the failure in the session results (so it appears in the
            # report) and mark the fixture skipped (so the refreshed coverage
            # snapshot excludes it even if a stale golden lingers on disk).
            state.skipped.add(fresh.name)
            state.results.append(FixtureResult(fixture_name=fresh.name, error=result.error))
            raise ExtractRegressFailure(
                f"refusing to record {fresh.name!r}: extraction errored "
                f"({result.error}); fix the extractor and re-record"
            )
        if mode is Mode.UPDATE or not fresh.has_golden():
            store.save(fresh.model_copy(update={"expected": result.value}))
            state.did_write = True
        return

    if result.error is not None:
        state.results.append(FixtureResult(fixture_name=fresh.name, error=result.error))
        report = RunReport(results=(FixtureResult(fixture_name=fresh.name, error=result.error),))
        raise ExtractRegressFailure(
            f"extraction error for {fresh.name!r}: {result.error}\n"
            + render_terminal(report, color=False)
        )

    diffs = diff_extraction(
        fresh.expected,
        result.value,
        er_config.tolerances,
        judge_fn=er_config.judge_fn,
    )
    fixture_result = FixtureResult(fixture_name=fresh.name, diffs=tuple(diffs))
    state.results.append(fixture_result)
    report = RunReport(results=(fixture_result,))

    failing = [d for d in diffs if d.failing]
    if failing:
        raise ExtractRegressFailure(
            f"{len(failing)} non-tolerated diff(s) for {fresh.name!r}:\n"
            + render_terminal(report, color=False)
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the report once and refresh the coverage snapshot once per run."""
    config = session.config
    state = config.stash.get(_SESSION_KEY, None)
    if state is None:
        # No fixtures were collected (empty session). A requested report must
        # still be written as a valid, empty :class:`RunReport` rather than
        # silently skipped; never crash here.
        if config.getoption("er_report"):
            _write_report_if_requested(config, render_markdown(RunReport()))
        return

    report = RunReport(results=tuple(state.results))
    _write_report_if_requested(config, render_markdown(report))

    # Refresh the coverage snapshot a single time after a record/update pass.
    if state.did_write:
        for er_config in state.er_configs.values():
            _refresh_snapshot(er_config, state.skipped)


def _write_report_if_requested(config: pytest.Config, markdown: str) -> None:
    spec = config.getoption("er_report")
    if not spec:
        return
    if spec.startswith("md:"):
        path = Path(spec[len("md:") :])
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write once (overwrite), not append: the report is the whole run.
        path.write_text(markdown, encoding="utf-8")


def _refresh_snapshot(er_config: ERConfig, skipped: set[str]) -> None:
    """Recompute the corpus coverage snapshot after a record/update pass.

    Reads the on-disk goldens. Fixtures that errored during this pass are
    excluded by name: although a stale golden may still exist on disk for them,
    counting it would skew the fill-rate sample with data the run could not
    reproduce, inconsistent with how run-mode drops errored extractions.
    """
    runner = Runner(er_config)
    fixtures = runner.store.load_all()
    extractions = [f.expected for f in fixtures if f.has_golden() and f.name not in skipped]
    fill_rates = coverage_mod.compute_fill_rates(extractions)
    coverage_mod.write_baseline(er_config.fixtures_dir, fill_rates)


__all__ = ["ExtractRegressItem", "case", "run_fixture"]
