"""extract-regress: pytest for LLM extraction.

Pin golden ``(source -> expected JSON)`` fixtures for an extraction function and
replay them in CI, comparing with a type-aware semantic field diff (dates,
numbers, casing, list ordering, near-duplicate strings) instead of brittle string
equality. Detects model/prompt drift and source-format coverage drift, enforces
cost/latency budgets, and offers an optional cached LLM-judge for free-text
fields.

Public entry points:

* :func:`case` — decorator sugar for binding a config to one fixture.
* :class:`ERConfig` — the runtime configuration returned by the conftest hook.
* :class:`Runner` — programmatic run orchestration.
* :func:`diff_extraction` — the field diff engine.
"""

from __future__ import annotations

from .budget import BudgetConfig, evaluate_budget
from .config import ERConfig, ProjectConfig, load_project_config
from .coverage import compute_fill_rates, diff_coverage
from .diff import diff_extraction
from .fixtures import Fixture, FixtureStore, schema_hash
from .judge import CachedJudge, JudgeCache, make_judge
from .pytest_plugin import case
from .runner import Mode, Runner
from .tolerances import ToleranceConfig, ToleranceRule
from .types import (
    BudgetOutcome,
    CoverageDelta,
    ExtractFn,
    ExtractInput,
    ExtractionResult,
    FieldDiff,
    FixtureResult,
    RunReport,
    Usage,
)

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022 - grouped by concern for readability, not sorted
    "__version__",
    # config / registration
    "ERConfig",
    "ProjectConfig",
    "load_project_config",
    "case",
    # orchestration
    "Runner",
    "Mode",
    # fixtures
    "Fixture",
    "FixtureStore",
    "schema_hash",
    # diff
    "diff_extraction",
    "ToleranceConfig",
    "ToleranceRule",
    # coverage
    "compute_fill_rates",
    "diff_coverage",
    # budget
    "BudgetConfig",
    "evaluate_budget",
    # judge
    "CachedJudge",
    "JudgeCache",
    "make_judge",
    # core types
    "ExtractFn",
    "ExtractInput",
    "ExtractionResult",
    "Usage",
    "FieldDiff",
    "FixtureResult",
    "CoverageDelta",
    "BudgetOutcome",
    "RunReport",
]
