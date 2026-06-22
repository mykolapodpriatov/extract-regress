"""Project configuration loading (plan §3.2, §3.9).

Configuration is read from ``[tool.extract_regress]`` in ``pyproject.toml`` or
from a standalone ``extract_regress.toml`` (the latter wins when both exist).
The :class:`ERConfig` returned by a user's ``extract_regress_config()`` conftest
hook supplies the live :data:`ExtractFn` and judge; the TOML supplies the static
surface (paths, tolerances, thresholds).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .budget import BudgetConfig
from .coverage import DEFAULT_DROP_THRESHOLD
from .tolerances import ToleranceConfig, ToleranceRule
from .types import ExtractFn, JudgeFn

DEFAULT_FIXTURES_DIR = "tests/extract_fixtures"


class ProjectConfig(BaseModel):
    """Static, file-backed configuration for a project."""

    model_config = ConfigDict(frozen=True)

    fixtures_dir: str = DEFAULT_FIXTURES_DIR
    coverage_drop_threshold: float = DEFAULT_DROP_THRESHOLD
    tolerances: ToleranceConfig = Field(default_factory=ToleranceConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    judge_version: int = 1

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ProjectConfig:
        """Build a config from a parsed ``[tool.extract_regress]`` table."""
        payload = dict(data)
        rules = payload.pop("tolerances", []) or []
        budget = payload.pop("budget", {}) or {}
        tol_config = ToleranceConfig(rules=tuple(ToleranceRule(**rule) for rule in rules))
        return cls(
            tolerances=tol_config,
            budget=BudgetConfig(**budget),
            **payload,
        )


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_project_config(start_dir: Path | str | None = None) -> ProjectConfig:
    """Discover and load project configuration.

    Searches ``start_dir`` (default: cwd) for ``extract_regress.toml`` first,
    then ``[tool.extract_regress]`` in ``pyproject.toml``. Returns defaults when
    neither is present.
    """
    base = Path(start_dir) if start_dir is not None else Path.cwd()

    standalone = base / "extract_regress.toml"
    if standalone.exists():
        return ProjectConfig.from_mapping(_read_toml(standalone))

    pyproject = base / "pyproject.toml"
    if pyproject.exists():
        data = _read_toml(pyproject)
        table = data.get("tool", {}).get("extract_regress")
        if table is not None:
            return ProjectConfig.from_mapping(table)

    return ProjectConfig()


@dataclass
class ERConfig:
    """The full runtime configuration returned by the conftest hook (§3.8).

    Combines the live callables (which cannot live in TOML) with the static
    project configuration. ``extract_fn`` is mandatory; the rest default to a
    sensible empty configuration.
    """

    extract_fn: ExtractFn
    fixtures_dir: str = DEFAULT_FIXTURES_DIR
    tolerances: ToleranceConfig = field(default_factory=ToleranceConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    coverage_drop_threshold: float = DEFAULT_DROP_THRESHOLD
    judge_fn: JudgeFn | None = None
    judge_version: int = 1

    @classmethod
    def from_project(
        cls,
        extract_fn: ExtractFn,
        project: ProjectConfig,
        *,
        judge_fn: JudgeFn | None = None,
    ) -> ERConfig:
        """Merge a live ``extract_fn`` with a static :class:`ProjectConfig`."""
        return cls(
            extract_fn=extract_fn,
            fixtures_dir=project.fixtures_dir,
            tolerances=project.tolerances,
            budget=project.budget,
            coverage_drop_threshold=project.coverage_drop_threshold,
            judge_fn=judge_fn,
            judge_version=project.judge_version,
        )
