"""Tests for project config loading from TOML (§3.2)."""

from __future__ import annotations

from pathlib import Path

from extract_regress.config import ERConfig, ProjectConfig, load_project_config


def _extract(source: object) -> dict[str, object]:  # pragma: no cover - trivial
    return {}


def test_defaults_when_no_config(tmp_path: Path) -> None:
    config = load_project_config(tmp_path)
    assert config.fixtures_dir == "tests/extract_fixtures"
    assert config.coverage_drop_threshold == 0.1
    assert config.tolerances.rules == ()
    assert not config.budget.enabled


def test_load_from_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.extract_regress]
fixtures_dir = "goldens"
coverage_drop_threshold = 0.2
judge_version = 3

[[tool.extract_regress.tolerances]]
path = "total"
abs_tol = 0.01

[[tool.extract_regress.tolerances]]
path = "vendor.*"
ignore_case = true

[tool.extract_regress.budget]
max_cost_usd_per_run = 0.5
max_p95_latency_ms = 2000
""",
        encoding="utf-8",
    )
    config = load_project_config(tmp_path)
    assert config.fixtures_dir == "goldens"
    assert config.coverage_drop_threshold == 0.2
    assert config.judge_version == 3
    assert len(config.tolerances.rules) == 2
    assert config.tolerances.resolve("total").abs_tol == 0.01  # type: ignore[union-attr]
    assert config.budget.max_cost_usd_per_run == 0.5
    assert config.budget.max_p95_latency_ms == 2000


def test_standalone_toml_takes_precedence(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.extract_regress]\nfixtures_dir = "from_pyproject"\n', encoding="utf-8"
    )
    (tmp_path / "extract_regress.toml").write_text(
        'fixtures_dir = "from_standalone"\n', encoding="utf-8"
    )
    assert load_project_config(tmp_path).fixtures_dir == "from_standalone"


def test_pyproject_without_table_uses_defaults(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.other]\nx = 1\n", encoding="utf-8")
    assert load_project_config(tmp_path).fixtures_dir == "tests/extract_fixtures"


def test_erconfig_from_project() -> None:
    project = ProjectConfig(fixtures_dir="g", judge_version=2)
    er = ERConfig.from_project(_extract, project)
    assert er.fixtures_dir == "g"
    assert er.judge_version == 2
    assert er.extract_fn is _extract
