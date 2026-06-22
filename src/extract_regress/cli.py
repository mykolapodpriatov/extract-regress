"""The ``extract-regress`` command-line interface (typer).

Commands mirror the pytest plugin options (plan §3.9):

* ``record`` — build goldens for fixtures lacking them + refresh the snapshot.
* ``run``    — replay + check; exits non-zero on regression or budget breach.
* ``update`` — accept current outputs as the new goldens + refresh the snapshot.
* ``report`` — render the last run in Markdown or terminal form.

The live :data:`ExtractFn` (and judge) cannot live in TOML, so the CLI loads an
:class:`ERConfig` from the user's ``conftest.py`` ``extract_regress_config()``
hook, exactly like the plugin. ``--config-module`` points at that module.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Annotated

import typer

from .config import ERConfig, load_project_config
from .report import render_markdown, render_terminal
from .runner import Runner
from .types import RunReport

app = typer.Typer(
    name="extract-regress",
    help="pytest for LLM extraction: pin golden extractions, catch silent drift.",
    no_args_is_help=True,
    add_completion=False,
)

CONFIG_HOOK = "extract_regress_config"


def _load_er_config(config_module: str, start_dir: Path) -> ERConfig:
    """Import ``config_module`` and call its ``extract_regress_config()`` hook.

    ``config_module`` may be a dotted import path or a filesystem path to a
    ``.py`` file (e.g. ``conftest.py``). The hook must return an
    :class:`ERConfig`.
    """
    module = _import_module(config_module, start_dir)
    hook = getattr(module, CONFIG_HOOK, None)
    if hook is None:
        raise typer.BadParameter(f"module {config_module!r} has no {CONFIG_HOOK}() hook")
    config = hook()
    if not isinstance(config, ERConfig):
        raise typer.BadParameter(
            f"{CONFIG_HOOK}() must return an ERConfig, got {type(config).__name__}"
        )
    return config


def _import_module(spec: str, start_dir: Path) -> object:
    candidate = (start_dir / spec).resolve()
    if candidate.suffix == ".py" and candidate.exists():
        # Put the file's own directory on the path first, so sibling imports
        # inside the config module (e.g. ``from fake_extractor import ...``)
        # resolve the same way they would under pytest. The entry is removed
        # again once the module has executed so we don't leak it across calls.
        parent = str(candidate.parent)
        added_to_path = parent not in sys.path
        if added_to_path:
            sys.path.insert(0, parent)
        # A unique module name (path digest) avoids clobbering a previously
        # imported ``conftest`` from a different project in ``sys.modules``.
        digest = hashlib.sha256(str(candidate).encode("utf-8")).hexdigest()[:12]
        mod_name = f"_extract_regress_cfg_{candidate.stem}_{digest}"
        mod_spec = importlib.util.spec_from_file_location(mod_name, candidate)
        if mod_spec is None or mod_spec.loader is None:  # pragma: no cover - defensive
            raise typer.BadParameter(f"cannot import {candidate}")
        module = importlib.util.module_from_spec(mod_spec)
        # The module is registered under its unique name only for the duration of
        # ``exec_module`` (so dataclasses/pickling that look it up resolve), then
        # popped: leaving it in ``sys.modules`` would grow unboundedly across
        # invocations and risk serving a stale module on a later import.
        sys.modules[mod_name] = module
        try:
            mod_spec.loader.exec_module(module)
        finally:
            sys.modules.pop(mod_name, None)
            if added_to_path:
                with contextlib.suppress(ValueError):  # pragma: no cover - defensive
                    sys.path.remove(parent)
        return module
    if str(start_dir) not in sys.path:
        sys.path.insert(0, str(start_dir))
    return importlib.import_module(spec)


def _resolve_config(
    config_module: str,
    fixtures_dir: str | None,
    project_dir: Path,
) -> ERConfig:
    er_config = _load_er_config(config_module, project_dir)
    project = load_project_config(project_dir)
    # TOML tolerances/budget take effect unless the hook already set them.
    if not er_config.tolerances.rules and project.tolerances.rules:
        er_config.tolerances = project.tolerances
    if not er_config.budget.enabled and project.budget.enabled:
        er_config.budget = project.budget
    if fixtures_dir is not None:
        er_config.fixtures_dir = fixtures_dir
    # Resolve a relative fixtures dir against the project dir, so the CLI works
    # regardless of the current working directory.
    fixtures_path = Path(er_config.fixtures_dir)
    if not fixtures_path.is_absolute():
        er_config.fixtures_dir = str((project_dir / fixtures_path).resolve())
    return er_config


def _warn_skipped(skipped: tuple[str, ...]) -> None:
    """Warn loudly when a record/update skipped errored fixtures.

    An errored extraction is deliberately not written as a golden (that would
    pin an empty ``{}`` and cause false failures forever), so the user is told
    which fixtures were left untouched and why.
    """
    if skipped:
        typer.echo(
            f"skipped {len(skipped)} errored fixture(s) (not recorded): {', '.join(skipped)}",
            err=True,
        )


_LAST_REPORT: RunReport | None = None


ConfigModuleOpt = Annotated[
    str,
    typer.Option("--config-module", "-c", help="Module/path exposing extract_regress_config()."),
]
ProjectDirOpt = Annotated[
    Path,
    typer.Option("--project-dir", help="Directory to resolve config and TOML from."),
]
FixturesDirOpt = Annotated[
    str | None,
    typer.Option("--fixtures-dir", help="Override the fixtures directory."),
]


@app.command()
def record(
    config_module: ConfigModuleOpt = "conftest.py",
    project_dir: ProjectDirOpt = Path(),
    fixtures_dir: FixturesDirOpt = None,
) -> None:
    """Build goldens for fixtures lacking them and refresh the snapshot."""
    config = _resolve_config(config_module, fixtures_dir, project_dir.resolve())
    runner = Runner(config)
    written = runner.record()
    if written:
        typer.echo(f"recorded {len(written)} fixture(s): {', '.join(written)}")
    else:
        typer.echo("no fixtures needed recording (all goldens present)")
    _warn_skipped(runner.last_skipped)


@app.command()
def update(
    config_module: ConfigModuleOpt = "conftest.py",
    project_dir: ProjectDirOpt = Path(),
    fixtures_dir: FixturesDirOpt = None,
) -> None:
    """Accept current outputs as the new goldens and refresh the snapshot."""
    config = _resolve_config(config_module, fixtures_dir, project_dir.resolve())
    runner = Runner(config)
    written = runner.update()
    typer.echo(f"updated {len(written)} golden(s): {', '.join(written)}")
    _warn_skipped(runner.last_skipped)


@app.command()
def run(
    config_module: ConfigModuleOpt = "conftest.py",
    project_dir: ProjectDirOpt = Path(),
    fixtures_dir: FixturesDirOpt = None,
    budget: Annotated[bool, typer.Option(help="Enforce cost/latency budgets.")] = True,
    report_format: Annotated[
        str, typer.Option("--format", help="Output format: term or md.")
    ] = "term",
) -> None:
    """Replay fixtures and check; exit non-zero on regression or budget breach."""
    global _LAST_REPORT
    config = _resolve_config(config_module, fixtures_dir, project_dir.resolve())
    report = Runner(config).run(check_budget=budget)
    _LAST_REPORT = report

    if report_format == "md":
        typer.echo(render_markdown(report))
    else:
        typer.echo(render_terminal(report, color=False))

    raise typer.Exit(code=0 if report.passed else 1)


@app.command()
def report(
    config_module: ConfigModuleOpt = "conftest.py",
    project_dir: ProjectDirOpt = Path(),
    fixtures_dir: FixturesDirOpt = None,
    report_format: Annotated[
        str, typer.Option("--format", help="Output format: term or md.")
    ] = "term",
) -> None:
    """Render the most recent run.

    For a fresh process with no in-memory run, this performs a read-only replay
    and renders it without changing the exit code semantics of ``run``.
    """
    config = _resolve_config(config_module, fixtures_dir, project_dir.resolve())
    current = _LAST_REPORT if _LAST_REPORT is not None else Runner(config).run()
    if report_format == "md":
        typer.echo(render_markdown(current))
    else:
        typer.echo(render_terminal(current, color=False))


if __name__ == "__main__":  # pragma: no cover
    app()
