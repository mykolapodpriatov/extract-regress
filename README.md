# extract-regress

> pytest for LLM extraction — pin golden extractions and catch silent regressions when the prompt, model, schema, or source format drifts.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

A pytest plugin + CLI that records golden `(source -> expected JSON)` fixtures for LLM data-extraction pipelines and replays them in CI. Comparisons use a type-aware semantic field diff (dates, numbers, casing, list ordering, near-duplicate strings) instead of brittle string equality, so you catch real regressions without flaky failures.

## Why

When you bump a model, tweak a prompt, or the upstream document format shifts, extraction quality can silently degrade. This makes that drift a failing test.

## Features

- Record golden `source -> JSON` fixtures and replay them as a CI test suite.
- Type-aware semantic field diff (built on deepdiff) with per-field tolerances.
- Two-axis drift detection: model/prompt drift **and** source-format coverage drift.
- CI cost/latency budgets that fail the build over thresholds.
- Optional cached, pinnable LLM-judge for free-text fields; provider-agnostic wrappers for OpenAI / Anthropic / Ollama.
- Fully offline and deterministic test runs — you supply any `Callable[[source], dict]`; no provider is hardcoded.

## Installation

Requires Python 3.11+.

```bash
pip install extract-regress

# optional provider wrappers:
pip install "extract-regress[openai]"
pip install "extract-regress[anthropic]"
```

From a clone, for development:

```bash
pip install -e ".[dev,openai,anthropic]"
```

## Quickstart

### 1. Register your extractor

In your project's `conftest.py`, expose an `extract_regress_config()` hook that
returns the extraction function plus any per-field tolerances:

```python
from extract_regress import ERConfig
from extract_regress.tolerances import ToleranceConfig, ToleranceRule

from my_pipeline import extract_invoice  # your Callable[[source], dict]


def extract_regress_config() -> ERConfig:
    return ERConfig(
        extract_fn=extract_invoice,
        fixtures_dir="tests/extract_fixtures",
        tolerances=ToleranceConfig(
            rules=(
                ToleranceRule(path="total", abs_tol=0.01),
                ToleranceRule(path="line_items[*].amount", abs_tol=0.01),
                ToleranceRule(path="invoice_date", as_date=True, date_granularity="day"),
                ToleranceRule(path="vendor", ignore_case=True, ignore_whitespace=True),
                # Free-text fields can fall back to a cached LLM-judge:
                ToleranceRule(path="summary", judge=True),
            )
        ),
    )
```

A fixture is one JSON file per case under `fixtures_dir`:

```json
{
  "version": 1,
  "name": "invoice_basic",
  "source_ref": "samples/invoice_basic.txt",
  "source_inline": null,
  "expected": { "vendor": "Acme Corporation", "total": 1250.0 }
}
```

Exactly one of `source_ref` / `source_inline` must be set. `source_ref`
resolves relative to the fixture file's own directory, so fixtures are portable
across machines and CI.

### 2. Record goldens, then replay

```bash
# Capture goldens for fixtures that lack them + write the coverage snapshot:
extract-regress record        # or:  pytest --er-record

# On every change, replay and fail on regressions or budget breaches:
extract-regress run           # or:  pytest
```

The plugin auto-collects one test per fixture — no per-fixture boilerplate. A
run **fails** if any non-tolerated field diff appears, a field's fill-rate drops
beyond the coverage threshold, or a cost/latency budget is exceeded.

### CLI

| Command | What it does |
| --- | --- |
| `extract-regress record` | Build goldens for fixtures lacking them; refresh the coverage snapshot. |
| `extract-regress run --format term\|md\|json` | Replay + check; exit non-zero on regression or budget breach. |
| `extract-regress update` | Accept current outputs as the new goldens; refresh the snapshot. |
| `extract-regress report --format term\|md\|json` | Render the last run. |

`record`, `run`, and `update` accept a repeatable `-k`/`--name PATTERN` option
to target one or more fixtures by `fnmatch`-style glob instead of the full
golden set, e.g. `extract-regress record -k invoice_123` or
`extract-regress run -k 'invoice_*'`. A pattern with no glob metacharacters
is still an exact name match. Multiple `-k` values compose as a union. The
coverage snapshot refresh after a filtered `record`/`update` is still computed
from the full on-disk golden set, not just the filtered subset.

## Configuration

Static settings can also live in `pyproject.toml` under `[tool.extract_regress]`
(or a standalone `extract_regress.toml`):

```toml
[tool.extract_regress]
fixtures_dir = "tests/extract_fixtures"
coverage_drop_threshold = 0.1

[[tool.extract_regress.tolerances]]
path = "total"
abs_tol = 0.01

[tool.extract_regress.budget]
max_cost_usd_per_run = 0.50
max_p95_latency_ms = 2000
```

## Runnable example

A complete, offline demo lives in [`examples/invoices/`](examples/invoices/) — it
uses a fake extractor, so it runs with zero API keys:

```bash
cd examples/invoices
pytest                       # replay the golden fixtures
extract-regress run --project-dir .   # the same check via the CLI
```

See [`examples/invoices/README.md`](examples/invoices/README.md) for details.

## GitHub Action

A composite action ships in the package
([`src/extract_regress/action/action.yml`](src/extract_regress/action/action.yml)).
It runs `extract-regress run`, fails the build on a regression, coverage drop, or
budget breach, and — when a token is available — posts the field-level diff as a
collapsible PR comment. Without a token the diff is written to the job log and the
comment is skipped, so it is safe on forks.

```yaml
- uses: mykolapodpriatov/extract-regress@main
  with:
    project-dir: examples/invoices
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

## How it works

You wrap your extraction function once; the plugin snapshots outputs into
versioned fixtures. On each run it re-extracts, diffs field-by-field with type
awareness, and reports coverage/fill-rate changes plus token cost and latency
against your budget. The optional LLM-judge caches its verdicts (keyed by the
*resolved* model id) to a committable file, so CI stays deterministic.

## Tech stack

- Python, pytest
- pydantic, deepdiff, python-dateutil, rapidfuzz
- typer, rich
- Optional: OpenAI / Anthropic SDKs, Ollama

## Status & roadmap

🚧 **Early development.** Core record/replay, the type-aware field diff,
coverage drift, budgets, the cached judge, the pytest plugin, and the CLI are
implemented and tested.

- [x] Core record/replay engine + type-aware field diff
- [x] pytest plugin + CLI with cost/latency budgets
- [x] Optional cached LLM-judge for free-text fields
- [x] GitHub Action that renders the field-level diff as a PR comment

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
