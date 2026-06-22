# extract-regress

> pytest for LLM extraction — pin golden extractions and catch silent regressions when the prompt, model, schema, or source format drifts.

![status](https://img.shields.io/badge/status-early%20development-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![license](https://img.shields.io/badge/license-MIT-green)

A pytest plugin + CLI that records golden `(source -> expected JSON)` fixtures for LLM data-extraction pipelines and replays them in CI. Comparisons use a type-aware semantic field diff (dates, numbers, casing, list ordering, near-duplicate strings) instead of brittle string equality, so you catch real regressions without flaky failures.

## Why

When you bump a model, tweak a prompt, or the upstream document format shifts, extraction quality can silently degrade. This makes that drift a failing test.

## Features

- Record golden `source -> JSON` fixtures and replay them as a CI test suite
- Type-aware semantic field diff (built on deepdiff) with per-field tolerances
- Two-axis drift detection: model/prompt drift **and** source-format coverage drift
- CI cost/latency budgets that fail the build over thresholds
- Cached, pinnable LLM-judge mode for free-text semantic equivalence; provider-agnostic wrapper

## How it works

You wrap your extraction function once; the plugin snapshots outputs into versioned fixtures. On each run it re-extracts, diffs field-by-field with type awareness, and reports coverage/fill-rate changes plus token cost and latency against your budget.

## Tech stack

- Python
- pytest
- Pydantic
- deepdiff
- OpenAI / Anthropic SDKs
- Ollama

## Status & roadmap

🚧 **Early development.** This repository is being built in the open; the scaffold and design are in place and the implementation is landing incrementally.

- [ ] Core record/replay engine + type-aware field diff
- [ ] pytest plugin + CLI with cost/latency budgets
- [ ] GitHub Action that renders the field-level diff as a PR comment
- [ ] Optional cached LLM-judge for free-text fields; auto-fixture mining from logs

## Installation

> Coming soon.

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
