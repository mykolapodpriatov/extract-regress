"""Source-format drift via per-field fill-rate (plan §3.5).

Fill-rate for a field = fraction of fixtures in which the field is *present and
non-null*. The baseline is a single project-level ``coverage_baseline.json``
snapshot (one ``field_path -> fill_rate`` map for the whole corpus), never
stored per fixture. ``run`` is read-only and flags fields whose recomputed
fill-rate dropped by more than ``coverage_drop_threshold`` versus the snapshot.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .types import CoverageDelta

DEFAULT_DROP_THRESHOLD = 0.1
BASELINE_FILENAME = "coverage_baseline.json"


def _flatten_present(value: Any, prefix: str, out: set[str]) -> None:
    """Record the leaf and container paths that are present and non-null.

    Dicts recurse by key; lists collapse indices to ``[*]`` so a field's
    presence is measured per-fixture, not per-element. A ``None`` leaf is
    treated as absent and is not recorded. A dict/list *container* is itself
    counted as present at its own path whenever it is non-null (whether empty or
    populated), so a populated container like ``tags`` or ``line_items`` never
    drops to a zero fill-rate just because we also descend into its contents.
    """
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, sub in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if sub is None:
                continue
            if isinstance(sub, Mapping | list):
                # The container itself is present at ``child`` (empty or not),
                # then we recurse to record its inner leaves.
                out.add(child)
                _flatten_present(sub, child, out)
            else:
                out.add(child)
    elif isinstance(value, list):
        for item in value:
            _flatten_present(item, f"{prefix}[*]", out)


def present_fields(extraction: Mapping[str, Any]) -> set[str]:
    """Set of non-null leaf field paths present in one extraction."""
    out: set[str] = set()
    _flatten_present(extraction, "", out)
    return out


def compute_fill_rates(extractions: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Per-field fill-rate across a corpus of extractions.

    The denominator is the number of fixtures; the numerator is the count of
    fixtures in which the field path is present and non-null.
    """
    extractions = list(extractions)
    total = len(extractions)
    if total == 0:
        return {}
    counts: dict[str, int] = {}
    for extraction in extractions:
        for field in present_fields(extraction):
            counts[field] = counts.get(field, 0) + 1
    return {field: count / total for field, count in counts.items()}


def diff_coverage(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> list[CoverageDelta]:
    """Compare current fill-rates to the baseline snapshot.

    A field is ``dropped`` when its fill-rate fell by more than
    ``drop_threshold`` relative to the baseline. Fields only in ``current``
    (newly seen) are reported with a zero baseline and never flagged as drops.
    """
    deltas: list[CoverageDelta] = []
    for field in sorted(set(baseline) | set(current)):
        base = baseline.get(field, 0.0)
        cur = current.get(field, 0.0)
        dropped = (base - cur) > drop_threshold
        deltas.append(
            CoverageDelta(
                path=field,
                baseline_fill_rate=base,
                current_fill_rate=cur,
                dropped=dropped,
            )
        )
    return deltas


def baseline_path(fixtures_dir: Path | str) -> Path:
    """Location of the project-level coverage snapshot."""
    return Path(fixtures_dir) / BASELINE_FILENAME


def load_baseline(fixtures_dir: Path | str) -> dict[str, float]:
    """Load the coverage snapshot, or an empty map when none exists."""
    path = baseline_path(fixtures_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def write_baseline(fixtures_dir: Path | str, fill_rates: Mapping[str, float]) -> Path:
    """Atomically rewrite the coverage snapshot from ``fill_rates``."""
    path = baseline_path(fixtures_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: fill_rates[k] for k in sorted(fill_rates)}
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return path
