"""Cost/latency accounting and threshold checks (plan §3.6).

The runner retains the full per-call ``latency_ms`` sample list and per-call
cost across a run; :class:`Usage` is one call's record, this module aggregates
the list. Thresholds: ``max_cost_usd_per_run`` (sum of per-call cost) and
``max_p95_latency_ms`` (95th percentile over *all* extractor calls). When usage
is unavailable (user did not wrap a provider) the check is **skipped** with a
note, never a hard error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from .types import BudgetOutcome, Usage


class BudgetConfig(BaseModel):
    """Per-run cost and latency thresholds."""

    model_config = ConfigDict(frozen=True)

    max_cost_usd_per_run: float | None = None
    max_p95_latency_ms: float | None = None

    @property
    def enabled(self) -> bool:
        """Whether any threshold is configured."""
        return self.max_cost_usd_per_run is not None or self.max_p95_latency_ms is not None


def percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same method as ``numpy`` default).

    Args:
        samples: Non-empty list of numeric samples.
        pct: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("percentile of an empty sample list is undefined")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def evaluate_budget(usages: Iterable[Usage], config: BudgetConfig) -> BudgetOutcome:
    """Aggregate per-call usage and check it against ``config``.

    Returns a not-checked :class:`BudgetOutcome` when budgets are disabled or
    when no usage carries the data a configured threshold needs.
    """
    usages = list(usages)

    if not config.enabled:
        return BudgetOutcome(checked=False, passed=True, messages=("budgets disabled",))

    costs = [u.cost_usd for u in usages if u.has_cost]
    latencies = [u.latency_ms for u in usages if u.has_latency]

    messages: list[str] = []
    passed = True
    total_cost: float | None = None
    p95: float | None = None

    # Cost --------------------------------------------------------------
    cost_checkable = config.max_cost_usd_per_run is not None
    if cost_checkable and not costs:
        messages.append("cost budget set but no per-call cost reported; skipping cost check")
        cost_checkable = False
    if cost_checkable:
        assert config.max_cost_usd_per_run is not None
        total_cost = math.fsum(c for c in costs if c is not None)
        if total_cost > config.max_cost_usd_per_run:
            passed = False
            messages.append(
                f"cost ${total_cost:.4f} exceeds budget ${config.max_cost_usd_per_run:.4f}"
            )
        else:
            messages.append(
                f"cost ${total_cost:.4f} within budget ${config.max_cost_usd_per_run:.4f}"
            )

    # Latency -----------------------------------------------------------
    latency_checkable = config.max_p95_latency_ms is not None
    if latency_checkable and not latencies:
        messages.append(
            "latency budget set but no per-call latency reported; skipping latency check"
        )
        latency_checkable = False
    if latency_checkable:
        assert config.max_p95_latency_ms is not None
        p95 = percentile([latency for latency in latencies if latency is not None], 95.0)
        if p95 > config.max_p95_latency_ms:
            passed = False
            messages.append(
                f"p95 latency {p95:.1f}ms exceeds budget {config.max_p95_latency_ms:.1f}ms"
            )
        else:
            messages.append(
                f"p95 latency {p95:.1f}ms within budget {config.max_p95_latency_ms:.1f}ms"
            )

    checked = cost_checkable or latency_checkable
    if not checked:
        # Thresholds were set but nothing was measurable.
        return BudgetOutcome(
            checked=False,
            passed=True,
            max_cost_usd=config.max_cost_usd_per_run,
            max_p95_latency_ms=config.max_p95_latency_ms,
            messages=tuple(messages),
        )

    return BudgetOutcome(
        checked=True,
        passed=passed,
        total_cost_usd=total_cost,
        p95_latency_ms=p95,
        max_cost_usd=config.max_cost_usd_per_run,
        max_p95_latency_ms=config.max_p95_latency_ms,
        messages=tuple(messages),
    )
