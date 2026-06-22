"""Tests for cost/latency budget accounting and thresholds (§3.6)."""

from __future__ import annotations

import pytest

from extract_regress.budget import BudgetConfig, evaluate_budget, percentile
from extract_regress.types import Usage


def test_percentile_interpolation() -> None:
    assert percentile([10.0], 95.0) == 10.0
    assert percentile([0.0, 100.0], 50.0) == 50.0
    # p95 of 1..100 sorted = 95.05 by linear interpolation.
    assert percentile([float(i) for i in range(1, 101)], 95.0) == pytest.approx(95.05)


def test_percentile_at_exact_rank_no_interpolation() -> None:
    # rank lands exactly on an index (lower == upper), so no interpolation.
    assert percentile([0.0, 10.0, 20.0], 0.0) == 0.0
    assert percentile([0.0, 10.0, 20.0], 100.0) == 20.0


def test_percentile_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 95.0)


def test_budget_disabled_is_not_checked() -> None:
    outcome = evaluate_budget([Usage(cost_usd=1.0)], BudgetConfig())
    assert not outcome.checked
    assert outcome.passed


def test_cost_under_budget_passes() -> None:
    usages = [Usage(cost_usd=0.01), Usage(cost_usd=0.02)]
    outcome = evaluate_budget(usages, BudgetConfig(max_cost_usd_per_run=0.10))
    assert outcome.checked
    assert outcome.passed
    assert outcome.total_cost_usd == pytest.approx(0.03)


def test_cost_over_budget_fails() -> None:
    usages = [Usage(cost_usd=0.06), Usage(cost_usd=0.06)]
    outcome = evaluate_budget(usages, BudgetConfig(max_cost_usd_per_run=0.10))
    assert outcome.checked
    assert outcome.failing
    assert any("exceeds budget" in m for m in outcome.messages)


def test_latency_p95_over_budget_fails() -> None:
    usages = [Usage(latency_ms=float(x)) for x in [100, 100, 100, 5000]]
    outcome = evaluate_budget(usages, BudgetConfig(max_p95_latency_ms=1000.0))
    assert outcome.failing
    assert outcome.p95_latency_ms is not None and outcome.p95_latency_ms > 1000.0


def test_latency_p95_under_budget_passes() -> None:
    usages = [Usage(latency_ms=float(x)) for x in [100, 120, 130, 150]]
    outcome = evaluate_budget(usages, BudgetConfig(max_p95_latency_ms=1000.0))
    assert outcome.passed
    assert outcome.checked


def test_missing_usage_skips_with_message_not_error() -> None:
    # Thresholds set but no usage carries cost/latency → skipped, never a hard fail.
    usages = [Usage(), Usage()]
    outcome = evaluate_budget(
        usages, BudgetConfig(max_cost_usd_per_run=0.10, max_p95_latency_ms=1000.0)
    )
    assert not outcome.checked
    assert outcome.passed
    assert any("skipping" in m for m in outcome.messages)


def test_partial_usage_checks_only_available_axis() -> None:
    # Cost present, latency absent: cost is checked, latency is skipped.
    usages = [Usage(cost_usd=0.5)]
    outcome = evaluate_budget(
        usages, BudgetConfig(max_cost_usd_per_run=0.10, max_p95_latency_ms=1000.0)
    )
    assert outcome.checked
    assert outcome.failing  # cost 0.5 > 0.10
    assert any("latency" in m and "skipping" in m for m in outcome.messages)


def test_budget_config_enabled_flag() -> None:
    assert not BudgetConfig().enabled
    assert BudgetConfig(max_cost_usd_per_run=1.0).enabled
    assert BudgetConfig(max_p95_latency_ms=1.0).enabled
