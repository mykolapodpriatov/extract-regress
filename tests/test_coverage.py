"""Tests for source-format drift via fill-rate (§3.5)."""

from __future__ import annotations

from pathlib import Path

from extract_regress.coverage import (
    compute_fill_rates,
    diff_coverage,
    load_baseline,
    present_fields,
    write_baseline,
)


def test_present_fields_ignores_nulls() -> None:
    fields = present_fields({"a": 1, "b": None, "c": "x"})
    assert fields == {"a", "c"}


def test_present_fields_nested_and_lists() -> None:
    fields = present_fields(
        {"vendor": {"name": "ACME", "tax_id": None}, "items": [{"sku": "1"}, {"sku": "2"}]}
    )
    assert "vendor.name" in fields
    assert "vendor.tax_id" not in fields
    assert "items[*].sku" in fields


def test_present_fields_empty_containers_count_as_present() -> None:
    # An empty dict/list is a present (non-null) leaf for fill-rate purposes.
    fields = present_fields({"meta": {}, "lines": [], "name": "x"})
    assert fields == {"meta", "lines", "name"}


def test_present_fields_populated_containers_count_at_own_path() -> None:
    # A non-empty dict/list must be present at its own path in addition to its
    # inner leaves; otherwise the container's fill-rate could drop to 0.
    fields = present_fields(
        {"tags": ["a", "b"], "line_items": [{"sku": "1"}], "vendor": {"name": "X"}}
    )
    assert "tags" in fields  # the populated list itself
    assert "tags[*]" not in fields  # scalar list elements have no leaf path
    assert "line_items" in fields  # the populated list of objects
    assert "line_items[*].sku" in fields  # and its inner leaf
    assert "vendor" in fields  # the populated dict itself
    assert "vendor.name" in fields  # and its inner leaf


def test_populated_container_fill_rate_does_not_drop() -> None:
    # Regression: a corpus where every fixture has a populated container must
    # report fill-rate 1.0 for that container path (not 0.0).
    corpus = [{"tags": ["x"]}, {"tags": ["y", "z"]}]
    rates = compute_fill_rates(corpus)
    assert rates["tags"] == 1.0


def test_present_fields_top_level_none() -> None:
    assert present_fields({"a": None}) == set()


def test_fill_rate_fractions() -> None:
    corpus = [
        {"a": 1, "b": 2},
        {"a": 1},  # b missing
        {"a": 1, "b": None},  # b null = absent
    ]
    rates = compute_fill_rates(corpus)
    assert rates["a"] == 1.0
    assert rates["b"] == 1 / 3


def test_fill_rate_empty_corpus() -> None:
    assert compute_fill_rates([]) == {}


def test_diff_coverage_flags_drop_over_threshold() -> None:
    baseline = {"vendor_tax_id": 1.0, "total": 1.0}
    current = {"vendor_tax_id": 0.5, "total": 1.0}
    deltas = {d.path: d for d in diff_coverage(baseline, current, drop_threshold=0.1)}
    assert deltas["vendor_tax_id"].dropped
    assert not deltas["total"].dropped
    assert deltas["vendor_tax_id"].delta == -0.5


def test_diff_coverage_small_drop_not_flagged() -> None:
    deltas = {d.path: d for d in diff_coverage({"x": 1.0}, {"x": 0.95}, drop_threshold=0.1)}
    assert not deltas["x"].dropped


def test_diff_coverage_new_field_not_flagged_as_drop() -> None:
    # A field only present in the current run has baseline 0 and never drops.
    deltas = {d.path: d for d in diff_coverage({}, {"new_field": 1.0})}
    assert not deltas["new_field"].dropped
    assert deltas["new_field"].baseline_fill_rate == 0.0


def test_diff_coverage_field_disappeared_is_flagged() -> None:
    deltas = {d.path: d for d in diff_coverage({"gone": 1.0}, {})}
    assert deltas["gone"].dropped
    assert deltas["gone"].current_fill_rate == 0.0


def test_baseline_roundtrip(fixtures_dir: Path) -> None:
    rates = {"b": 0.5, "a": 1.0}
    write_baseline(fixtures_dir, rates)
    loaded = load_baseline(fixtures_dir)
    assert loaded == rates


def test_load_missing_baseline_is_empty(tmp_path: Path) -> None:
    assert load_baseline(tmp_path) == {}


def test_baseline_is_written_sorted(fixtures_dir: Path) -> None:
    write_baseline(fixtures_dir, {"z": 1.0, "a": 0.5})
    text = (fixtures_dir / "coverage_baseline.json").read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"z"')
