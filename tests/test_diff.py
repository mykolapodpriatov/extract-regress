"""Tests for the type-aware diff engine and its deterministic order (§3.4)."""

from __future__ import annotations

from extract_regress.diff import (
    _order_insensitive_parents,
    concrete_path,
    diff_extraction,
    glob_path,
)
from extract_regress.tolerances import ToleranceConfig, ToleranceRule
from extract_regress.types import FieldDiff


def _only(diffs: list[FieldDiff], path: str) -> FieldDiff:
    matches = [d for d in diffs if d.path == path]
    assert len(matches) == 1, f"expected exactly one diff at {path}, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def test_number_abs_tolerance_pass_and_fail() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="total", abs_tol=0.01),))
    ok = diff_extraction({"total": 100.0}, {"total": 100.005}, cfg)
    assert _only(ok, "total").tolerated

    bad = diff_extraction({"total": 100.0}, {"total": 100.5}, cfg)
    assert _only(bad, "total").failing


def test_number_rel_tolerance() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="amount", rel_tol=0.01),))
    ok = diff_extraction({"amount": 1000.0}, {"amount": 1009.0}, cfg)
    assert _only(ok, "amount").tolerated
    bad = diff_extraction({"amount": 1000.0}, {"amount": 1100.0}, cfg)
    assert _only(bad, "amount").failing


def test_int_float_is_numeric_change_not_type_change() -> None:
    # int<->float must be treated as a numeric `changed`, not a type_change.
    cfg = ToleranceConfig(rules=(ToleranceRule(path="qty", abs_tol=0.0),))
    diffs = diff_extraction({"qty": 2}, {"qty": 2.0}, cfg)
    d = _only(diffs, "qty")
    assert d.kind == "changed"
    assert d.tolerated  # 2 == 2.0 numerically


def test_no_rule_means_strict_equality() -> None:
    diffs = diff_extraction({"x": 1}, {"x": 2}, ToleranceConfig())
    assert _only(diffs, "x").failing


def test_bool_vs_int_is_a_type_change_not_numeric() -> None:
    # bool is a subclass of int, but True->1 is a real type change and must not
    # be reclassified to a numeric `changed` (nor tolerated as 1 == 1).
    cfg = ToleranceConfig(rules=(ToleranceRule(path="flag", abs_tol=10.0),))
    diffs = diff_extraction({"flag": True}, {"flag": 1}, cfg)
    d = _only(diffs, "flag")
    assert d.kind == "type_changed"
    assert d.failing


def test_int_vs_bool_is_a_type_change() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="flag", abs_tol=10.0),))
    diffs = diff_extraction({"flag": 0}, {"flag": False}, cfg)
    d = _only(diffs, "flag")
    assert d.kind == "type_changed"
    assert d.failing


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def test_date_instant_equality_across_formats() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="date", as_date=True),))
    diffs = diff_extraction({"date": "2020-01-01T00:00:00"}, {"date": "2020-01-01 00:00:00"}, cfg)
    assert _only(diffs, "date").tolerated


def test_date_day_granularity_ignores_time() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="date", as_date=True, date_granularity="day"),))
    ok = diff_extraction({"date": "2020-01-01"}, {"date": "2020-01-01T15:30:00"}, cfg)
    assert _only(ok, "date").tolerated


def test_date_second_granularity_detects_time_change() -> None:
    cfg = ToleranceConfig(
        rules=(ToleranceRule(path="date", as_date=True, date_granularity="second"),)
    )
    bad = diff_extraction({"date": "2020-01-01T00:00:00"}, {"date": "2020-01-01T00:00:05"}, cfg)
    assert _only(bad, "date").failing


def test_unparseable_date_falls_through_to_failing() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="date", as_date=True),))
    bad = diff_extraction({"date": "not-a-date"}, {"date": "also bad"}, cfg)
    assert _only(bad, "date").failing


def test_date_tolerance_accepts_native_date_and_datetime_objects() -> None:
    import datetime as dt

    cfg = ToleranceConfig(rules=(ToleranceRule(path="d", as_date=True, date_granularity="day"),))
    # A native date on one side and an equivalent datetime string on the other.
    diffs = diff_extraction({"d": dt.date(2020, 1, 1)}, {"d": "2020-01-01T09:00:00"}, cfg)
    assert _only(diffs, "d").tolerated


def test_date_tolerance_accepts_native_datetime_object() -> None:
    import datetime as dt

    cfg = ToleranceConfig(rules=(ToleranceRule(path="ts", as_date=True),))
    diffs = diff_extraction(
        {"ts": dt.datetime(2020, 1, 1, 12, 0, 0)},
        {"ts": "2020-01-01T12:00:00"},
        cfg,
    )
    assert _only(diffs, "ts").tolerated


def test_date_same_instant_different_offsets_match_at_second_granularity() -> None:
    # Same instant, expressed at two different UTC offsets, must compare equal
    # once both sides are normalized to UTC.
    cfg = ToleranceConfig(
        rules=(ToleranceRule(path="ts", as_date=True, date_granularity="second"),)
    )
    diffs = diff_extraction(
        {"ts": "2020-01-01T02:00:00+02:00"},
        {"ts": "2020-01-01T00:00:00+00:00"},
        cfg,
    )
    assert _only(diffs, "ts").tolerated


def test_date_naive_assumed_utc_matches_equivalent_offset_string() -> None:
    # A naive datetime golden (assumed UTC) and an aware actual at +05:00 that
    # denotes the same UTC instant must compare equal. The naive-UTC policy is
    # what makes this hold; a naive datetime treated as local would not.
    import datetime as dt

    cfg = ToleranceConfig(
        rules=(ToleranceRule(path="ts", as_date=True, date_granularity="second"),)
    )
    diffs = diff_extraction(
        {"ts": dt.datetime(2020, 1, 1, 5, 0, 0)},  # naive -> 05:00 UTC
        {"ts": "2020-01-01T10:00:00+05:00"},  # aware -> 05:00 UTC
        cfg,
    )
    assert _only(diffs, "ts").tolerated


def test_date_day_granularity_normalizes_naive_to_utc() -> None:
    # A naive datetime golden (assumed UTC) vs an aware *string* actual that
    # lands on the same UTC calendar day, even though the string's local date is
    # the previous day. The string side is parsed by the tolerance (not by
    # deepdiff), so this genuinely exercises the UTC normalization.
    import datetime as dt

    cfg = ToleranceConfig(rules=(ToleranceRule(path="d", as_date=True, date_granularity="day"),))
    diffs = diff_extraction(
        {"d": dt.datetime(2020, 1, 2, 0, 30, 0)},  # naive -> 2020-01-02 UTC
        # Local date 2020-01-01 at -01:00 is 2020-01-02 00:30 UTC: same UTC day.
        {"d": "2020-01-01T23:30:00-01:00"},
        cfg,
    )
    assert _only(diffs, "d").tolerated


def test_as_date_on_non_date_value_hard_fails_as_type_change() -> None:
    # ``as_date`` set, but the values are not date-like → no re-routing; the
    # underlying type change still hard-fails.
    cfg = ToleranceConfig(rules=(ToleranceRule(path="d", as_date=True),))
    diffs = diff_extraction({"d": 5}, {"d": "text"}, cfg)
    assert _only(diffs, "d").failing


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def test_case_insensitive_match() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="name", ignore_case=True),))
    assert _only(diff_extraction({"name": "ACME"}, {"name": "acme"}, cfg), "name").tolerated


def test_whitespace_normalized_match() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="addr", ignore_whitespace=True),))
    diffs = diff_extraction({"addr": "1  Main  St"}, {"addr": " 1 Main St "}, cfg)
    assert _only(diffs, "addr").tolerated


def test_fuzzy_near_duplicate_string() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="vendor", fuzzy_ratio=85.0),))
    # Near-duplicate (trailing punctuation / period) is accepted.
    ok = diff_extraction({"vendor": "Acme Corporation"}, {"vendor": "Acme Corporation."}, cfg)
    assert _only(ok, "vendor").tolerated
    # A genuinely different vendor name is rejected.
    bad = diff_extraction({"vendor": "Acme Corporation"}, {"vendor": "Globex Inc"}, cfg)
    assert _only(bad, "vendor").failing


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_list_order_insensitive_scalars() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="tags", ignore_order=True),))
    diffs = diff_extraction({"tags": ["a", "b", "c"]}, {"tags": ["c", "b", "a"]}, cfg)
    assert _only(diffs, "tags").tolerated


def test_list_order_insensitive_detects_real_difference() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="tags", ignore_order=True),))
    diffs = diff_extraction({"tags": ["a", "b"]}, {"tags": ["a", "x"]}, cfg)
    assert _only(diffs, "tags").failing


def test_list_order_insensitive_with_element_tolerance() -> None:
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="amounts", ignore_order=True),
            ToleranceRule(path="amounts[*]", abs_tol=0.01),
        )
    )
    diffs = diff_extraction({"amounts": [1.0, 2.0]}, {"amounts": [2.005, 1.0]}, cfg)
    assert _only(diffs, "amounts").tolerated


def test_ignore_order_and_abs_tol_on_same_list_path() -> None:
    # A single rule on the list path carries BOTH ignore_order and abs_tol; the
    # tolerance must reach the (reordered) scalar elements, not just the list.
    cfg = ToleranceConfig(rules=(ToleranceRule(path="amounts", ignore_order=True, abs_tol=0.01),))
    diffs = diff_extraction({"amounts": [1.0, 2.0]}, {"amounts": [2.005, 1.0]}, cfg)
    assert _only(diffs, "amounts").tolerated

    # And a genuinely out-of-tolerance value still fails the field.
    bad = diff_extraction({"amounts": [1.0, 2.0]}, {"amounts": [2.5, 1.0]}, cfg)
    assert _only(bad, "amounts").failing


def test_ignore_order_list_of_objects_with_nested_field_tolerance() -> None:
    # Unordered list of objects, compared recursively so a per-field numeric
    # tolerance on a nested key applies during element matching.
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="line_items", ignore_order=True),
            ToleranceRule(path="line_items[*].amount", abs_tol=0.01),
        )
    )
    golden = {
        "line_items": [
            {"sku": "A", "amount": 10.0},
            {"sku": "B", "amount": 20.0},
        ]
    }
    # Reordered, with amounts within tolerance; sku must still match exactly.
    actual = {
        "line_items": [
            {"sku": "B", "amount": 20.004},
            {"sku": "A", "amount": 9.997},
        ]
    }
    assert _only(diff_extraction(golden, actual, cfg), "line_items").tolerated


def test_ignore_order_list_of_objects_detects_real_field_change() -> None:
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="line_items", ignore_order=True),
            ToleranceRule(path="line_items[*].amount", abs_tol=0.01),
        )
    )
    golden = {"line_items": [{"sku": "A", "amount": 10.0}, {"sku": "B", "amount": 20.0}]}
    # Same skus/order-insensitive, but one amount is well outside tolerance.
    actual = {"line_items": [{"sku": "B", "amount": 20.0}, {"sku": "A", "amount": 99.0}]}
    assert _only(diff_extraction(golden, actual, cfg), "line_items").failing


def test_ordered_list_reports_per_element() -> None:
    diffs = diff_extraction({"xs": [1, 2]}, {"xs": [2, 1]}, ToleranceConfig())
    paths = {d.path for d in diffs}
    assert paths == {"xs[0]", "xs[1]"}


def test_order_insensitive_parents_discovered_in_sorted_order() -> None:
    # Sibling order-insensitive lists must be discovered in a deterministic
    # (key-sorted) order regardless of dict insertion order.
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="zebra", ignore_order=True),
            ToleranceRule(path="alpha", ignore_order=True),
            ToleranceRule(path="mango", ignore_order=True),
        )
    )
    golden = {"zebra": [1], "mango": [2], "alpha": [3]}
    actual = {"zebra": [1], "mango": [2], "alpha": [3]}
    parents = _order_insensitive_parents(golden, actual, cfg)
    assert parents == [["alpha"], ["mango"], ["zebra"]]


# ---------------------------------------------------------------------------
# Structure: added / removed / type-change
# ---------------------------------------------------------------------------


def test_added_and_removed_fields() -> None:
    # A shared key anchors the dict so deepdiff reports key-level add/remove
    # rather than a whole-object replacement.
    golden = {"shared": 1, "only_golden": 1}
    actual = {"shared": 1, "only_actual": 2}
    diffs = diff_extraction(golden, actual, ToleranceConfig())
    removed = _only(diffs, "only_golden")
    added = _only(diffs, "only_actual")
    assert removed.kind == "removed" and removed.failing
    assert added.kind == "added" and added.failing


def test_type_change_always_hard_fails() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="v", ignore_case=True, judge=True),))
    diffs = diff_extraction({"v": "1"}, {"v": [1]}, cfg)
    d = _only(diffs, "v")
    assert d.kind == "type_changed"
    assert d.failing


# ---------------------------------------------------------------------------
# Nested paths and globs
# ---------------------------------------------------------------------------


def test_nested_glob_tolerance_applies_to_each_element() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="line_items[*].amount", abs_tol=0.01),))
    golden = {"line_items": [{"amount": 10.0}, {"amount": 20.0}]}
    actual = {"line_items": [{"amount": 10.005}, {"amount": 20.004}]}
    diffs = diff_extraction(golden, actual, cfg)
    assert all(d.tolerated for d in diffs), diffs


def test_path_helpers_render_indices() -> None:
    assert glob_path(["line_items", 0, "amount"]) == "line_items[*].amount"
    assert concrete_path(["line_items", 0, "amount"]) == "line_items[0].amount"


# ---------------------------------------------------------------------------
# Deterministic evaluation order (§3.4) — asserted so precedence cannot drift.
# ---------------------------------------------------------------------------


def test_type_change_never_reaches_judge() -> None:
    """Step 3: type_changed hard-fails and is never sent to the judge."""
    calls: list[tuple[str, str, str]] = []

    def judge(path: str, exp: str, act: str) -> tuple[bool, str]:
        calls.append((path, exp, act))
        return True, "fake"

    cfg = ToleranceConfig(rules=(ToleranceRule(path="v", judge=True),))
    diffs = diff_extraction({"v": "abc"}, {"v": 123}, cfg, judge_fn=judge)
    assert _only(diffs, "v").kind == "type_changed"
    assert _only(diffs, "v").failing
    assert calls == [], "judge must not be consulted for a type change"


def test_judge_only_runs_for_non_tolerated_string_pairs() -> None:
    """Step 5: judge runs only when not tolerated, rule.judge, both strings."""
    seen: list[tuple[str, str, str]] = []

    def judge(path: str, exp: str, act: str) -> tuple[bool, str]:
        seen.append((path, exp, act))
        return True, "fake-model"

    cfg = ToleranceConfig(rules=(ToleranceRule(path="summary", judge=True),))
    diffs = diff_extraction(
        {"summary": "A short note."}, {"summary": "A brief note."}, cfg, judge_fn=judge
    )
    d = _only(diffs, "summary")
    assert d.tolerated
    assert "judge accepted" in d.reason
    assert seen == [("summary", "A short note.", "A brief note.")]


def test_judge_not_called_when_tolerance_already_passes() -> None:
    """Step 4 precedes step 5: a passing tolerance short-circuits the judge."""
    called = False

    def judge(path: str, exp: str, act: str) -> tuple[bool, str]:
        nonlocal called
        called = True
        return True, "fake"

    cfg = ToleranceConfig(rules=(ToleranceRule(path="name", ignore_case=True, judge=True),))
    diffs = diff_extraction({"name": "ACME"}, {"name": "acme"}, cfg, judge_fn=judge)
    assert _only(diffs, "name").tolerated
    assert not called


def test_judge_rejection_fails_the_diff() -> None:
    def judge(path: str, exp: str, act: str) -> tuple[bool, str]:
        return False, "fake-model"

    cfg = ToleranceConfig(rules=(ToleranceRule(path="summary", judge=True),))
    diffs = diff_extraction(
        {"summary": "Cats are great."}, {"summary": "Dogs are loud."}, cfg, judge_fn=judge
    )
    d = _only(diffs, "summary")
    assert d.failing
    assert "judge rejected" in d.reason


def test_judge_not_used_without_judge_flag() -> None:
    def judge(path: str, exp: str, act: str) -> tuple[bool, str]:
        return True, "fake"

    cfg = ToleranceConfig(rules=(ToleranceRule(path="summary"),))  # judge=False
    diffs = diff_extraction({"summary": "x"}, {"summary": "y"}, cfg, judge_fn=judge)
    assert _only(diffs, "summary").failing


def test_most_specific_rule_wins_over_glob() -> None:
    """Step 2: exact path beats a more permissive wildcard rule."""
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="line_items[*].amount", abs_tol=1000.0),
            ToleranceRule(path="line_items[0].amount", abs_tol=0.0),
        )
    )
    golden = {"line_items": [{"amount": 10.0}, {"amount": 20.0}]}
    actual = {"line_items": [{"amount": 11.0}, {"amount": 21.0}]}
    diffs = diff_extraction(golden, actual, cfg)
    assert _only(diffs, "line_items[0].amount").failing  # exact rule, abs_tol=0
    assert _only(diffs, "line_items[1].amount").tolerated  # wildcard, abs_tol=1000


def test_diffs_are_sorted_stably() -> None:
    diffs = diff_extraction({"b": 1, "a": 1, "c": 1}, {"b": 2, "a": 2, "c": 2}, ToleranceConfig())
    assert [d.path for d in diffs] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Order-insensitive lists whose elements themselves contain nested lists
# ---------------------------------------------------------------------------


def test_ignore_order_does_not_mask_nested_list_value_change() -> None:
    """A real change inside a nested list must FAIL, never be a false PASS.

    The parent list ``groups`` is order-insensitive; each element is an object
    holding a nested ``items`` list. A genuine value change deep inside that
    nested list must surface as a non-tolerated diff: the parent's
    ``ignore_order`` rule must NOT be re-applied to the nested child list during
    element comparison (which would let ``[1, 2]`` match ``[1, 999]``).
    """
    cfg = ToleranceConfig(rules=(ToleranceRule(path="groups", ignore_order=True),))
    golden = {
        "groups": [
            {"gid": "X", "items": [{"n": 1}, {"n": 2}]},
            {"gid": "Y", "items": [{"n": 3}]},
        ]
    }
    actual = {
        "groups": [
            {"gid": "X", "items": [{"n": 1}, {"n": 999}]},  # nested value changed
            {"gid": "Y", "items": [{"n": 3}]},
        ]
    }
    diffs = diff_extraction(golden, actual, cfg)
    assert _only(diffs, "groups").failing


def test_ignore_order_does_not_mask_nested_list_reorder_without_nested_rule() -> None:
    """A list-of-lists with only the OUTER list order-insensitive.

    The inner lists are reordered but there is no ``ignore_order`` rule for the
    nested path, so the inner reorder is a real difference and the field fails.
    This pins that the outer rule does not leak onto the inner lists.
    """
    cfg = ToleranceConfig(rules=(ToleranceRule(path="matrix", ignore_order=True),))
    golden = {"matrix": [[1, 2, 3], [4, 5, 6]]}
    actual = {"matrix": [[1, 2, 3], [6, 5, 4]]}  # inner reorder, no nested rule
    diffs = diff_extraction(golden, actual, cfg)
    assert _only(diffs, "matrix").failing


def test_unordered_list_equal_reports_no_diff() -> None:
    """Two equal order-insensitive lists must produce NO diff row (not noise).

    Previously a tolerated ``changed`` row was emitted even when the lists were
    identical, cluttering the report. Equal lists are simply not a difference.
    """
    cfg = ToleranceConfig(rules=(ToleranceRule(path="tags", ignore_order=True),))
    diffs = diff_extraction({"tags": ["a", "b", "c"]}, {"tags": ["a", "b", "c"]}, cfg)
    assert [d for d in diffs if d.path == "tags"] == []


def test_unordered_list_equal_in_full_doc_is_clean() -> None:
    # Identical documents with an order-insensitive list and a changed scalar:
    # only the genuine scalar change is reported; the equal list adds no row.
    cfg = ToleranceConfig(rules=(ToleranceRule(path="tags", ignore_order=True),))
    golden = {"tags": ["x", "y"], "total": 1}
    actual = {"tags": ["x", "y"], "total": 2}
    diffs = diff_extraction(golden, actual, cfg)
    assert {d.path for d in diffs} == {"total"}


def test_values_tolerated_guard_for_nonstandard_eq_in_unordered_element() -> None:
    """A non-standard ``__eq__`` element must not become a false PASS.

    When a list element's type reports ``!=`` yet :class:`deepdiff` finds no
    structural nodes, the empty diff list must be treated as NOT tolerated
    (``all([])`` would otherwise be ``True``), so the unordered match fails.
    """

    class AlwaysUnequal:
        __slots__ = ()

        def __eq__(self, other: object) -> bool:
            return False

        def __hash__(self) -> int:
            return 0

    cfg = ToleranceConfig(rules=(ToleranceRule(path="xs", ignore_order=True),))
    golden = {"xs": [AlwaysUnequal()]}
    actual = {"xs": [AlwaysUnequal()]}
    diffs = diff_extraction(golden, actual, cfg)
    # The two elements are unequal and uncovered by any tolerance, so the
    # unordered list must fail rather than silently passing on an empty diff set.
    assert _only(diffs, "xs").failing
