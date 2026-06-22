"""Tests for tolerance config, glob grammar, and §3.4 precedence resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from extract_regress.tolerances import ToleranceConfig, ToleranceRule, path_matches

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"path": "x", "fuzzy_ratio": -1.0},
        {"path": "x", "fuzzy_ratio": 101.0},
        {"path": "x", "rel_tol": -0.1},
        {"path": "x", "abs_tol": -0.1},
    ],
)
def test_invalid_rule_values_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ToleranceRule(**kwargs)


def test_valid_boundary_values() -> None:
    assert ToleranceRule(path="x", fuzzy_ratio=0.0).fuzzy_ratio == 0.0
    assert ToleranceRule(path="x", fuzzy_ratio=100.0).fuzzy_ratio == 100.0


# ---------------------------------------------------------------------------
# Glob grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,path,expected",
    [
        ("total", "total", True),
        ("line_items[*].amount", "line_items[0].amount", True),
        ("line_items[*].amount", "line_items[12].amount", True),
        ("line_items[0].amount", "line_items[0].amount", True),
        ("line_items[1].amount", "line_items[0].amount", False),
        ("vendor.*", "vendor.name", True),
        ("vendor.*", "vendor.address.city", False),  # token count differs
        ("*", "total", True),
        ("*", "items[0]", False),  # token count differs
        ("a.*", "a[0]", False),  # same length: '*' won't match an index token
        ("[*]", "items", False),  # '[*]' matches an index, not a key
        ("a.b.c", "a.b.c", True),
        ("a.b.c", "a.b.d", False),
    ],
)
def test_path_matches(pattern: str, path: str, expected: bool) -> None:
    assert path_matches(pattern, path) is expected


# ---------------------------------------------------------------------------
# Precedence (§3.4): exact > fewest wildcards > longest prefix > decl order
# ---------------------------------------------------------------------------


def test_exact_beats_wildcard() -> None:
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="a.b[*].c", abs_tol=999.0),
            ToleranceRule(path="a.b[0].c", abs_tol=0.0),
        )
    )
    resolved = cfg.resolve("a.b[0].c")
    assert resolved is not None and resolved.abs_tol == 0.0


def test_fewest_wildcards_wins() -> None:
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="*.*", abs_tol=1.0),
            ToleranceRule(path="vendor.*", abs_tol=2.0),
        )
    )
    resolved = cfg.resolve("vendor.name")
    assert resolved is not None and resolved.abs_tol == 2.0


def test_longest_literal_prefix_wins_on_wildcard_tie() -> None:
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="a.*", abs_tol=1.0),
            ToleranceRule(path="abc.*", abs_tol=2.0),
        )
    )
    resolved = cfg.resolve("abc.x")
    assert resolved is not None and resolved.abs_tol == 2.0


def test_declaration_order_breaks_remaining_ties() -> None:
    cfg = ToleranceConfig(
        rules=(
            ToleranceRule(path="x.*", abs_tol=1.0),
            ToleranceRule(path="x.*", abs_tol=2.0),
        )
    )
    resolved = cfg.resolve("x.y")
    assert resolved is not None and resolved.abs_tol == 1.0


def test_no_match_returns_none() -> None:
    cfg = ToleranceConfig(rules=(ToleranceRule(path="a.b", abs_tol=1.0),))
    assert cfg.resolve("c.d") is None


# ---------------------------------------------------------------------------
# from_iterable coercion
# ---------------------------------------------------------------------------


def test_from_iterable_accepts_rules_and_mappings() -> None:
    cfg = ToleranceConfig.from_iterable(
        [
            ToleranceRule(path="a", abs_tol=1.0),
            {"path": "b", "ignore_case": True},
        ]
    )
    assert len(cfg.rules) == 2
    assert cfg.resolve("a").abs_tol == 1.0  # type: ignore[union-attr]
    assert cfg.resolve("b").ignore_case is True  # type: ignore[union-attr]


def test_from_iterable_rejects_garbage() -> None:
    with pytest.raises(TypeError):
        ToleranceConfig.from_iterable([42])
