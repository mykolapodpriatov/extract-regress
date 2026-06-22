"""Per-field tolerance configuration and rule resolution.

A :class:`ToleranceRule` describes how leniently one field path (with glob
support) should be compared. :class:`ToleranceConfig` holds the ordered list
of rules and resolves the single most-specific rule for a concrete field path
using the deterministic precedence defined in the plan (§3.4):

    (a) exact literal path
    (b) fewest wildcards
    (c) longest literal prefix
    (d) declaration order

Exactly one rule applies to any path; ties are impossible by construction.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DateGranularity = Literal["day", "second"]


class ToleranceRule(BaseModel):
    """Comparison leniency for fields matching ``path``.

    ``path`` is a glob over the dotted/bracketed field path produced by the
    diff engine, e.g. ``line_items[*].amount`` or ``vendor.*``.
    """

    model_config = ConfigDict(frozen=True)

    path: str

    # Numbers ---------------------------------------------------------------
    abs_tol: float | None = None
    """Absolute numeric tolerance: ``|a - b| <= abs_tol``."""
    rel_tol: float | None = None
    """Relative numeric tolerance: ``|a - b| <= rel_tol * max(|a|, |b|)``."""

    # Dates -----------------------------------------------------------------
    as_date: bool = False
    """Parse both sides as dates/datetimes and compare as instants.

    Both sides are normalized to UTC before comparison, so the same instant at
    different offsets compares equal. A naive datetime (no timezone) is assumed
    to be UTC.
    """
    date_granularity: DateGranularity = "second"
    """Compare to the ``day`` or to the ``second`` (after UTC normalization)."""

    # Strings ---------------------------------------------------------------
    ignore_case: bool = False
    """Case-insensitive string comparison."""
    ignore_whitespace: bool = False
    """Collapse runs of whitespace and strip ends before comparing."""
    fuzzy_ratio: float | None = None
    """Accept strings whose ``rapidfuzz`` ratio is ``>= fuzzy_ratio`` (0..100)."""

    # Lists -----------------------------------------------------------------
    ignore_order: bool = False
    """Compare list elements order-insensitively."""

    # Free text -------------------------------------------------------------
    judge: bool = False
    """Allow LLM-judge fallback for non-tolerated *string* diffs on this path."""

    @model_validator(mode="after")
    def _validate(self) -> ToleranceRule:
        if self.fuzzy_ratio is not None and not (0.0 <= self.fuzzy_ratio <= 100.0):
            raise ValueError("fuzzy_ratio must be between 0 and 100")
        if self.rel_tol is not None and self.rel_tol < 0:
            raise ValueError("rel_tol must be non-negative")
        if self.abs_tol is not None and self.abs_tol < 0:
            raise ValueError("abs_tol must be non-negative")
        return self


# A field path is a sequence of tokens: a ``.key`` segment, a ``[N]`` index, or
# the wildcards ``*`` (any key segment) and ``[*]`` (any index). ``[N]`` is a
# *literal* index reference, never a character class — so we do not use
# ``fnmatch``, whose ``[...]`` semantics would mis-handle our index brackets.
_TOKEN_RE = re.compile(r"\[\*\]|\[\d+\]|\*|[^.\[\]]+")
_WILDCARD_TOKENS = frozenset({"*", "[*]"})


def _tokenize(path: str) -> list[str]:
    """Split a path/pattern into its segment and index tokens."""
    return _TOKEN_RE.findall(path)


def _count_wildcards(pattern: str) -> int:
    """Number of wildcard tokens (``*`` or ``[*]``) in a pattern."""
    return sum(1 for tok in _tokenize(pattern) if tok in _WILDCARD_TOKENS)


def _literal_prefix_len(pattern: str) -> int:
    """Length of the leading run of literal characters before any wildcard."""
    length = 0
    for tok in _tokenize(pattern):
        if tok in _WILDCARD_TOKENS:
            break
        length += len(tok)
    return length


def path_matches(pattern: str, path: str) -> bool:
    """Whether ``path`` matches glob ``pattern`` under our path grammar.

    ``*`` matches exactly one key segment and ``[*]`` matches any index; ``[N]``
    is a literal index. Token counts must align, so ``vendor.*`` matches
    ``vendor.name`` but not ``vendor.address.city``.
    """
    pat_tokens = _tokenize(pattern)
    path_tokens = _tokenize(path)
    if len(pat_tokens) != len(path_tokens):
        return False
    for pat_tok, path_tok in zip(pat_tokens, path_tokens, strict=True):
        if pat_tok == "*":
            if path_tok.startswith("["):  # ``*`` matches a key, not an index
                return False
        elif pat_tok == "[*]":
            if not path_tok.startswith("["):  # ``[*]`` matches an index only
                return False
        elif pat_tok != path_tok:
            return False
    return True


class ToleranceConfig(BaseModel):
    """Ordered collection of :class:`ToleranceRule` with deterministic lookup."""

    model_config = ConfigDict(frozen=True)

    rules: tuple[ToleranceRule, ...] = Field(default_factory=tuple)

    @classmethod
    def from_iterable(cls, rules: object) -> ToleranceConfig:
        """Build a config from an iterable of rules or rule-shaped mappings."""
        coerced: list[ToleranceRule] = []
        for item in rules:  # type: ignore[attr-defined]
            if isinstance(item, ToleranceRule):
                coerced.append(item)
            elif isinstance(item, dict):
                coerced.append(ToleranceRule(**item))
            else:  # pragma: no cover - defensive
                raise TypeError(f"cannot coerce {item!r} into a ToleranceRule")
        return cls(rules=tuple(coerced))

    def resolve(self, path: str) -> ToleranceRule | None:
        """Return the single most-specific rule matching ``path``.

        Implements the precedence from the plan (§3.4). Returns ``None`` when
        no rule matches, in which case strict equality is used.
        """
        candidates = [
            (index, rule) for index, rule in enumerate(self.rules) if path_matches(rule.path, path)
        ]
        if not candidates:
            return None

        def sort_key(item: tuple[int, ToleranceRule]) -> tuple[int, int, int, int]:
            index, rule = item
            is_exact = rule.path == path
            return (
                0 if is_exact else 1,  # (a) exact literal path first
                _count_wildcards(rule.path),  # (b) fewest wildcards
                -_literal_prefix_len(rule.path),  # (c) longest literal prefix
                index,  # (d) declaration order
            )

        candidates.sort(key=sort_key)
        return candidates[0][1]
