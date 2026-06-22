"""Type-aware semantic field diff with a deterministic tolerance pass.

The pipeline (plan §3.4), applied per field and asserted in ``test_diff.py``:

1. **Structural classification** via :class:`deepdiff.DeepDiff` (tree view) →
   ``kind`` in ``{added, removed, changed, type_changed}``.
2. **Tolerance-rule resolution** for the field path, most-specific wins.
3. ``type_changed`` always hard-fails and is never judged (an ``int``/``float``
   pair is treated as a numeric ``changed`` rather than a shape change).
4. Apply the resolved tolerance; passing → ``tolerated=True``.
5. **Judge fallback** only when not tolerated, the rule sets ``judge=True`` and
   both sides are strings.

The judge call itself lives in :mod:`extract_regress.judge`; this module takes a
ready ``judge_fn`` and only decides *whether* a diff is judge-eligible.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any

from dateutil import parser as date_parser
from deepdiff import DeepDiff
from rapidfuzz import fuzz

from .tolerances import ToleranceConfig, ToleranceRule
from .types import DiffKind, FieldDiff, JudgeFn

__all__ = ["concrete_path", "diff_extraction", "glob_path"]

_NUMERIC = (int, float)


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real number for tolerance purposes.

    ``bool`` is a subclass of ``int`` in Python, but a ``bool``↔number change is
    a genuine type change (``True`` is not the number 1 for our purposes), so it
    is deliberately excluded from numeric matching and reclassification.
    """
    return isinstance(value, _NUMERIC) and not isinstance(value, bool)


def _path_components(node: Any) -> list[str | int]:
    """deepdiff path as a list of components, e.g. ``['items', 0, 'amount']``."""
    components: list[str | int] = node.path(output_format="list")
    return components


def _join(components: Sequence[str | int], *, wildcard_index: bool) -> str:
    """Render path components into our canonical dotted/bracketed form.

    With ``wildcard_index`` true, integer indices become ``[*]`` so the path
    can be matched against tolerance globs; otherwise the concrete index is
    kept for display/reporting.
    """
    out: list[str] = []
    for comp in components:
        if isinstance(comp, int):
            out.append("[*]" if wildcard_index else f"[{comp}]")
        else:
            out.append(comp if not out else f".{comp}")
    return "".join(out)


def glob_path(components: Sequence[str | int]) -> str:
    """Path with indices wildcarded, used for tolerance-rule matching."""
    return _join(components, wildcard_index=True)


def concrete_path(components: Sequence[str | int]) -> str:
    """Path with concrete indices, used for human-facing reporting."""
    return _join(components, wildcard_index=False)


# ---------------------------------------------------------------------------
# Tolerance predicates (return True iff the two values are within tolerance)
# ---------------------------------------------------------------------------


def _numbers_match(a: Any, b: Any, rule: ToleranceRule) -> bool:
    if not (_is_number(a) and _is_number(b)):
        return False
    if a == b:
        return True
    abs_tol = rule.abs_tol if rule.abs_tol is not None else 0.0
    rel_tol = rule.rel_tol if rule.rel_tol is not None else 0.0
    if abs_tol == 0.0 and rel_tol == 0.0:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return date_parser.parse(value)
        except (ValueError, OverflowError):
            return None
    return None


def _to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC for instant comparison.

    A *naive* datetime (no ``tzinfo``) is assumed to already be in UTC; an
    *aware* datetime is converted to UTC. This makes the same instant expressed
    at different offsets (e.g. ``...T02:00:00+02:00`` and ``...T00:00:00+00:00``)
    compare equal at both day and second granularity.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _dates_match(a: Any, b: Any, rule: ToleranceRule) -> bool:
    da, db = _to_datetime(a), _to_datetime(b)
    if da is None or db is None:
        return False
    da, db = _to_utc(da), _to_utc(db)
    if rule.date_granularity == "day":
        return da.date() == db.date()
    return da == db


_WS_RE = re.compile(r"\s+")


def _normalize_string(value: str, rule: ToleranceRule) -> str:
    if rule.ignore_whitespace:
        value = _WS_RE.sub(" ", value).strip()
    if rule.ignore_case:
        value = value.casefold()
    return value


def _strings_match(a: Any, b: Any, rule: ToleranceRule) -> bool:
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    na, nb = _normalize_string(a, rule), _normalize_string(b, rule)
    if na == nb:
        return True
    if rule.fuzzy_ratio is not None:
        return fuzz.ratio(na, nb) >= rule.fuzzy_ratio
    return False


def _scalar_tolerated(a: Any, b: Any, rule: ToleranceRule) -> tuple[bool, str]:
    """Apply numeric/date/string tolerances to a scalar pair.

    Returns ``(tolerated, reason)``.
    """
    if rule.as_date and _dates_match(a, b, rule):
        return True, f"dates equal at {rule.date_granularity} granularity"
    if _numbers_match(a, b, rule):
        return True, "within numeric tolerance"
    if _strings_match(a, b, rule):
        return True, "strings equal under normalization"
    return False, ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def diff_extraction(
    golden: dict[str, Any],
    actual: dict[str, Any],
    config: ToleranceConfig | None = None,
    *,
    judge_fn: JudgeFn | None = None,
) -> list[FieldDiff]:
    """Compute the tolerated/failing field diffs between ``golden`` and ``actual``.

    Args:
        golden: The pinned expected extraction.
        actual: The freshly produced extraction.
        config: Per-field tolerance rules. ``None`` means strict equality.
        judge_fn: Optional judge callable; consulted only for non-tolerated
            string diffs whose resolved rule sets ``judge=True``.

    Returns:
        One :class:`FieldDiff` per structural change, ordered for stable output.
    """
    config = config or ToleranceConfig()

    # Suppress per-element noise for list fields configured order-insensitive
    # by collecting their parent paths first, then re-diffing them holistically.
    order_insensitive_parents = _order_insensitive_parents(golden, actual, config)

    diffs = _collect_value_diffs(
        golden,
        actual,
        config,
        judge_fn,
        prefix=(),
        skip=lambda comps: _under_any(comps, order_insensitive_parents),
    )

    for components in order_insensitive_parents:
        item = _diff_unordered_list(golden, actual, components, config, judge_fn)
        if item is not None:
            diffs.append(item)

    diffs.sort(key=lambda d: (d.path, d.kind))
    return diffs


_KIND_KEYS: tuple[tuple[DiffKind, str], ...] = (
    ("type_changed", "type_changes"),
    ("removed", "dictionary_item_removed"),
    ("removed", "iterable_item_removed"),
    ("added", "dictionary_item_added"),
    ("added", "iterable_item_added"),
    ("changed", "values_changed"),
)


def _collect_value_diffs(
    golden: Any,
    actual: Any,
    config: ToleranceConfig,
    judge_fn: JudgeFn | None,
    *,
    prefix: tuple[str | int, ...],
    skip: Callable[[Sequence[str | int]], bool] | None = None,
) -> list[FieldDiff]:
    """Resolve every field diff between two values, paths anchored at ``prefix``.

    ``prefix`` is the path of the values being compared (empty at the top level,
    ``('line_items', 2)`` for the third element of a list, etc.). Each diff node
    reported by :class:`DeepDiff` is rebased onto ``prefix`` so tolerance rules
    resolve against the correct absolute path. This is what makes nested
    tolerances work both at the top level and inside unordered-list elements.

    ``skip`` receives the absolute components of each node and suppresses it when
    truthy (used to defer order-insensitive list subtrees to their holistic pass).
    """
    deep = DeepDiff(golden, actual, view="tree")
    diffs: list[FieldDiff] = []
    for kind, key in _KIND_KEYS:
        for node in deep.get(key, []):
            components = (*prefix, *_path_components(node))
            if skip is not None and skip(components):
                continue
            diffs.append(_resolve_diff(kind, components, node.t1, node.t2, config, judge_fn))
    return diffs


def _values_tolerated(
    golden: Any,
    actual: Any,
    config: ToleranceConfig,
    judge_fn: JudgeFn | None,
    *,
    prefix: tuple[str | int, ...],
) -> bool:
    """Whether two values are equal once nested tolerances at ``prefix`` apply."""
    if golden == actual:
        return True
    diffs = _collect_value_diffs(golden, actual, config, judge_fn, prefix=prefix)
    # The values are known to be unequal here. If :class:`DeepDiff` reported no
    # structural nodes at all (e.g. a type with a non-standard ``__eq__`` that
    # deepdiff serializes identically), ``all([])`` would be ``True`` and wrongly
    # mark the pair tolerated. Guard the empty case as NOT tolerated.
    if not diffs:
        return False
    return all(d.tolerated for d in diffs)


def _resolve_diff(
    kind: DiffKind,
    components: Sequence[str | int],
    t1: Any,
    t2: Any,
    config: ToleranceConfig,
    judge_fn: JudgeFn | None,
) -> FieldDiff:
    cpath = concrete_path(components)
    # Resolve against the *concrete* path so an exact-index rule
    # (``line_items[0].amount``) and a wildcard rule (``line_items[*].amount``)
    # can both match and compete via the §3.4 precedence; ``[*]`` matches ``[N]``.
    rule = config.resolve(cpath)

    # int<->float is a numeric change, not a shape regression. Re-classify so
    # numeric tolerances can apply and the judge is never consulted for it.
    if kind == "type_changed" and _is_number(t1) and _is_number(t2):
        kind = "changed"

    # A date field opted into via ``as_date`` re-routes a date<->string type
    # change to the date comparison (e.g. a native ``date`` golden vs an ISO
    # string actual). This is an explicit opt-in, unlike string<->number.
    if (
        kind == "type_changed"
        and rule is not None
        and rule.as_date
        and _to_datetime(t1) is not None
        and _to_datetime(t2) is not None
    ):
        kind = "changed"

    if kind == "type_changed":
        return FieldDiff(
            path=cpath,
            kind="type_changed",
            golden=t1,
            actual=t2,
            tolerated=False,
            reason=f"type changed {type(t1).__name__} -> {type(t2).__name__}",
        )

    if kind in ("added", "removed"):
        return FieldDiff(
            path=cpath,
            kind=kind,
            golden=t1 if kind == "removed" else None,
            actual=t2 if kind == "added" else None,
            tolerated=False,
            reason="field added in actual" if kind == "added" else "field missing in actual",
        )

    # kind == "changed"
    if rule is not None:
        tolerated, reason = _scalar_tolerated(t1, t2, rule)
        if tolerated:
            return FieldDiff(
                path=cpath,
                kind="changed",
                golden=t1,
                actual=t2,
                tolerated=True,
                reason=reason,
            )

    # Judge fallback: only for non-tolerated string pairs on a judge rule.
    if (
        judge_fn is not None
        and rule is not None
        and rule.judge
        and isinstance(t1, str)
        and isinstance(t2, str)
    ):
        verdict, model_id = judge_fn(cpath, t1, t2)
        if verdict:
            return FieldDiff(
                path=cpath,
                kind="changed",
                golden=t1,
                actual=t2,
                tolerated=True,
                reason=f"judge accepted ({model_id})",
            )
        return FieldDiff(
            path=cpath,
            kind="changed",
            golden=t1,
            actual=t2,
            tolerated=False,
            reason=f"judge rejected ({model_id})",
        )

    return FieldDiff(
        path=cpath,
        kind="changed",
        golden=t1,
        actual=t2,
        tolerated=False,
        reason="value differs",
    )


# ---------------------------------------------------------------------------
# Order-insensitive list handling
# ---------------------------------------------------------------------------


def _order_insensitive_parents(
    golden: dict[str, Any],
    actual: dict[str, Any],
    config: ToleranceConfig,
) -> list[list[str | int]]:
    """Find list fields whose resolved rule sets ``ignore_order``.

    We discover candidate list paths by walking ``golden`` and ``actual`` in
    parallel and checking the resolved tolerance for each list-valued path.
    """
    found: list[list[str | int]] = []
    seen: set[str] = set()

    def visit(g: Any, a: Any, components: list[str | int]) -> None:
        if isinstance(g, list) and isinstance(a, list):
            gpath = glob_path(components)
            rule = config.resolve(gpath)
            if rule is not None and rule.ignore_order and gpath not in seen:
                seen.add(gpath)
                found.append(list(components))
            return
        if isinstance(g, dict) and isinstance(a, dict):
            # Sorted for deterministic intermediate ordering of discovered paths.
            for key in sorted(g.keys() & a.keys()):
                visit(g[key], a[key], [*components, key])

    visit(golden, actual, [])
    return found


def _under_any(components: Sequence[str | int], parents: list[list[str | int]]) -> bool:
    """Whether ``components`` is at or beneath one of the ``parents`` paths."""
    for parent in parents:
        if len(components) >= len(parent) and list(components[: len(parent)]) == parent:
            return True
    return False


def _dig(root: dict[str, Any], components: Sequence[str | int]) -> Any:
    cur: Any = root
    for comp in components:
        cur = cur[comp]
    return cur


def _diff_unordered_list(
    golden: dict[str, Any],
    actual: dict[str, Any],
    components: Sequence[str | int],
    config: ToleranceConfig,
    judge_fn: JudgeFn | None,
) -> FieldDiff | None:
    """Compare a list field order-insensitively under element tolerances.

    Greedy multiset matching: each golden element must find an as-yet-unused
    actual element it is element-equal to. Elements are compared *recursively*
    so per-field tolerances inside objects (``amounts[*].amount``) and a
    tolerance declared on the list path itself both apply; any unmatched element
    on either side fails the field as a whole.
    """
    g_list = _dig(golden, components)
    a_list = _dig(actual, components)
    cpath = concrete_path(components)

    # Equal lists are not a difference at all: emitting a tolerated ``changed``
    # row here is pure noise in the report. Report no diff for this field.
    if g_list == a_list:
        return None

    remaining = list(range(len(a_list)))
    unmatched_golden: list[Any] = []
    for g_item in g_list:
        for pos, idx in enumerate(remaining):
            if _element_equal(g_item, a_list[idx], components, config, judge_fn):
                remaining.pop(pos)
                break
        else:
            unmatched_golden.append(g_item)

    if not unmatched_golden and not remaining:
        return FieldDiff(
            path=cpath,
            kind="changed",
            golden=g_list,
            actual=a_list,
            tolerated=True,
            reason="lists equal ignoring order",
        )
    return FieldDiff(
        path=cpath,
        kind="changed",
        golden=g_list,
        actual=a_list,
        tolerated=False,
        reason=(
            f"unordered list mismatch: {len(unmatched_golden)} unmatched golden, "
            f"{len(remaining)} extra actual"
        ),
    )


def _element_equal(
    g: Any,
    a: Any,
    list_components: Sequence[str | int],
    config: ToleranceConfig,
    judge_fn: JudgeFn | None,
) -> bool:
    """Whether two unordered-list elements match under the configured tolerances.

    Comparison is recursive: an element is compared at the path
    ``<list>[*]`` so nested per-field rules apply. A scalar element additionally
    falls back to a tolerance declared directly on the *list path* (e.g.
    ``ToleranceRule(path="amounts", ignore_order=True, abs_tol=0.01)``), which is
    the natural place to put an element tolerance for a list of scalars.
    """
    if g == a:
        return True

    # Recursive comparison anchored at a representative element index. Rule
    # matching treats ``[0]`` and ``[*]`` equivalently, so element-path and
    # wildcard rules both resolve here.
    elem_prefix = (*tuple(list_components), 0)
    if _values_tolerated(g, a, config, judge_fn, prefix=elem_prefix):
        return True

    # Fallback for scalars: honor a tolerance set on the list path itself.
    if not isinstance(g, dict | list) and not isinstance(a, dict | list):
        list_rule = config.resolve(glob_path(list(list_components)))
        if list_rule is not None:
            tolerated, _ = _scalar_tolerated(g, a, list_rule)
            return tolerated
    return False
