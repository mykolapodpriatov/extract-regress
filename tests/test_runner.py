"""Tests for the run orchestrator (§3.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from extract_regress.config import ERConfig
from extract_regress.fixtures import Fixture, FixtureError, FixtureStore
from extract_regress.runner import Runner, normalize_result
from extract_regress.tolerances import ToleranceConfig, ToleranceRule
from extract_regress.types import ExtractionResult, Usage
from tests._fakes import DriftedExtractor, FakeExtractor


def _seed(fixtures_dir: Path, name: str, source: str, expected: dict) -> None:
    """Seed a fixture that already has a recorded golden ``expected``."""
    FixtureStore(fixtures_dir).save(Fixture(name=name, source_inline=source, expected=expected))


def _seed_unrecorded(fixtures_dir: Path, name: str, source: str) -> None:
    """Seed a fixture with NO golden recorded (no ``expected`` key on disk)."""
    import json

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "name": name, "source_inline": source}
    (fixtures_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# normalize_result
# ---------------------------------------------------------------------------


def test_normalize_dict_wraps_with_empty_usage() -> None:
    result = normalize_result({"a": 1})
    assert isinstance(result, ExtractionResult)
    assert result.value == {"a": 1}
    assert result.usage == Usage()


def test_normalize_passthrough_extraction_result() -> None:
    original = ExtractionResult(value={"a": 1}, usage=Usage(cost_usd=0.5))
    assert normalize_result(original) is original


def test_normalize_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        normalize_result("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_passes_when_outputs_match(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "f", "doc", {"total": 100})
    extractor = FakeExtractor({"doc": {"total": 100}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))
    report = Runner(config).run()
    assert report.passed
    assert len(report.results) == 1


def test_run_fails_on_drift(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "f", "doc", {"total": 100, "vendor": "ACME"})
    extractor = DriftedExtractor(
        {"doc": {"total": 100, "vendor": "ACME"}},
        mutate=lambda v: {**v, "vendor": "Globex"},
    )
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))
    report = Runner(config).run()
    assert not report.passed
    assert report.failing_results


def test_run_applies_tolerances(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "f", "doc", {"vendor": "ACME"})
    extractor = FakeExtractor({"doc": {"vendor": "acme"}})
    config = ERConfig(
        extract_fn=extractor,
        fixtures_dir=str(fixtures_dir),
        tolerances=ToleranceConfig(rules=(ToleranceRule(path="vendor", ignore_case=True),)),
    )
    assert Runner(config).run().passed


def test_run_surfaces_extraction_error(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "f", "doc", {"total": 100})

    def failing(source: object) -> ExtractionResult:
        return ExtractionResult(error="boom")

    config = ERConfig(extract_fn=failing, fixtures_dir=str(fixtures_dir))
    report = Runner(config).run()
    assert not report.passed
    assert report.results[0].error == "boom"


# ---------------------------------------------------------------------------
# record / update
# ---------------------------------------------------------------------------


def test_record_fills_missing_golden_only(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "has", "doc-a", {"total": 1})
    _seed_unrecorded(fixtures_dir, "missing", "doc-b")  # no golden recorded yet
    extractor = FakeExtractor({"doc-a": {"total": 1}, "doc-b": {"total": 2}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    written = Runner(config).record()
    assert written == ["missing"]
    # The previously-present golden was left intact.
    assert FixtureStore(fixtures_dir).load("has").expected == {"total": 1}
    # The un-recorded one was filled.
    assert FixtureStore(fixtures_dir).load("missing").expected == {"total": 2}
    # A coverage snapshot was written.
    assert (fixtures_dir / "coverage_baseline.json").exists()


def test_record_does_not_re_record_empty_golden(fixtures_dir: Path) -> None:
    # A legitimately empty golden ``{}`` already counts as recorded and must NOT
    # be re-recorded on a plain ``record`` run (the bug was treating {} as
    # "no golden" and rewriting it every time).
    _seed(fixtures_dir, "empty", "doc", {})  # explicit empty golden, recorded
    extractor = FakeExtractor({"doc": {"total": 2}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    written = Runner(config).record()
    assert written == []  # the empty golden was left untouched
    assert FixtureStore(fixtures_dir).load("empty").expected == {}


def test_update_overwrites_all_goldens(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "f", "doc", {"total": -1})
    extractor = FakeExtractor({"doc": {"total": 100}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))
    written = Runner(config).update()
    assert written == ["f"]
    assert FixtureStore(fixtures_dir).load("f").expected == {"total": 100}


def test_record_does_not_write_empty_golden_on_error(fixtures_dir: Path) -> None:
    # An extractor that errors must never overwrite a missing golden with {}.
    _seed_unrecorded(fixtures_dir, "f", "doc")  # no golden yet

    def failing(source: object) -> ExtractionResult:
        return ExtractionResult(error="boom")

    config = ERConfig(extract_fn=failing, fixtures_dir=str(fixtures_dir))
    runner = Runner(config)
    written = runner.record()

    assert written == []  # nothing recorded
    assert runner.last_skipped == ("f",)  # the errored fixture was skipped
    # The fixture stays un-recorded (NOT a pinned {} that would FAIL forever).
    assert not FixtureStore(fixtures_dir).load("f").has_golden()
    assert FixtureStore(fixtures_dir).load("f").expected == {}


def test_update_does_not_clobber_golden_on_error(fixtures_dir: Path) -> None:
    # An errored extraction during --update must not wipe a good golden to {}.
    _seed(fixtures_dir, "f", "doc", {"total": 100, "vendor": "ACME"})

    def failing(source: object) -> ExtractionResult:
        return ExtractionResult(error="boom")

    config = ERConfig(extract_fn=failing, fixtures_dir=str(fixtures_dir))
    runner = Runner(config)
    written = runner.update()

    assert written == []
    assert runner.last_skipped == ("f",)
    # The previously-good golden is preserved verbatim.
    assert FixtureStore(fixtures_dir).load("f").expected == {"total": 100, "vendor": "ACME"}


def test_errored_fixture_excluded_from_coverage_sample(fixtures_dir: Path) -> None:
    # A field present in the only healthy fixture must not be dragged to a
    # spurious "drop" by an errored sibling contributing {} to the corpus.
    _seed(fixtures_dir, "ok", "doc-ok", {"total": 1, "tax_id": "X1"})
    _seed(fixtures_dir, "broken", "doc-bad", {"total": 1, "tax_id": "X2"})
    from extract_regress.coverage import write_baseline

    write_baseline(fixtures_dir, {"total": 1.0, "tax_id": 1.0})

    def extract(source: object) -> ExtractionResult:
        text = source if isinstance(source, str) else source.read_text()  # type: ignore[union-attr]
        if text == "doc-bad":
            return ExtractionResult(error="boom")
        return ExtractionResult(value={"total": 1, "tax_id": "X1"})

    config = ERConfig(extract_fn=extract, fixtures_dir=str(fixtures_dir))
    report = Runner(config).run()

    # The healthy fixture keeps tax_id at fill-rate 1.0; no false coverage drop.
    assert not any(d.path == "tax_id" for d in report.dropped_coverage)
    # The errored fixture is still surfaced as a failing result.
    assert any(r.error == "boom" for r in report.results)


# ---------------------------------------------------------------------------
# -k / names filter (#10)
# ---------------------------------------------------------------------------


def test_run_names_filters_to_one_fixture(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "a", "doc-a", {"total": 1})
    _seed(fixtures_dir, "b", "doc-b", {"total": 2})
    extractor = FakeExtractor({"doc-a": {"total": 1}, "doc-b": {"total": 2}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    report = Runner(config).run(names=["a"])

    assert [r.fixture_name for r in report.results] == ["a"]
    # The unselected fixture is never extracted (no wasted/paid call).
    assert extractor.calls == ["doc-a"]


def test_run_names_accepts_multiple_values(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "a", "doc-a", {"total": 1})
    _seed(fixtures_dir, "b", "doc-b", {"total": 2})
    _seed(fixtures_dir, "c", "doc-c", {"total": 3})
    extractor = FakeExtractor({"doc-a": {"total": 1}, "doc-b": {"total": 2}, "doc-c": {"total": 3}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    report = Runner(config).run(names=["a", "c"])

    assert {r.fixture_name for r in report.results} == {"a", "c"}
    assert sorted(extractor.calls) == ["doc-a", "doc-c"]


def test_run_names_unknown_name_raises(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "a", "doc-a", {"total": 1})
    extractor = FakeExtractor({"doc-a": {"total": 1}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    with pytest.raises(FixtureError, match="nonexistent"):
        Runner(config).run(names=["nonexistent"])
    # Nothing was extracted before the error was raised.
    assert extractor.calls == []


def test_record_names_filters_extraction_and_write(fixtures_dir: Path) -> None:
    _seed_unrecorded(fixtures_dir, "a", "doc-a")
    _seed_unrecorded(fixtures_dir, "b", "doc-b")
    extractor = FakeExtractor({"doc-a": {"total": 1}, "doc-b": {"total": 2}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    written = Runner(config).record(names=["a"])

    assert written == ["a"]
    assert extractor.calls == ["doc-a"]  # "b" was never extracted
    assert FixtureStore(fixtures_dir).load("a").expected == {"total": 1}
    assert not FixtureStore(fixtures_dir).load("b").has_golden()


def test_record_names_coverage_sample_includes_full_on_disk_set(fixtures_dir: Path) -> None:
    # "b" already has a golden with a field the filtered fixture lacks; the
    # refreshed coverage snapshot must still count it even though a filtered
    # ``record -k a`` never touches "b".
    _seed_unrecorded(fixtures_dir, "a", "doc-a")
    _seed(fixtures_dir, "b", "doc-b", {"total": 2, "tax_id": "X1"})
    extractor = FakeExtractor({"doc-a": {"total": 1}, "doc-b": {"total": 2, "tax_id": "X1"}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    Runner(config).record(names=["a"])

    from extract_regress.coverage import load_baseline

    baseline = load_baseline(fixtures_dir)
    # "tax_id" appears in 1 of the 2 on-disk fixtures ("b", never extracted
    # this run). A baseline computed from only the filtered subset (just "a")
    # would show 0.0 here instead of the full-corpus 0.5.
    assert baseline["tax_id"] == 0.5


def test_update_names_multiple_values(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "a", "doc-a", {"total": -1})
    _seed(fixtures_dir, "b", "doc-b", {"total": -2})
    _seed(fixtures_dir, "c", "doc-c", {"total": -3})
    extractor = FakeExtractor({"doc-a": {"total": 1}, "doc-b": {"total": 2}, "doc-c": {"total": 3}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    written = Runner(config).update(names=["a", "c"])

    assert sorted(written) == ["a", "c"]
    assert FixtureStore(fixtures_dir).load("a").expected == {"total": 1}
    assert FixtureStore(fixtures_dir).load("c").expected == {"total": 3}
    # "b" was neither extracted nor overwritten.
    assert FixtureStore(fixtures_dir).load("b").expected == {"total": -2}
    assert sorted(extractor.calls) == ["doc-a", "doc-c"]


def test_update_names_unknown_name_raises(fixtures_dir: Path) -> None:
    _seed(fixtures_dir, "a", "doc-a", {"total": 1})
    extractor = FakeExtractor({"doc-a": {"total": 1}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))

    with pytest.raises(FixtureError, match="nonexistent"):
        Runner(config).update(names=["nonexistent"])


# ---------------------------------------------------------------------------
# budget + coverage integration
# ---------------------------------------------------------------------------


def test_run_enforces_budget(fixtures_dir: Path) -> None:
    from extract_regress.budget import BudgetConfig

    _seed(fixtures_dir, "f", "doc", {"total": 100})
    extractor = FakeExtractor(
        {"doc": {"total": 100}}, default_usage=Usage(cost_usd=5.0, latency_ms=10.0)
    )
    config = ERConfig(
        extract_fn=extractor,
        fixtures_dir=str(fixtures_dir),
        budget=BudgetConfig(max_cost_usd_per_run=1.0),
    )
    report = Runner(config).run()
    assert report.budget.failing
    assert not report.passed


def test_run_can_skip_budget(fixtures_dir: Path) -> None:
    from extract_regress.budget import BudgetConfig

    _seed(fixtures_dir, "f", "doc", {"total": 100})
    extractor = FakeExtractor({"doc": {"total": 100}}, default_usage=Usage(cost_usd=5.0))
    config = ERConfig(
        extract_fn=extractor,
        fixtures_dir=str(fixtures_dir),
        budget=BudgetConfig(max_cost_usd_per_run=1.0),
    )
    report = Runner(config).run(check_budget=False)
    assert not report.budget.checked
    assert report.passed


def test_run_detects_coverage_drop(fixtures_dir: Path) -> None:
    from extract_regress.coverage import write_baseline

    _seed(fixtures_dir, "f", "doc", {"total": 100, "tax_id": "X1"})
    # Baseline says tax_id was always present; the extractor now omits it.
    write_baseline(fixtures_dir, {"total": 1.0, "tax_id": 1.0})
    extractor = FakeExtractor({"doc": {"total": 100}})
    config = ERConfig(extract_fn=extractor, fixtures_dir=str(fixtures_dir))
    report = Runner(config).run()
    assert report.dropped_coverage
    assert any(d.path == "tax_id" for d in report.dropped_coverage)
    assert not report.passed
