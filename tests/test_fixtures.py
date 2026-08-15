"""Tests for the fixture model and on-disk store (§3.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from extract_regress.fixtures import (
    CURRENT_VERSION,
    Fixture,
    FixtureError,
    FixtureStore,
    schema_hash,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(fixtures_dir: Path) -> None:
    store = FixtureStore(fixtures_dir)
    fixture = Fixture(
        name="invoice_basic",
        source_inline="INVOICE total 100",
        expected={"total": 100},
    )
    store.save(fixture)
    loaded = store.load("invoice_basic")
    assert loaded.name == "invoice_basic"
    assert loaded.expected == {"total": 100}
    assert loaded.version == CURRENT_VERSION


def test_load_all_is_sorted_and_skips_baseline(fixtures_dir: Path) -> None:
    store = FixtureStore(fixtures_dir)
    store.save(Fixture(name="b", source_inline="x", expected={"v": 1}))
    store.save(Fixture(name="a", source_inline="y", expected={"v": 2}))
    _write(fixtures_dir / "coverage_baseline.json", {"v": 1.0})
    names = [f.name for f in store.load_all()]
    assert names == ["a", "b"]


def test_load_all_empty_dir_returns_empty(tmp_path: Path) -> None:
    store = FixtureStore(tmp_path / "does_not_exist")
    assert store.load_all() == []


# ---------------------------------------------------------------------------
# Mutual exclusivity of source fields
# ---------------------------------------------------------------------------


def test_both_sources_set_is_error(fixtures_dir: Path) -> None:
    _write(
        fixtures_dir / "bad.json",
        {"name": "bad", "source_ref": "a.txt", "source_inline": "x", "expected": {}},
    )
    with pytest.raises(FixtureError, match="exactly one of"):
        FixtureStore(fixtures_dir).load("bad")


def test_neither_source_set_is_error(fixtures_dir: Path) -> None:
    _write(fixtures_dir / "bad.json", {"name": "bad", "expected": {}})
    with pytest.raises(FixtureError, match="exactly one of"):
        FixtureStore(fixtures_dir).load("bad")


# ---------------------------------------------------------------------------
# source_ref resolution
# ---------------------------------------------------------------------------


def test_source_ref_resolves_relative_to_fixture_dir(tmp_path: Path) -> None:
    fdir = tmp_path / "fixtures"
    fdir.mkdir()
    (fdir / "doc.txt").write_text("hello", encoding="utf-8")
    _write(
        fdir / "case.json",
        {"name": "case", "source_ref": "doc.txt", "expected": {}},
    )
    # Resolution must not depend on the process cwd.
    fixture = FixtureStore(fdir).load("case")
    resolved = fixture.resolve_source()
    assert isinstance(resolved, Path)
    assert resolved == (fdir / "doc.txt").resolve()
    assert resolved.read_text(encoding="utf-8") == "hello"


def test_source_inline_returns_text(fixtures_dir: Path) -> None:
    store = FixtureStore(fixtures_dir)
    store.save(Fixture(name="inline", source_inline="raw text", expected={}))
    assert store.load("inline").resolve_source() == "raw text"


def test_source_ref_cannot_escape_via_parent_dirs(tmp_path: Path) -> None:
    # A secret outside the fixture tree must remain unreadable.
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    fdir = tmp_path / "fixtures"
    fdir.mkdir()
    _write(
        fdir / "evil.json",
        {"name": "evil", "source_ref": "../secret.txt", "expected": {}},
    )
    fixture = FixtureStore(fdir).load("evil")
    with pytest.raises(FixtureError, match="escapes the fixture directory"):
        fixture.resolve_source()


def test_source_ref_cannot_be_absolute_path(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    fdir = tmp_path / "fixtures"
    fdir.mkdir()
    _write(
        fdir / "evil.json",
        {"name": "evil", "source_ref": str(secret), "expected": {}},
    )
    fixture = FixtureStore(fdir).load("evil")
    with pytest.raises(FixtureError, match="escapes the fixture directory"):
        fixture.resolve_source()


def _write_ref_fixture(
    fixtures_dir: Path,
    name: str,
    text: str,
    *,
    digest: str | None = "",
) -> Path:
    """Write a source_ref fixture and its sample file.

    ``digest=""`` (default) pins the current file hash. ``digest=None`` omits
    ``source_sha256`` (legacy golden). Any other string is stored as-is.
    """
    samples = fixtures_dir / "samples"
    samples.mkdir(exist_ok=True)
    src = samples / f"{name}.txt"
    src.write_text(text, encoding="utf-8")
    payload: dict[str, object] = {
        "version": 1,
        "name": name,
        "source_ref": f"samples/{name}.txt",
        "expected": {"total": 1},
    }
    if digest is None:
        pass
    elif digest == "":
        payload["source_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    else:
        payload["source_sha256"] = digest
    _write(fixtures_dir / f"{name}.json", payload)
    return src


def test_source_digest_unchanged_file_passes(tmp_path: Path) -> None:
    fdir = tmp_path / "fixtures"
    fdir.mkdir()
    _write_ref_fixture(fdir, "invoice_basic", "hello")
    fixture = FixtureStore(fdir).load("invoice_basic")
    fixture.check_source_digest(require_digest=True)
    assert fixture.source_sha256 == hashlib.sha256(b"hello").hexdigest()


def test_source_digest_one_byte_edit_fails_with_fixture_name(tmp_path: Path) -> None:
    fdir = tmp_path / "fixtures"
    fdir.mkdir()
    src = _write_ref_fixture(fdir, "invoice_basic", "hello")
    src.write_text("hellp", encoding="utf-8")
    fixture = FixtureStore(fdir).load("invoice_basic")
    with pytest.raises(FixtureError, match="source drifted") as excinfo:
        fixture.check_source_digest()
    assert "invoice_basic" in str(excinfo.value)


def test_source_ref_without_digest_fails_when_required(tmp_path: Path) -> None:
    fdir = tmp_path / "fixtures"
    fdir.mkdir()
    _write_ref_fixture(fdir, "legacy", "hello", digest=None)
    fixture = FixtureStore(fdir).load("legacy")
    with pytest.raises(FixtureError, match="no source_sha256"):
        fixture.check_source_digest(require_digest=True)
    fixture.check_source_digest(require_digest=False)


def test_inline_source_skips_digest(fixtures_dir: Path) -> None:
    fixture = Fixture(name="inline", source_inline="raw", expected={})
    fixture.check_source_digest(require_digest=True)
    assert fixture.source_sha256 is None


def test_source_ref_cannot_escape_via_symlink(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    fdir = tmp_path / "fixtures"
    fdir.mkdir()
    # A symlink that lives inside the fixture dir but points outside it.
    (fdir / "link.txt").symlink_to(secret)
    _write(
        fdir / "evil.json",
        {"name": "evil", "source_ref": "link.txt", "expected": {}},
    )
    fixture = FixtureStore(fdir).load("evil")
    with pytest.raises(FixtureError, match="escapes the fixture directory"):
        fixture.resolve_source()


# ---------------------------------------------------------------------------
# Versioning / forward-compat
# ---------------------------------------------------------------------------


def test_legacy_version_zero_errors_with_clear_message(fixtures_dir: Path) -> None:
    _write(
        fixtures_dir / "legacy.json",
        {"version": 0, "name": "legacy", "source_inline": "x", "expected": {}},
    )
    with pytest.raises(FixtureError, match="legacy version 0"):
        FixtureStore(fixtures_dir).load("legacy")


def test_future_version_errors(fixtures_dir: Path) -> None:
    _write(
        fixtures_dir / "future.json",
        {
            "version": CURRENT_VERSION + 1,
            "name": "future",
            "source_inline": "x",
            "expected": {},
        },
    )
    with pytest.raises(FixtureError, match="newer than this release"):
        FixtureStore(fixtures_dir).load("future")


def test_missing_file_errors(fixtures_dir: Path) -> None:
    with pytest.raises(FixtureError, match="not found"):
        FixtureStore(fixtures_dir).load("nope")


def test_invalid_json_errors(fixtures_dir: Path) -> None:
    (fixtures_dir / "broken.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(FixtureError, match="invalid JSON"):
        FixtureStore(fixtures_dir).load("broken")


# ---------------------------------------------------------------------------
# schema_hash
# ---------------------------------------------------------------------------


def test_schema_hash_is_stable_and_order_independent() -> None:
    a = schema_hash({"a": 1, "b": 2})
    b = schema_hash({"b": 2, "a": 1})
    assert a == b
    assert a.startswith("sha256:")


def test_schema_hash_changes_with_schema() -> None:
    assert schema_hash({"a": 1}) != schema_hash({"a": 2})


def test_has_golden_reflects_expected() -> None:
    assert not Fixture(name="x", source_inline="s").has_golden()
    assert Fixture(name="x", source_inline="s", expected={"k": 1}).has_golden()


def test_empty_golden_is_recorded_and_not_re_recorded(fixtures_dir: Path) -> None:
    # A legitimately empty golden ``{}`` must count as recorded so ``record``
    # does not silently re-record it every run. Constructing with an explicit
    # empty ``expected`` and round-tripping through disk both report has_golden.
    store = FixtureStore(fixtures_dir)
    store.save(Fixture(name="empty", source_inline="s", expected={}))

    reloaded = store.load("empty")
    assert reloaded.expected == {}
    assert reloaded.has_golden(), "an empty {} golden must be treated as recorded"


def test_fixture_without_expected_key_is_unrecorded(fixtures_dir: Path) -> None:
    # A fixture file that never carried an ``expected`` key at all is the only
    # case that should be considered un-recorded (eligible for ``record``).
    _write(
        fixtures_dir / "fresh.json",
        {"version": 1, "name": "fresh", "source_inline": "s"},
    )
    fixture = FixtureStore(fixtures_dir).load("fresh")
    assert fixture.expected == {}
    assert not fixture.has_golden()


def test_existing_fixture_with_expected_key_migrates_to_recorded(fixtures_dir: Path) -> None:
    # Migration: existing fixtures that already have an ``expected`` key present
    # on disk (empty or not) must continue to read as having a golden.
    _write(
        fixtures_dir / "legacy_empty.json",
        {"version": 1, "name": "legacy_empty", "source_inline": "s", "expected": {}},
    )
    _write(
        fixtures_dir / "legacy_full.json",
        {"version": 1, "name": "legacy_full", "source_inline": "s", "expected": {"k": 1}},
    )
    store = FixtureStore(fixtures_dir)
    assert store.load("legacy_empty").has_golden()
    assert store.load("legacy_full").has_golden()


def test_model_copy_with_expected_update_is_recorded() -> None:
    # The record/update path uses ``model_copy(update={"expected": ...})``; its
    # result must report a recorded golden even though the after-validator is not
    # re-run by ``model_copy``.
    base = Fixture(name="c", source_inline="s")
    assert not base.has_golden()
    assert base.model_copy(update={"expected": {}}).has_golden()
    assert base.model_copy(update={"expected": {"k": 1}}).has_golden()
