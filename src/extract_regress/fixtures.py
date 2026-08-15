"""Fixture model and on-disk store (record / load / update).

A fixture is one JSON file under ``fixtures_dir`` (default
``tests/extract_fixtures/``). Exactly one of ``source_ref`` / ``source_inline``
must be set, and ``source_ref`` resolves relative to the *fixture file's own
directory* so fixtures stay portable across machines and CI (plan §3.3).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import ExtractInput

CURRENT_VERSION = 1
"""On-disk fixture schema version understood by this release."""


class FixtureError(Exception):
    """Raised when a fixture file cannot be loaded or is malformed."""


class RecordedWith(BaseModel):
    """Provenance for a recorded golden (informational + drift warnings)."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt_version: str | None = None
    schema_hash: str | None = None


class Fixture(BaseModel):
    """A single golden ``(source -> expected JSON)`` case."""

    model_config = ConfigDict(extra="forbid")

    version: int = CURRENT_VERSION
    name: str
    source_ref: str | None = None
    source_inline: str | None = None
    source_sha256: str | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    recorded_with: RecordedWith = Field(default_factory=RecordedWith)

    # Populated on load so ``source_ref`` can resolve against the right directory.
    _base_dir: Path | None = None
    # Whether a golden was actually recorded for this fixture. This is distinct
    # from ``expected`` being truthy: an intentionally empty golden ``{}`` is a
    # *recorded* golden and must not be re-recorded every run. The flag is seeded
    # from whether ``expected`` was explicitly supplied (so an empty ``{}`` from a
    # loaded file or an explicit constructor argument counts), and a default
    # ``expected`` (never provided) does not.
    _has_expected: bool = False

    @model_validator(mode="after")
    def _validate_sources(self) -> Fixture:
        has_ref = self.source_ref is not None
        has_inline = self.source_inline is not None
        if has_ref == has_inline:
            raise ValueError(
                "exactly one of 'source_ref' or 'source_inline' must be set "
                f"(fixture {self.name!r})"
            )
        # Seed the golden flag from explicit field provision. ``model_fields_set``
        # is true when ``expected`` was passed to the constructor, to
        # ``model_validate`` (i.e. present as a JSON key), or to ``model_copy``.
        self._has_expected = "expected" in self.model_fields_set
        return self

    def with_base_dir(self, base_dir: Path) -> Fixture:
        """Attach the directory used to resolve ``source_ref``."""
        self._base_dir = base_dir
        return self

    def resolve_source(self) -> ExtractInput:
        """Return the extractor input for this fixture.

        Inline sources are returned verbatim. A ``source_ref`` is resolved
        against the fixture's own directory and returned as a :class:`Path`.

        The resolved path is confined to the fixture's base directory: a
        ``source_ref`` that escapes it via ``..``, an absolute path, or a
        symlink is rejected with a :class:`FixtureError`, so a malicious or
        mistaken fixture cannot read arbitrary files on disk.
        """
        if self.source_inline is not None:
            return self.source_inline
        assert self.source_ref is not None  # guaranteed by validator
        base = (self._base_dir or Path.cwd()).resolve()
        resolved = (base / self.source_ref).resolve()
        if not resolved.is_relative_to(base):
            raise FixtureError(
                f"source_ref {self.source_ref!r} for fixture {self.name!r} escapes the "
                f"fixture directory {base}; references must stay inside it"
            )
        return resolved

    def hash_source(self) -> str:
        """Return the sha256 hex digest of the resolved ``source_ref`` bytes.

        Inline sources have no on-disk file to pin; calling this on one raises
        :class:`FixtureError`.
        """
        source = self.resolve_source()
        if not isinstance(source, Path):
            raise FixtureError(f"cannot hash inline source for fixture {self.name!r}")
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise FixtureError(
                f"cannot read source_ref {self.source_ref!r} for fixture {self.name!r}: {exc}"
            ) from exc
        return hashlib.sha256(data).hexdigest()

    def check_source_digest(self, *, require_digest: bool = False) -> None:
        """Re-hash a ``source_ref`` file and fail if it no longer matches.

        Inline sources are skipped. When ``require_digest`` is true (validate),
        a ``source_ref`` fixture with no ``source_sha256`` is also an error so
        old goldens get backfilled on the next ``update``. A mismatch raises
        :class:`FixtureError` containing ``source drifted`` and the fixture
        name, so callers can fail before treating this as an extraction
        regression.
        """
        if self.source_ref is None:
            return
        self.resolve_source()
        if self.source_sha256 is None:
            if require_digest:
                raise FixtureError(
                    f"source_ref fixture {self.name!r} has no source_sha256; "
                    "re-run update to pin the digest"
                )
            return
        digest = self.hash_source()
        if digest != self.source_sha256:
            raise FixtureError(
                f"source drifted: {self.name} "
                f"(pinned {self.source_sha256}, now {digest})"
            )

    def has_golden(self) -> bool:
        """Whether this fixture already has a recorded golden.

        A golden counts as recorded whenever an ``expected`` value was supplied,
        *including an empty* ``{}``. Only a fixture that never carried an
        ``expected`` key at all is treated as un-recorded, so a legitimately
        empty golden is not silently re-recorded on every ``record`` run.

        ``model_copy(update={"expected": ...})`` (used by record/update) does not
        re-run the validator, so its result is also honored via
        ``model_fields_set``, which that copy updates.
        """
        return self._has_expected or "expected" in self.model_fields_set


def schema_hash(schema: object) -> str:
    """Stable ``sha256:`` digest of a JSON-serializable schema object.

    Used to detect when a user's Pydantic schema changed since recording.
    """
    payload = json.dumps(schema, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class FixtureStore:
    """Filesystem-backed collection of fixtures in ``fixtures_dir``."""

    def __init__(self, fixtures_dir: Path | str) -> None:
        self.fixtures_dir = Path(fixtures_dir)

    def path_for(self, name: str) -> Path:
        """Path of the JSON file backing fixture ``name``."""
        return self.fixtures_dir / f"{name}.json"

    def load(self, name: str) -> Fixture:
        """Load and validate a single fixture by name."""
        path = self.path_for(name)
        return self._load_path(path)

    def load_all(self) -> list[Fixture]:
        """Load every ``*.json`` fixture, sorted by name for determinism."""
        if not self.fixtures_dir.exists():
            return []
        fixtures = [
            self._load_path(p)
            for p in sorted(self.fixtures_dir.glob("*.json"))
            if p.name != "coverage_baseline.json"
        ]
        return fixtures

    def _load_path(self, path: Path) -> Fixture:
        if not path.exists():
            raise FixtureError(f"fixture file not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FixtureError(f"invalid JSON in fixture {path}: {exc}") from exc

        version = raw.get("version")
        if version is not None and version < CURRENT_VERSION:
            raise FixtureError(
                f"fixture {path} uses unsupported legacy version {version}; "
                f"this release reads version {CURRENT_VERSION}. Re-record it with "
                "`extract-regress record` to migrate."
            )
        if version is not None and version > CURRENT_VERSION:
            raise FixtureError(
                f"fixture {path} uses version {version}, newer than this release "
                f"supports ({CURRENT_VERSION}); upgrade extract-regress."
            )

        try:
            fixture = Fixture.model_validate(raw)
        except ValueError as exc:
            raise FixtureError(f"invalid fixture {path}: {exc}") from exc
        return fixture.with_base_dir(path.parent)

    def save(self, fixture: Fixture) -> Path:
        """Persist a fixture to disk, creating ``fixtures_dir`` if needed."""
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(fixture.name)
        payload = fixture.model_dump(mode="json", exclude_none=False)
        # Inline fixtures skip the digest; omit a null so we don't write noise.
        if payload.get("source_sha256") is None:
            payload.pop("source_sha256", None)
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        return path
