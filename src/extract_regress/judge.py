"""Optional cached LLM-judge for free-text fields (plan §3.7).

Only fields whose resolved tolerance sets ``judge=True`` reach the judge, and
only when both sides are strings (the diff engine enforces this). The judge is
made deterministic for CI by a committable verdict cache.

**Cache key** = ``sha256(judge_prompt_text + resolved_model_id + judge_version +
field_path + expected + actual)``. ``resolved_model_id`` is the concrete model
string the backend reports *after its first call* (never a ``*-latest`` alias),
so a silent upstream model bump invalidates the cache instead of returning a
stale verdict. ``judge_version`` is bumped by the user when the prompt changes.

**Bootstrapping protocol:** the full key is uncomputable until the resolved
model id is known, so for each eligible field the backend is invoked once to get
``(verdict, resolved_model_id)``, and only *then* is the cache consulted. On a
hit the stored verdict wins and the freshly computed one is discarded; the first
call for any key is therefore always live. Implementations MUST NOT short-circuit
by reading the cache before the resolved model id is known.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from .types import JudgeFn

DEFAULT_CACHE_DIR = ".extract_regress_cache"
CACHE_FILENAME = "judge.json"

# A backend takes the rendered prompt and returns ``(verdict, resolved_model_id)``.
JudgeBackend = Callable[[str], tuple[bool, str]]


def default_prompt(field_path: str, expected: str, actual: str) -> str:
    """Render the default judge prompt for an ``(expected, actual)`` pair."""
    return (
        "You are grading whether two extracted values are semantically "
        "equivalent for the same field.\n"
        f"Field: {field_path}\n"
        f"Expected: {expected}\n"
        f"Actual: {actual}\n"
        "Answer strictly 'yes' if they mean the same thing, otherwise 'no'."
    )


def compute_cache_key(
    *,
    prompt_text: str,
    resolved_model_id: str,
    judge_version: int,
    field_path: str,
    expected: str,
    actual: str,
) -> str:
    """Deterministic ``sha256`` cache key binding the resolved model id.

    The concrete ``resolved_model_id`` (not a user alias) and ``judge_version``
    are part of the key, so a model bump or prompt edit cleanly invalidates the
    cache rather than returning a stale verdict.

    Components are JSON-encoded as a list before hashing so the boundaries are
    unambiguous: no separator character (such as ``\\x00``) appearing inside a
    component can make two distinct component tuples hash to the same key.
    """
    payload = json.dumps(
        [
            prompt_text,
            resolved_model_id,
            judge_version,
            field_path,
            expected,
            actual,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """JSON-file verdict cache, committable for deterministic CI."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / CACHE_FILENAME
        self._data: dict[str, bool] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = {str(k): bool(v) for k, v in raw.items()}
        self._loaded = True

    def get(self, key: str) -> bool | None:
        """Return the cached verdict for ``key`` or ``None`` on a miss."""
        self._ensure_loaded()
        return self._data.get(key)

    def set(self, key: str, verdict: bool) -> None:
        """Store ``verdict`` for ``key`` and persist the cache to disk."""
        self._ensure_loaded()
        self._data[key] = verdict
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ordered = {k: self._data[k] for k in sorted(self._data)}
        self.path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")

    def __contains__(self, key: str) -> bool:
        self._ensure_loaded()
        return key in self._data


class CachedJudge:
    """Wraps a :data:`JudgeBackend` with the cached, deterministic protocol.

    The resulting instance is itself a :data:`~extract_regress.types.JudgeFn`
    (``(field_path, expected, actual) -> (verdict, resolved_model_id)``) and can
    be handed straight to the diff engine.
    """

    def __init__(
        self,
        backend: JudgeBackend,
        *,
        judge_version: int = 1,
        cache: JudgeCache | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        prompt_builder: Callable[[str, str, str], str] = default_prompt,
    ) -> None:
        self.backend = backend
        self.judge_version = judge_version
        self.cache = cache if cache is not None else JudgeCache(cache_dir)
        self.prompt_builder = prompt_builder

    def __call__(self, field_path: str, expected: str, actual: str) -> tuple[bool, str]:
        """Evaluate one pair, honoring the bootstrapping cache protocol."""
        prompt = self.prompt_builder(field_path, expected, actual)

        # Bootstrap: a live call is required to learn the resolved model id
        # before the cache key can exist. The cache is consulted only after.
        live_verdict, resolved_model_id = self.backend(prompt)

        key = compute_cache_key(
            prompt_text=prompt,
            resolved_model_id=resolved_model_id,
            judge_version=self.judge_version,
            field_path=field_path,
            expected=expected,
            actual=actual,
        )

        cached = self.cache.get(key)
        if cached is not None:
            # Cache hit: the stored verdict wins; discard the live one.
            return cached, resolved_model_id

        self.cache.set(key, live_verdict)
        return live_verdict, resolved_model_id


def make_judge(
    backend: JudgeBackend,
    *,
    judge_version: int = 1,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> JudgeFn:
    """Convenience factory returning a ready-to-use cached :data:`JudgeFn`."""
    return CachedJudge(backend, judge_version=judge_version, cache_dir=cache_dir)
