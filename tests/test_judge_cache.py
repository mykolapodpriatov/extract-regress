"""Tests for the cached LLM-judge and its invalidation contract (§3.7)."""

from __future__ import annotations

from pathlib import Path

from extract_regress.judge import CachedJudge, JudgeCache, compute_cache_key


class CountingBackend:
    """A fake :data:`JudgeBackend` that counts live calls and is fully offline.

    ``resolved_model_id`` can be flipped between calls to simulate a silent
    upstream model bump.
    """

    def __init__(self, verdict: bool = True, resolved_model_id: str = "model-A") -> None:
        self.verdict = verdict
        self.resolved_model_id = resolved_model_id
        self.calls = 0

    def __call__(self, prompt: str) -> tuple[bool, str]:
        self.calls += 1
        return self.verdict, self.resolved_model_id


# ---------------------------------------------------------------------------
# Determinism: a cache hit returns the stored verdict
# ---------------------------------------------------------------------------


def test_cache_hit_avoids_second_persisted_verdict(tmp_path: Path) -> None:
    backend = CountingBackend(verdict=True)
    judge = CachedJudge(backend, judge_version=1, cache_dir=tmp_path)

    v1, m1 = judge("summary", "Cats are great.", "Cats are wonderful.")
    assert v1 is True and m1 == "model-A"
    assert backend.calls == 1

    # The cache file now holds the verdict for this key.
    cache = JudgeCache(tmp_path)
    assert len(cache._data) == 0  # not yet loaded
    key = compute_cache_key(
        prompt_text=judge.prompt_builder("summary", "Cats are great.", "Cats are wonderful."),
        resolved_model_id="model-A",
        judge_version=1,
        field_path="summary",
        expected="Cats are great.",
        actual="Cats are wonderful.",
    )
    assert cache.get(key) is True


def test_cache_hit_returns_stored_verdict_not_live(tmp_path: Path) -> None:
    # Seed the cache with True via a first judge that returns True.
    seed_backend = CountingBackend(verdict=True, resolved_model_id="model-A")
    CachedJudge(seed_backend, judge_version=1, cache_dir=tmp_path)("summary", "exp", "act")

    # A second judge with the SAME resolved model id but a flipped live verdict
    # must still return the cached True (the live False is discarded on a hit).
    flipped_backend = CountingBackend(verdict=False, resolved_model_id="model-A")
    judge = CachedJudge(flipped_backend, judge_version=1, cache_dir=tmp_path)
    verdict, _ = judge("summary", "exp", "act")
    assert verdict is True  # cached value wins
    assert flipped_backend.calls == 1  # bootstrap call still happened


# ---------------------------------------------------------------------------
# Bootstrapping protocol: the first call for any key is always live
# ---------------------------------------------------------------------------


def test_first_call_is_always_live(tmp_path: Path) -> None:
    backend = CountingBackend(verdict=True)
    judge = CachedJudge(backend, cache_dir=tmp_path)
    judge("f", "a", "b")
    assert backend.calls == 1, "the resolved model id is unknown until a live call"


def test_repeated_identical_pairs_still_bootstrap_then_serve_cache(tmp_path: Path) -> None:
    backend = CountingBackend(verdict=True)
    judge = CachedJudge(backend, cache_dir=tmp_path)
    judge("f", "a", "b")
    judge("f", "a", "b")
    # Each invocation makes its bootstrap call, but the verdict is served from
    # cache on the second (verified by flipping verdict in the other test).
    assert backend.calls == 2


# ---------------------------------------------------------------------------
# Invalidation: a stale verdict must NOT be reused
# ---------------------------------------------------------------------------


def test_resolved_model_bump_invalidates_cache(tmp_path: Path) -> None:
    # Seed cache with model-A -> True.
    CachedJudge(CountingBackend(True, "model-A"), cache_dir=tmp_path)("f", "a", "b")

    # Same pair, but the backend now resolves to model-B and lives-votes False.
    # Because the resolved id is part of the key, this is a MISS, so the new
    # live False is used and stored — no stale True returned.
    backend_b = CountingBackend(False, "model-B")
    judge = CachedJudge(backend_b, cache_dir=tmp_path)
    verdict, model_id = judge("f", "a", "b")
    assert model_id == "model-B"
    assert verdict is False, "model bump must not reuse the stale True verdict"


def test_judge_version_bump_invalidates_cache(tmp_path: Path) -> None:
    # Seed cache at version 1 with True.
    CachedJudge(CountingBackend(True, "model-A"), judge_version=1, cache_dir=tmp_path)(
        "f", "a", "b"
    )
    # Version 2 with a live False is a different key → False is used.
    judge = CachedJudge(CountingBackend(False, "model-A"), judge_version=2, cache_dir=tmp_path)
    verdict, _ = judge("f", "a", "b")
    assert verdict is False, "judge_version bump must not reuse the stale verdict"


def test_cache_key_components_change_the_key() -> None:
    base: dict[str, object] = {
        "prompt_text": "p",
        "resolved_model_id": "m",
        "judge_version": 1,
        "field_path": "f",
        "expected": "e",
        "actual": "a",
    }
    baseline = compute_cache_key(**base)
    assert compute_cache_key(**{**base, "resolved_model_id": "m2"}) != baseline
    assert compute_cache_key(**{**base, "judge_version": 2}) != baseline
    assert compute_cache_key(**{**base, "expected": "e2"}) != baseline
    assert compute_cache_key(**{**base, "actual": "a2"}) != baseline
    assert compute_cache_key(**{**base, "field_path": "f2"}) != baseline


def test_cache_key_no_collision_on_null_byte_components() -> None:
    # Under naive "\x00".join, ("a\x00b", "c") and ("a", "b\x00c") collide on
    # field_path/expected boundaries. The encoding must keep them distinct.
    base: dict[str, object] = {
        "prompt_text": "p",
        "resolved_model_id": "m",
        "judge_version": 1,
        "actual": "a",
    }
    key1 = compute_cache_key(field_path="a\x00b", expected="c", **base)
    key2 = compute_cache_key(field_path="a", expected="b\x00c", **base)
    assert key1 != key2


def test_cache_key_no_collision_across_field_boundaries() -> None:
    # The "\x00"-shift collision generalizes: moving a "\x00"-prefixed chunk from
    # one component to the next must not yield the same key.
    a = compute_cache_key(
        prompt_text="x",
        resolved_model_id="m",
        judge_version=1,
        field_path="f",
        expected="e\x00",
        actual="extra",
    )
    b = compute_cache_key(
        prompt_text="x",
        resolved_model_id="m",
        judge_version=1,
        field_path="f",
        expected="e",
        actual="\x00extra",
    )
    assert a != b


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    CachedJudge(CountingBackend(True, "m"), cache_dir=tmp_path)("f", "a", "b")
    # A brand-new cache object loads the committed verdict from disk.
    reopened = CachedJudge(CountingBackend(False, "m"), cache_dir=tmp_path)
    verdict, _ = reopened("f", "a", "b")
    assert verdict is True


def test_cache_contains_membership(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path)
    assert "missing" not in cache
    cache.set("present", True)
    assert "present" in cache


def test_make_judge_factory_returns_callable(tmp_path: Path) -> None:
    from extract_regress.judge import make_judge

    judge = make_judge(CountingBackend(True, "m"), judge_version=2, cache_dir=tmp_path)
    verdict, model_id = judge("field", "exp", "act")
    assert verdict is True
    assert model_id == "m"
