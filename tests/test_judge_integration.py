"""End-to-end: the cached judge wired into the diff engine (§3.4 step 5)."""

from __future__ import annotations

from pathlib import Path

from extract_regress.diff import diff_extraction
from extract_regress.judge import CachedJudge
from extract_regress.tolerances import ToleranceConfig, ToleranceRule
from tests._fakes import make_fake_judge


def test_cached_judge_accepts_paraphrase_in_diff(tmp_path: Path) -> None:
    backend = make_fake_judge(True, resolved_model_id="judge-x")
    judge = CachedJudge(backend, cache_dir=tmp_path)
    cfg = ToleranceConfig(rules=(ToleranceRule(path="summary", judge=True),))

    diffs = diff_extraction(
        {"summary": "The cat sat on the mat."},
        {"summary": "A cat was sitting on a mat."},
        cfg,
        judge_fn=judge,
    )
    assert len(diffs) == 1
    assert diffs[0].tolerated
    assert "judge accepted" in diffs[0].reason
    # The verdict was cached, so a second identical run does no extra live work
    # beyond the bootstrap call.
    assert len(backend.seen) == 1  # type: ignore[attr-defined]


def test_cached_judge_rejection_in_diff(tmp_path: Path) -> None:
    backend = make_fake_judge(False)
    judge = CachedJudge(backend, cache_dir=tmp_path)
    cfg = ToleranceConfig(rules=(ToleranceRule(path="summary", judge=True),))
    diffs = diff_extraction(
        {"summary": "Quarterly revenue rose."},
        {"summary": "The office repainted its walls."},
        cfg,
        judge_fn=judge,
    )
    assert diffs[0].failing
    assert "judge rejected" in diffs[0].reason


def test_judge_keyed_by_substrings(tmp_path: Path) -> None:
    # The fake judge can return different verdicts per (expected, actual) pair.
    backend = make_fake_judge({("alpha", "ALPHA"): True, ("beta", "gamma"): False})
    judge = CachedJudge(backend, cache_dir=tmp_path)
    cfg = ToleranceConfig(rules=(ToleranceRule(path="v", judge=True),))

    ok = diff_extraction({"v": "alpha"}, {"v": "ALPHA"}, cfg, judge_fn=judge)
    assert ok[0].tolerated

    bad = diff_extraction({"v": "beta"}, {"v": "gamma"}, cfg, judge_fn=judge)
    assert bad[0].failing
