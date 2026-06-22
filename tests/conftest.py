"""pytest fixtures and re-exports of the shared offline fakes.

The fakes live in :mod:`tests._fakes` so they can be imported directly by test
modules under any runner (plain ``pytest`` or ``coverage run -m pytest``); this
module re-exports them for convenience and provides the ``fixtures_dir`` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._fakes import DriftedExtractor, FakeExtractor, make_fake_judge

__all__ = ["DriftedExtractor", "FakeExtractor", "make_fake_judge"]


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """An empty, writable fixtures directory under the test's tmp_path."""
    target = tmp_path / "extract_fixtures"
    target.mkdir()
    return target
