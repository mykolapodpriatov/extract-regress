"""Shared offline test fakes (importable as ``tests._fakes``).

Everything here is deterministic and network-free, per the plan's determinism
strategy (§4). ``FakeExtractor`` returns canned values; ``DriftedExtractor``
mutates them to exercise regression paths. ``make_fake_judge`` builds a judge
backend so the judge tests never touch a provider.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from extract_regress.types import ExtractInput, ExtractionResult, Usage


class FakeExtractor:
    """A canned extractor keyed by source text/inline content.

    The same input always yields the same output, so replays are deterministic.
    Optional ``usage`` makes budgets exercisable without a real provider.
    """

    def __init__(
        self,
        responses: Mapping[str, dict[str, Any]],
        *,
        usage: Mapping[str, Usage] | None = None,
        default_usage: Usage | None = None,
    ) -> None:
        self._responses = dict(responses)
        self._usage = dict(usage or {})
        self._default_usage = default_usage
        self.calls: list[str] = []

    def _key(self, source: ExtractInput) -> str:
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8")
        if isinstance(source, bytes):
            return source.decode("utf-8")
        return source

    def __call__(self, source: ExtractInput) -> dict[str, Any] | ExtractionResult:
        key = self._key(source)
        self.calls.append(key)
        value = self._responses[key]
        usage = self._usage.get(key, self._default_usage)
        if usage is not None:
            return ExtractionResult(value=value, usage=usage)
        return value


class DriftedExtractor(FakeExtractor):
    """A :class:`FakeExtractor` whose outputs are transformed to simulate drift.

    The ``mutate`` callable receives the canned value and returns the drifted
    one (e.g. drop a field, change a number, paraphrase a string).
    """

    def __init__(
        self,
        responses: Mapping[str, dict[str, Any]],
        mutate: Callable[[dict[str, Any]], dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(responses, **kwargs)
        self._mutate = mutate

    def __call__(self, source: ExtractInput) -> dict[str, Any] | ExtractionResult:
        result = super().__call__(source)
        if isinstance(result, ExtractionResult):
            return result.model_copy(update={"value": self._mutate(dict(result.value))})
        return self._mutate(dict(result))


def make_fake_judge(
    verdicts: Mapping[tuple[str, str], bool] | bool,
    resolved_model_id: str = "fake-judge-1",
) -> Callable[[str], tuple[bool, str]]:
    """Build a fake :data:`JudgeBackend` that never hits the network.

    ``verdicts`` may be a single boolean (applied to every prompt) or a mapping
    keyed by ``(expected_substr, actual_substr)`` for fine-grained control. The
    returned backend records each prompt it sees in ``backend.seen``.
    """
    seen: list[str] = []

    def backend(prompt: str) -> tuple[bool, str]:
        seen.append(prompt)
        if isinstance(verdicts, bool):
            return verdicts, resolved_model_id
        for (exp_sub, act_sub), verdict in verdicts.items():
            if exp_sub in prompt and act_sub in prompt:
                return verdict, resolved_model_id
        return False, resolved_model_id

    backend.seen = seen  # type: ignore[attr-defined]
    return backend
