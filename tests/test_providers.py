"""Tests for provider wrapper helpers using fully offline fake clients.

No real SDK or network is involved; each fake mimics just the surface the
adapter touches. These verify that the *resolved* model id (not the requested
alias) is reported, which is what makes the judge cache invalidation correct.
"""

from __future__ import annotations

from typing import Any

from extract_regress.providers import (
    anthropic_judge_backend,
    ollama_judge_backend,
    openai_extractor,
    openai_judge_backend,
)
from extract_regress.types import ExtractionResult

# ---------------------------------------------------------------------------
# Fakes mirroring SDK response shapes
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _OpenAIResponse:
    def __init__(self, content: str, model: str, usage: Any = None) -> None:
        self.choices = [_Msg(content)]
        self.model = model
        self.usage = usage


class FakeOpenAIClient:
    def __init__(self, content: str, resolved_model: str, usage: Any = None) -> None:
        self._content = content
        self._resolved = resolved_model
        self._usage = usage
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs: Any) -> _OpenAIResponse:
        return _OpenAIResponse(self._content, self._resolved, self._usage)


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _AnthropicResponse:
    def __init__(self, text: str, model: str) -> None:
        self.content = [_Block(text)]
        self.model = model


class FakeAnthropicClient:
    def __init__(self, text: str, resolved_model: str) -> None:
        self._text = text
        self._resolved = resolved_model
        self.messages = self

    def create(self, **kwargs: Any) -> _AnthropicResponse:
        return _AnthropicResponse(self._text, self._resolved)


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self._content = content

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        return {"message": {"content": self._content}}


# ---------------------------------------------------------------------------
# Judge backends report the *resolved* model id
# ---------------------------------------------------------------------------


def test_openai_judge_backend_reports_resolved_model() -> None:
    client = FakeOpenAIClient("yes", resolved_model="gpt-4o-mini-2024-07-18")
    backend = openai_judge_backend(client, model="gpt-4o-mini")
    verdict, resolved = backend("prompt")
    assert verdict is True
    assert resolved == "gpt-4o-mini-2024-07-18"  # concrete id, not the alias


def test_openai_judge_backend_negative() -> None:
    backend = openai_judge_backend(FakeOpenAIClient("no", "m"), model="m")
    verdict, _ = backend("prompt")
    assert verdict is False


def test_anthropic_judge_backend_reports_resolved_model() -> None:
    client = FakeAnthropicClient("Yes.", resolved_model="claude-3-5-sonnet-20241022")
    backend = anthropic_judge_backend(client, model="claude-3-5-sonnet-latest")
    verdict, resolved = backend("prompt")
    assert verdict is True
    assert resolved == "claude-3-5-sonnet-20241022"


def test_ollama_judge_backend() -> None:
    backend = ollama_judge_backend(FakeOllamaClient("true"), model="llama3")
    verdict, resolved = backend("prompt")
    assert verdict is True
    assert resolved == "llama3"


# ---------------------------------------------------------------------------
# OpenAI extractor populates Usage for budgets
# ---------------------------------------------------------------------------


def test_openai_extractor_populates_usage() -> None:
    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
    client = FakeOpenAIClient('{"total": 100}', "gpt-4o", usage=usage)
    extract = openai_extractor(
        client,
        model="gpt-4o",
        build_prompt=lambda src: f"extract from {src}",
        parse=lambda text: __import__("json").loads(text),
        cost_per_call_usd=0.002,
    )
    result = extract("a document")
    assert isinstance(result, ExtractionResult)
    assert result.value == {"total": 100}
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.cost_usd == 0.002
    assert result.usage.latency_ms is not None and result.usage.latency_ms >= 0.0


def test_openai_extractor_reports_parse_error() -> None:
    client = FakeOpenAIClient("not json", "gpt-4o")
    extract = openai_extractor(
        client,
        model="gpt-4o",
        build_prompt=lambda src: "p",
        parse=lambda text: __import__("json").loads(text),
    )
    result = extract("doc")
    assert result.error is not None
    assert "parse error" in result.error
