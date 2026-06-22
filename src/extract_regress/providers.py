"""Optional provider-usage wrapper helpers (plan §3.6, §3.7).

These adapters are intentionally thin and lazily import their SDKs, so the core
package keeps zero hard provider dependencies. They turn a raw provider client
into either:

* an :data:`ExtractFn` that returns an :class:`ExtractionResult` with populated
  :class:`Usage` (cost/latency), so budgets work; or
* a :data:`JudgeBackend` returning ``(verdict, resolved_model_id)`` for the
  cached judge — crucially reporting the *concrete* model id the API echoes back.

None of these are exercised by the offline test suite; tests inject fakes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .judge import JudgeBackend
from .types import ExtractionResult, Usage

# A user supplies a function mapping a source to the messages/prompt payload.
PromptBuilder = Callable[[Any], str]


def _affirmative(text: str) -> bool:
    """Interpret a judge model's free-text answer as a boolean verdict."""
    normalized = text.strip().lower()
    return normalized.startswith("y") or normalized.startswith("true")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def openai_judge_backend(
    client: Any,
    *,
    model: str,
    temperature: float = 0.0,
) -> JudgeBackend:
    """Build a :data:`JudgeBackend` from an OpenAI client.

    The returned callable reports ``response.model`` (the concrete, dated model
    id the API echoes) as the resolved model id, so a silent upstream bump
    invalidates the judge cache.
    """

    def backend(prompt: str) -> tuple[bool, str]:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = _affirmative(response.choices[0].message.content or "")
        resolved = getattr(response, "model", model)
        return verdict, resolved

    return backend


def openai_extractor(
    client: Any,
    *,
    model: str,
    build_prompt: PromptBuilder,
    parse: Callable[[str], dict[str, Any]],
    cost_per_call_usd: float | None = None,
) -> Callable[[Any], ExtractionResult]:
    """Wrap an OpenAI client as a budget-aware :data:`ExtractFn`.

    ``parse`` converts the model's text response into the extracted ``dict``.
    Token counts come from ``response.usage``; ``cost_per_call_usd`` (if given)
    feeds the cost budget.
    """

    def extract(source: Any) -> ExtractionResult:
        prompt = build_prompt(source)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        text = response.choices[0].message.content or ""
        usage_obj = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", None),
            completion_tokens=getattr(usage_obj, "completion_tokens", None),
            cost_usd=cost_per_call_usd,
            latency_ms=latency_ms,
        )
        try:
            value = parse(text)
            return ExtractionResult(value=value, usage=usage)
        except Exception as exc:
            return ExtractionResult(usage=usage, error=f"parse error: {exc}")

    return extract


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def anthropic_judge_backend(
    client: Any,
    *,
    model: str,
    max_tokens: int = 16,
) -> JudgeBackend:
    """Build a :data:`JudgeBackend` from an Anthropic client.

    Reports ``response.model`` (the concrete model id) as the resolved id.
    """

    def backend(prompt: str) -> tuple[bool, str]:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(block, "text", "") for block in getattr(response, "content", []))
        verdict = _affirmative(text)
        resolved = getattr(response, "model", model)
        return verdict, resolved

    return backend


# ---------------------------------------------------------------------------
# Ollama (local; cost is zero, latency is measured)
# ---------------------------------------------------------------------------


def ollama_judge_backend(
    client: Any,
    *,
    model: str,
) -> JudgeBackend:
    """Build a :data:`JudgeBackend` from an Ollama client.

    Ollama runs locally; the resolved model id is the requested ``model`` tag
    (digests are not echoed by the chat API), so bumping the local model
    requires bumping ``judge_version`` to invalidate the cache.
    """

    def backend(prompt: str) -> tuple[bool, str]:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response["message"]["content"]
        return _affirmative(text), getattr(response, "model", None) or model

    return backend
