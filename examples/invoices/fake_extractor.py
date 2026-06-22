"""A deterministic, offline stand-in for an LLM invoice extractor.

In a real project this function would call your model (e.g. via the
``openai_extractor`` / ``anthropic_judge_backend`` helpers in
``extract_regress.providers``). Here it returns canned JSON keyed by the source
text so the example needs no network or API keys, while still exercising the
full record / replay / diff loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from extract_regress.types import ExtractInput

# Canned model outputs, keyed by the (stripped) source document text. The slight
# formatting differences from the goldens (casing, cents, date format) are what
# the configured tolerances absorb.
_OUTPUTS: dict[str, dict[str, Any]] = {
    "acme": {
        "vendor": "ACME Corporation",
        "invoice_number": "INV-1001",
        "invoice_date": "2024-03-15",
        "total": 1250.00,
        "line_items": [
            {"description": "Widget", "amount": 1000.00},
            {"description": "Shipping", "amount": 250.00},
        ],
    },
    "globex": {
        "vendor": "  globex inc  ",  # different casing/whitespace vs golden
        "invoice_number": "GBX-77",
        "invoice_date": "2024-04-01T00:00:00",  # different date format vs golden
        "total": 499.995,  # rounds to the golden's 500.00 within abs_tol
        "line_items": [
            {"description": "Subscription", "amount": 499.995},
        ],
    },
}


def _read(source: ExtractInput) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8").strip()
    if isinstance(source, bytes):
        return source.decode("utf-8").strip()
    return source.strip()


def extract_invoice(source: ExtractInput) -> dict[str, Any]:
    """Return the canned extraction for ``source`` (raw text or a file path)."""
    key = _read(source).splitlines()[0].strip().lower()
    if key not in _OUTPUTS:
        raise KeyError(f"no canned output for invoice {key!r}")
    return _OUTPUTS[key]
