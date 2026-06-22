"""Example conftest wiring extract-regress to a zero-dependency fake extractor.

This demo runs entirely offline: ``fake_extractor`` mimics an LLM invoice
extractor by returning canned JSON per source document, so ``pytest`` and the
``extract-regress`` CLI work with no API keys.

Run it from this directory::

    pytest                       # replay the golden fixtures
    extract-regress run          # the same check via the CLI
    extract-regress record       # (re)build goldens for new fixtures
"""

from __future__ import annotations

from fake_extractor import extract_invoice

from extract_regress import ERConfig
from extract_regress.tolerances import ToleranceConfig, ToleranceRule


def extract_regress_config() -> ERConfig:
    """Register the extractor, fixtures directory, and field tolerances."""
    tolerances = ToleranceConfig(
        rules=(
            # Totals may round to the cent.
            ToleranceRule(path="total", abs_tol=0.01),
            ToleranceRule(path="line_items[*].amount", abs_tol=0.01),
            # Invoice dates compare as instants regardless of formatting.
            ToleranceRule(path="invoice_date", as_date=True, date_granularity="day"),
            # Vendor name casing/whitespace is not a regression.
            ToleranceRule(path="vendor", ignore_case=True, ignore_whitespace=True),
        )
    )
    return ERConfig(
        extract_fn=extract_invoice,
        fixtures_dir="fixtures",
        tolerances=tolerances,
    )
