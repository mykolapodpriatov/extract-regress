# Invoices example

A self-contained, **offline** demo of `extract-regress`. It uses a fake invoice
extractor (`fake_extractor.py`) that returns canned JSON, so it runs with **zero
API keys**.

## Layout

```
examples/invoices/
  conftest.py            # registers the extractor + tolerances via the hook
  fake_extractor.py      # deterministic stand-in for a real LLM extractor
  fixtures/              # golden (source -> expected JSON) fixtures
    samples/             # raw "source documents" (one per invoice)
```

`source_ref` paths are resolved against — and confined to — the fixture's own
directory, so the sample documents live under `fixtures/samples/` rather than a
sibling tree. A `source_ref` that escapes the fixture directory (via `..`, an
absolute path, or a symlink) is rejected.

The goldens deliberately differ from the extractor's raw output in ways a real
pipeline would: vendor casing/whitespace, invoice-date formatting, and totals
rounded to the cent. The tolerances in `conftest.py` absorb exactly those, so
the run passes — while a genuine value regression would still fail.

## Run it

From this directory, with the package installed (`pip install -e ".[dev]"`):

```bash
# Replay the golden fixtures as a test suite:
pytest

# The same check through the CLI (works from any directory):
extract-regress run --project-dir .

# Render the field-diff report as Markdown:
extract-regress run --project-dir . --format md

# (Re)build goldens for any fixtures that lack them:
extract-regress record --project-dir .
```

To see a failure, edit a value in one of the `fixtures/*.json` files (e.g.
change a `total` beyond the `abs_tol`) and re-run — the offending field is
reported with a clear reason.
