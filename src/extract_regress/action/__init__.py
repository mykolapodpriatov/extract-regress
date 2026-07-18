"""GitHub Action support for extract-regress.

The composite action (``action.yml``) installs the package and runs
:mod:`extract_regress.action.entrypoint`, which executes an ``extract-regress
run`` against the committed goldens and -- when a token is present -- posts the
field-level diff as a PR comment. The comment step is skipped entirely without a
token, so the action is safe to use on forks and in token-less contexts.
"""

from __future__ import annotations
