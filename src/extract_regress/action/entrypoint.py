"""Token-gated entrypoint for the extract-regress GitHub Action.

This runs inside the composite action. It resolves the project's
``extract_regress_config()`` hook, replays the golden fixtures, renders the
field-level diff as a compact Markdown PR comment, and writes the verdict to
``$GITHUB_OUTPUT``. When a token and a pull-request event are present it posts
the comment via the GitHub REST API; otherwise it prints the comment to stdout
and cleanly skips the API call, so the action works on forks and is fully
testable offline.

Network access is only ever used to post the comment, behind an explicit token
check -- there is no hidden network use, keeping the rest of the tool offline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer

from ..cli import _resolve_config
from ..report import render_pr_comment, summary_line
from ..runner import Runner
from ..types import RunReport

#: Exit code returned when a regression/budget breach is detected.
EXIT_REGRESSION = 1
#: Exit code returned on a configuration/usage error.
EXIT_ERROR = 2


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (``true``/``1``/``yes``/``on``)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _write_output(key: str, value: str) -> None:
    """Append ``key=value`` to ``$GITHUB_OUTPUT`` if it is set."""
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with Path(out_path).open("a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def _run_report() -> RunReport:
    """Resolve the config from the action's environment and replay fixtures."""
    config_module = os.environ.get("ER_CONFIG_MODULE", "conftest.py")
    project_dir = Path(os.environ.get("ER_PROJECT_DIR", ".")).resolve()
    fixtures_dir = os.environ.get("ER_FIXTURES_DIR") or None
    config = _resolve_config(config_module, fixtures_dir, project_dir)
    return Runner(config).run(check_budget=_env_bool("ER_BUDGET", default=True))


def _post_comment(body: str) -> bool:
    """Post ``body`` as a PR comment if a token and PR event are present.

    Returns:
        ``True`` if a comment was posted, ``False`` if skipped (no token,
        comments disabled, not a PR, or the event payload was unavailable).
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token or not _env_bool("ER_COMMENT", default=True):
        return False
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not event_path or not repo or not Path(event_path).exists():
        return False
    try:
        event: dict[str, Any] = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    pr = event.get("pull_request")
    if not isinstance(pr, dict):
        return False
    number = pr.get("number")
    if number is None:
        return False

    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    url = f"{api}/repos/{repo}/issues/{number}/comments"
    return _http_post_comment(url, token, body)


def _http_post_comment(url: str, token: str, body: str) -> bool:
    """POST a comment via urllib; import-guarded so the module loads offline."""
    import urllib.error
    import urllib.request

    payload = json.dumps({"body": body}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "extract-regress-action",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError):  # pragma: no cover - network
        return False


def main() -> int:
    """Run the action: replay, render, optionally comment, mirror the exit code."""
    try:
        report = _run_report()
    except (typer.BadParameter, ImportError, OSError, ValueError, TypeError) as exc:
        sys.stderr.write(f"extract-regress action error: {exc}\n")
        return EXIT_ERROR

    comment_body = render_pr_comment(report)
    sys.stdout.write(comment_body)
    _write_output("passed", "true" if report.passed else "false")

    posted = _post_comment(comment_body)
    if not posted:
        sys.stderr.write("extract-regress: PR comment skipped (no token/PR context)\n")

    if not report.passed:
        sys.stderr.write(summary_line(report) + "\n")
        return EXIT_REGRESSION
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI shim
    raise SystemExit(main())
