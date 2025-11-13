"""Centralized helpers for analysis output directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

_REPORTS_ENV_VAR = "PANCSVRIEND_REPORTS_DIR"


def get_reports_dir() -> Path:
    """Return the base directory for analysis outputs.

    The path defaults to ``reports`` relative to the current working directory
    but respects the ``PANCSVRIEND_REPORTS_DIR`` environment variable when set.
    The returned value is expanded for ``~`` to support user home shortcuts.
    """
    configured = os.environ.get(_REPORTS_ENV_VAR, "reports")
    return Path(configured).expanduser()


def set_reports_dir(path: Union[str, Path]) -> Path:
    """Update the reports directory environment override and return the path."""
    resolved = Path(path).expanduser()
    os.environ[_REPORTS_ENV_VAR] = str(resolved)
    return resolved


def reports_subdir(*parts: Union[str, Path]) -> Path:
    """Convenience helper to build a path under the reports directory."""
    return get_reports_dir().joinpath(*parts)
