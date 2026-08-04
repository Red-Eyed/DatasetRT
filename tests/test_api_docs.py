from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_public_api_docs_are_generated_from_docstrings(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "sync_api_docs.py"), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
