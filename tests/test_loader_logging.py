"""Standard logger integration keeps fallback output separate from application configuration."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING

from pydantic import BaseModel

from dataset_rt.integrations.loader_logging import LOGGER_NAME, loader_event

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class ContinuationReport(BaseModel):
    """Validate bounded subprocess results before checking sampled stream continuation."""

    first: list[int]
    second: list[int]
    expected: list[int]


def test_configured_logging_receives_one_event(caplog: pytest.LogCaptureFixture) -> None:
    """Application logging captures one record without an added library handler."""
    logger = logging.getLogger(LOGGER_NAME)
    before = tuple(logger.handlers)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        loader_event("event=probe seed=%s", 17)
    assert [record.getMessage() for record in caplog.records if record.name == LOGGER_NAME] == [
        "event=probe seed=17"
    ]
    assert tuple(logger.handlers) == before


def test_unconfigured_process_uses_stderr_fallback() -> None:
    """Fresh spawned applications can see INFO without a global basicConfig call."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dataset_rt.integrations.loader_logging import loader_event; loader_event('event=probe seed=%s', 17)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
    assert result.stderr.count("event=probe seed=17") == 1


def test_persistent_spawn_logs_stream_continuation(tmp_path: Path) -> None:
    """Two real passes retain one worker and exactly continue its native sampling stream."""
    result = subprocess.run(
        [sys.executable, "tests/fixtures/loader_logging_process.py"],
        env={**os.environ, "DATASETRT_LOG_TEST_CACHE_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    report = ContinuationReport.model_validate_json(result.stdout)
    assert len(report.first) == len(report.second) == 4
    assert report.first + report.second == report.expected
    starts = re.findall(r"event=reader_started pid=(\d+) rank=0 worker=0 seed=(\d+)", result.stderr)
    assert len(starts) == 1
    assert result.stderr.count("sampling_reset=True") == 1
    assert "persistent=True" in result.stderr
