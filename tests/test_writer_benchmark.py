"""Verify bounded writer benchmark controls using real serial and spawned cases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.bench_writer import WriterReport


def test_writer_benchmark_cli_smoke(tmp_path: Path) -> None:
    """Cheap and heavy cases verify work counts, source PIDs, and typed JSON output."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "writer.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "bench_writer.py"),
            "--samples",
            "16",
            "--sources",
            "4",
            "--preparation-rounds",
            "10",
            "--repeats",
            "1",
            "--quiet",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    report = WriterReport.model_validate_json(output.read_text())
    assert len(report.measurements) == 8
    assert set(report.paired_speedups) == {"heavy", "cheap"}
    for measurement in report.measurements:
        assert measurement.accepted_samples == 64
        assert measurement.accepted_bytes == 64 * 1024
        assert (
            measurement.write_seconds
            > measurement.first_sample_seconds
            >= measurement.first_source_seconds
            >= 0
        )
        assert len(measurement.writer_pids) <= max(1, measurement.processes)
        assert measurement.resources.peak_processes >= 1
        assert measurement.resources.peak_rss_bytes > 0


def test_writer_benchmark_rejects_duplicate_process_counts(tmp_path: Path) -> None:
    """An ambiguous control matrix fails before any workload starts."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "bench_writer.py"),
            "--processes",
            "[0,0]",
            "--output",
            str(tmp_path / "bad.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "distinct nonnegative" in result.stderr
    assert not (tmp_path / "bad.json").exists()
