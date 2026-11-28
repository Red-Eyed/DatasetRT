"""Small CPU workloads and failure contracts for the portable benchmark runner."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

from dataset_rt import CachedSample
from dataset_rt.benchmarks.config import BenchmarkConfig, Topology
from dataset_rt.benchmarks.distributed import (
    Report,
    Source,
    Transform,
    device_for,
    file_digest,
)
from dataset_rt.run_benchmark import launch

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("shuffle", [False, True])
def test_cpu_benchmark_reports_real_training_and_seeded_replay(
    tmp_path: Path, shuffle: bool
) -> None:
    """The normal test suite exercises finite training through the public launcher."""
    output = tmp_path / "nested" / "report.json"
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        environment.pop(key, None)
    if sys.platform == "darwin":
        environment["GLOO_SOCKET_IFNAME"] = "lo0"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dataset_rt.run_benchmark",
            "--quiet",
            "--samples-per-cache",
            "5",
            "--caches",
            "2",
            "--batch-size",
            "3",
            "--input-features",
            "32",
            "--model-width",
            "4",
            "--payload-bytes",
            "16",
            "--metadata-bytes",
            "24",
            "--transform-rounds",
            "1",
            "--repeats",
            "2",
            "--timeout-seconds",
            "45",
            "--shuffle" if shuffle else "--no-shuffle",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = Report.model_validate_json(output.read_bytes())
    assert report.world_size == 1
    assert report.backend == "gloo"
    assert report.synthetic
    assert len(report.ranks) == 1
    rank = report.ranks[0]
    assert rank.population == rank.global_effective_samples == 10
    assert rank.loader_batches == 4
    assert rank.replay_verified
    assert rank.sequential_fixture_order_verified == (not shuffle)
    assert rank.parameter_max_difference == 0
    assert len(rank.reader_sha256) == len(rank.benchmark_sha256) == 64
    for result in rank.passes:
        assert result.samples == 10
        assert result.batches == 4
        assert result.resources.processes >= 1
        assert result.gpu_peak_allocated_bytes == result.gpu_peak_reserved_bytes == 0
    assert all(rate > 0 for rate in report.global_samples_per_second)


@pytest.mark.parametrize("status", [0, 7])
def test_launcher_preserves_exit_status(status: int) -> None:
    """A failed rank launch preserves its exit code and reaps only its owned session."""
    process = MagicMock()
    process.pid = 1234
    process.wait.return_value = status
    with (
        patch("dataset_rt.run_benchmark.subprocess.Popen") as popen,
        patch("dataset_rt.run_benchmark.os.killpg") as kill,
    ):
        popen.return_value.__enter__.return_value = process
        if status:
            with pytest.raises(SystemExit) as caught:
                launch(BenchmarkConfig(timeout_seconds=3))
            assert caught.value.code == status
            kill.assert_called_once_with(1234, signal.SIGKILL)
        else:
            launch(BenchmarkConfig(timeout_seconds=3))
            kill.assert_not_called()
        assert popen.call_args.kwargs["start_new_session"]
        assert process.wait.call_args_list[0].kwargs == {"timeout": 3}


def test_launcher_timeout_reaps_an_already_exited_session() -> None:
    """Process disappearance during timeout cleanup preserves the original timeout."""
    process = MagicMock()
    timeout = subprocess.TimeoutExpired("torchrun", 1)
    process.wait.side_effect = [timeout, 0]
    with (
        patch("dataset_rt.run_benchmark.subprocess.Popen") as popen,
        patch("dataset_rt.run_benchmark.os.killpg", side_effect=ProcessLookupError),
    ):
        popen.return_value.__enter__.return_value = process
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            launch(BenchmarkConfig(timeout_seconds=1))
        assert caught.value is timeout
        assert process.wait.call_args_list[-1].kwargs == {"timeout": 10}


@pytest.mark.parametrize("bad_position", [True, "0"])
def test_synthetic_transform_rejects_non_integer_identity(bad_position: bool | str) -> None:
    """Synthetic metadata must encode an integer position rather than a coercible value."""
    sample = CachedSample(
        data=bytes(16), metadata={"benchmark_position": bad_position}, cache_id=1, sample_id=0
    )
    with pytest.raises(ValueError, match="synthetic position must be an integer"):
        Transform(32, 0, True)(sample)


def test_synthetic_transform_rejects_payload_identity_mismatch() -> None:
    """A valid-looking metadata identity cannot disguise the wrong stored payload."""
    sample = CachedSample(
        data=bytes(16), metadata={"benchmark_position": 1}, cache_id=1, sample_id=0
    )
    with pytest.raises(ValueError, match="payload identity differs"):
        Transform(32, 0, True)(sample)


def test_domain_transform_does_not_require_synthetic_metadata() -> None:
    """Existing domain caches produce deterministic features without synthetic headers."""
    sample = CachedSample(data=b"domain payload", metadata={}, cache_id=9, sample_id=4)
    transformed = Transform(64, 2, False)(sample)
    assert transformed.inputs.shape == (64,)
    assert torch.equal(transformed.inputs[:32], transformed.inputs[32:])
    assert transformed.identity.cache_id == 9
    assert transformed.identity.sample_id == 4


def test_file_digest_handles_multiple_chunks(tmp_path: Path) -> None:
    """Artifact fingerprints include every chunk and the partial final chunk."""
    data = b"bounded-digest" * 100_000
    path = tmp_path / "artifact"
    path.write_bytes(data)
    assert file_digest(path) == hashlib.sha256(data).hexdigest()


def test_synthetic_source_encodes_cache_and_row_positions() -> None:
    """Fixture payloads independently encode global and local row identities."""
    rows = list(Source("test", 2, 3, 19, 0))
    assert len(rows) == 3
    for index, row in enumerate(rows):
        assert len(row.data) == 19
        assert row.data[:8] == (6 + index).to_bytes(8, "little")
        assert row.data[8:16] == index.to_bytes(8, "little")
        assert row.metadata == {"benchmark_position": 6 + index, "wide": ""}


def test_cuda_topology_is_rejected_without_available_devices() -> None:
    """Explicit CUDA requests fail before any distributed process group is initialized."""
    topology = Topology(rank=0, local_rank=0, world_size=1, local_world_size=1)
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ValueError, match="exceeds available CUDA devices"):
            device_for(topology, BenchmarkConfig(device="cuda"))
