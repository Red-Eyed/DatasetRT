"""Reject invalid benchmark workloads before launching expensive process topologies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from dataset_rt.benchmarks.config import BenchmarkConfig
from dataset_rt.benchmarks.distributed import local_samples

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("ranks,workers", [(0, 1), (1, -1)])
def test_invalid_process_counts(ranks: int, workers: int) -> None:
    """Process counts must represent a nonempty training group and nonnegative workers."""
    with pytest.raises(ValidationError):
        BenchmarkConfig(ranks=ranks, num_workers=workers)


def test_cuda_rejects_fork_workers() -> None:
    """Avoid poison-fork failures before a CUDA process or worker is created."""
    with pytest.raises(ValidationError, match="spawn or forkserver"):
        BenchmarkConfig(device="cuda", num_workers=1, worker_context="fork")


@pytest.mark.parametrize("features", [0, 31, 33])
def test_invalid_feature_shapes(features: int) -> None:
    """Digest feature repetition must produce exactly the declared model input width."""
    with pytest.raises(ValidationError):
        BenchmarkConfig(input_features=features)


def test_missing_cache_directory(tmp_path: Path) -> None:
    """Existing-cache mode validates its filesystem boundary before rank launch."""
    with pytest.raises(ValidationError):
        BenchmarkConfig(cache_paths=(tmp_path / "missing",))


@pytest.mark.parametrize("drop_last,expected", [(False, 334), (True, 320)])
def test_expected_global_quotas(drop_last: bool, expected: int) -> None:
    """Independently verify the documented global 1000-sample, three-rank example."""
    assert local_samples(1000, 3, 32, drop_last) == expected


def test_empty_dropped_epoch() -> None:
    """A tiny global budget is a valid zero-work run, with no invented samples."""
    assert local_samples(3, 4, 32, True) == 0
