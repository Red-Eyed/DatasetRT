"""Manual benchmark entry point acceptance with real torchrun and spawned consumers."""

from __future__ import annotations

import fcntl
import hashlib
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist

from dataset_rt import CacheInput, CacheWriteSuccess, DatasetRuntime
from dataset_rt.benchmarks.distributed import Report
from dataset_rt.config import WriterConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class Case:
    """Keep explicit rank/workload choices separate from expected observed counts."""

    args: tuple[str, ...]
    ranks: int
    samples: int
    batches: int
    sequential: bool = False


@pytest.fixture(autouse=True)
def execution_slot() -> Iterator[None]:
    """Share the existing DDP acceptance slot to bound actual concurrent RAM usage."""
    with (Path(gettempdir()) / "dataset-rt-ddp-tests.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@pytest.fixture
def environment() -> dict[str, str]:
    """Select Mac loopback only in local tests; the portable runner owns no network policy."""
    values = os.environ.copy()
    if sys.platform == "darwin":
        values["GLOO_SOCKET_IFNAME"] = "lo0"
    return values


def invoke(
    args: tuple[str, ...], output: Path, environment: dict[str, str], *, external: bool = False
) -> Report:
    """Run the real module with a separate finite deadline and owned process session."""
    command = [
        sys.executable,
        "-m",
        "dataset_rt.run_benchmark",
        "--quiet",
        "--samples-per-cache",
        "13",
        "--caches",
        "1",
        "--batch-size",
        "3",
        "--input-features",
        "32",
        "--model-width",
        "16",
        "--repeats",
        "2",
        "--timeout-seconds",
        "90",
        "--output",
        str(output),
        *args,
    ]
    if external:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr",
            "127.0.0.1",
            "--nproc-per-node",
            "2",
            "--max-restarts",
            "0",
            "--module",
            "dataset_rt.run_benchmark",
            *command[3:],
        ]
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=110)
            assert process.returncode == 0, stderr
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    assert stdout.strip()
    return Report.model_validate_json(output.read_bytes())


@pytest.mark.parametrize(
    "case",
    [
        Case(("--no-shuffle",), 1, 13, 5, True),
        Case(("--ranks", "2", "--num-workers", "2", "--no-shuffle"), 2, 7, 3, True),
        Case(("--ranks", "2", "--num-workers", "1", "--worker-partition", "replicate"), 2, 7, 3),
        Case(("--ranks", "2", "--num-workers", "1", "--samples-per-epoch", "60"), 2, 30, 10),
        Case(("--ranks", "2", "--num-workers", "2", "--no-shuffle", "--drop-last"), 2, 6, 2, True),
        Case(
            ("--ranks", "2", "--num-workers", "2", "--drop-last", "--samples-per-epoch", "1"),
            2,
            0,
            0,
        ),
        Case(
            (
                "--ranks",
                "2",
                "--num-workers",
                "1",
                "--worker-context",
                "forkserver",
                "--no-persistent-workers",
            ),
            2,
            7,
            3,
        ),
    ],
)
def test_module_runs_real_rank_workloads(
    case: Case, tmp_path: Path, environment: dict[str, str]
) -> None:
    """Counts, replay, model agreement, and fixture ordering hold through the public CLI."""
    report = invoke(case.args, tmp_path / "report.json", environment)
    assert report.world_size == len(report.ranks) == case.ranks
    assert len(report.global_samples_per_second) == 2
    for rank in report.ranks:
        assert rank.replay_verified
        assert rank.parameter_max_difference == 0
        assert rank.sequential_fixture_order_verified == case.sequential
        assert rank.loader_batches == case.batches
        assert all(
            item.samples == case.samples and item.batches == case.batches for item in rank.passes
        )
    assert all(rate >= 0 for rate in report.global_samples_per_second)


class ExistingSource:
    """Use domain payloads without the benchmark's synthetic header or metadata schema."""

    name = "existing-benchmark-cache"

    def __iter__(self) -> Iterator[CacheInput]:
        """Keep generation streaming while exercising the existing-cache path."""
        for index in range(13):
            yield CacheInput(f"domain payload {index}".encode(), {"index": index})


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    """Publish one immutable native cache before the external module is launched."""
    result = DatasetRuntime(num_workers=1).write_cache(
        ExistingSource(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    return result.path


def test_existing_caches_remain_immutable(
    cache_path: Path, tmp_path: Path, environment: dict[str, str]
) -> None:
    """Manual measurements read actual stored bytes without rewriting cache artifacts."""
    before = tuple(
        (path.relative_to(cache_path), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(cache_path.rglob("*"))
        if path.is_file()
    )
    report = invoke(
        ("--ranks", "2", "--num-workers", "1", "--cache-paths", f'["{cache_path}"]'),
        tmp_path / "existing.json",
        environment,
    )
    after = tuple(
        (path.relative_to(cache_path), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(cache_path.rglob("*"))
        if path.is_file()
    )
    assert not report.synthetic
    assert all(rank.population == 13 and rank.replay_verified for rank in report.ranks)
    assert before == after


def test_external_torchrun_uses_launcher_identity(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    """External launches use WORLD_SIZE despite the CLI's default one-rank local launcher."""
    report = invoke(
        ("--num-workers", "1", "--no-shuffle"),
        tmp_path / "external.json",
        environment,
        external=True,
    )
    assert report.settings.ranks == 1
    assert report.world_size == len(report.ranks) == 2
    assert report.backend == "gloo"
    assert {rank.rank for rank in report.ranks} == {0, 1}


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("workers", [0, 2])
def test_cuda_module(ranks: int, workers: int, tmp_path: Path, environment: dict[str, str]) -> None:
    """Run actual CUDA/NCCL and pinned transfers when the requested GPUs are present."""
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        pytest.skip("requires CUDA-enabled PyTorch and NCCL")
    if torch.cuda.device_count() < ranks:
        pytest.skip(f"requires {ranks} visible CUDA devices")
    report = invoke(
        ("--device", "cuda", "--ranks", str(ranks), "--num-workers", str(workers), "--pin-memory"),
        tmp_path / f"cuda-{ranks}-{workers}.json",
        environment,
    )
    assert report.backend == "nccl"
    assert report.world_size == ranks
    for rank in report.ranks:
        assert rank.replay_verified and rank.parameter_max_difference <= 1e-6
        assert all(item.gpu_peak_allocated_bytes > 0 for item in rank.passes)
