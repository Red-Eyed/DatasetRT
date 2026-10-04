"""Paired reader benchmarks; resource observers never transport dataset samples."""

from __future__ import annotations

import gc
import hashlib
import multiprocessing as mp
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import psutil
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict

from dataset_rt import (
    CachedSample,
    CacheInput,
    CacheSource,
    CacheWriteSuccess,
    DatasetRuntime,
    ReaderConfig,
    WriterConfig,
)
from dataset_rt.integrations.loading import derive_seed

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from dataset_rt import CachedDataset

Context = Literal["serial", "fork", "spawn", "forkserver"]
Workload = Literal["heavy", "cheap"]
BoundaryT = TypeVar("BoundaryT")


class Record(BaseModel):
    """Validate JSON once at benchmark process boundaries, rejecting unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class ReaderCase(Record):
    """Separate the native control from caller-configured PyTorch consumption."""

    mode: Literal["native", "loader"]
    workers: int = Field(ge=0)
    context: Context

    @model_validator(mode="after")
    def _validate_context(self) -> ReaderCase:
        """Reject mode/worker combinations that cannot describe a real consumer."""
        if self.mode == "native" and self.workers != 0:
            raise ValueError("native control cannot have loading workers")
        if (self.workers == 0) != (self.context == "serial"):
            raise ValueError("serial context requires zero workers, other contexts require workers")
        return self


@dataclass(frozen=True)
class Output:
    """Transport compact transformed output rather than cached payloads."""

    bytes_read: int
    pid: int
    digest: bytes


@dataclass(frozen=True)
class Transform:
    """Spend explicit CPU work on delivered bytes, with identical work in every mode."""

    rounds: int

    def __call__(self, sample: CachedSample) -> Output:
        """Check physical identity before hashing; never operate on preloaded samples."""
        assert int.from_bytes(sample.data[:8], "little") == sample.cache_id
        assert int.from_bytes(sample.data[8:16], "little") == sample.sample_id
        digest = hashlib.sha256(sample.data).digest()
        for _ in range(self.rounds):
            digest = hashlib.sha256(digest).digest()
        return Output(len(sample.data), os.getpid(), digest)


@dataclass
class Source:
    """Stream deterministic fixed-size payloads without retaining a full fixture."""

    name: str
    cache_id: int
    samples: int
    payload_bytes: int

    def __iter__(self) -> Iterator[CacheInput]:
        """Encode physical identity independently of the cache writer's metadata."""
        for index in range(self.samples):
            header = self.cache_id.to_bytes(8, "little") + index.to_bytes(8, "little")
            payload = (header * ((self.payload_bytes + 15) // 16))[: self.payload_bytes]
            yield CacheInput(payload, {"index": index})


class Resources(Record):
    """Sampled process-tree maxima; summed RSS may count shared pages repeatedly."""

    rss_bytes: int = 0
    threads: int = 0
    file_descriptors: int = 0
    processes: int = 0


class PassMeasurement(Record):
    """Keep first-batch latency separate from useful steady-state throughput."""

    iterator_seconds: float = Field(ge=0)
    first_batch_seconds: float = Field(ge=0)
    steady_seconds: float = Field(gt=0)
    steady_samples: int = Field(gt=0)
    accepted: int = Field(gt=0)
    bytes_read: int = Field(gt=0)
    transform_pids: tuple[int, ...]
    resources: Resources

    @property
    def samples_per_second(self) -> float:
        """Normalize tail sizes using actually consumed post-first-batch samples."""
        return self.steady_samples / self.steady_seconds


class TrialResult(Record):
    """Both passes share one source dataset and, when requested, persistent workers."""

    case: ReaderCase
    shuffle: bool
    workload: Workload
    pair: int
    construction_seconds: float
    cold: PassMeasurement
    warm: PassMeasurement
    cleanup_seconds: float


class Gate(Record):
    """Report paired ratios without converting noisy timing targets into pytest assertions."""

    shuffle: bool
    workload: Workload
    metric: Literal["four_worker_speedup", "serial_regression"]
    context: Context
    paired_ratios: tuple[float, ...]
    median: float
    threshold: float
    sufficient_repeats: bool
    passed: bool


class Environment(Record):
    """Identify the code and machine that produced the local, warm-filesystem evidence."""

    recorded_at: datetime
    revision: str
    worktree_dirty: bool
    benchmark_sha256: str
    reader_sha256: str
    platform: str
    python: str
    torch: str
    psutil: str
    cpus: int
    memory_bytes: int


class Report(Record):
    """Retain bounded per-trial evidence and explicit acceptance outcomes."""

    environment: Environment
    settings: Benchmark
    trials: tuple[TrialResult, ...]
    gates: tuple[Gate, ...]


class Benchmark(BaseSettings):
    """Generate fixtures automatically and compare caller-selected worker contexts."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    samples_per_cache: int = Field(default=1000, gt=0)
    caches: int = Field(default=4, gt=0)
    payload_bytes: int = Field(default=1024, ge=16)
    transform_rounds: int = Field(default=1000, ge=0)
    seed: int = Field(default=7, ge=0, le=(1 << 64) - 1)
    batch_size: int = Field(default=32, gt=0)
    native_workers: int = Field(default=1, gt=0)
    prefetch_size: int = Field(default=16, gt=0)
    prefetch_factor: int = Field(default=2, gt=0)
    contexts: tuple[Literal["fork", "spawn", "forkserver"], ...] = ("fork",)
    workloads: tuple[Workload, ...] = ("heavy", "cheap")
    shuffles: tuple[bool, ...] = (True, False)
    repeats: int = Field(default=5, gt=0)
    trial_timeout: float = Field(default=120, gt=0)
    persistent_workers: CliImplicitFlag[bool] = True
    output: Path = Path("plan/evidence/reader.json")
    quiet: CliImplicitFlag[bool] = False
    internal_trial: CliImplicitFlag[bool] = False

    @model_validator(mode="after")
    def _validate_work(self) -> Benchmark:
        """Prevent partial shuffled batches from disguising discarded work as speedup."""
        total = self.samples_per_cache * self.caches
        if total <= self.batch_size or total % self.batch_size:
            raise ValueError("total fixture samples must exceed and be divisible by batch_size")
        if not self.contexts or not self.workloads or not self.shuffles:
            raise ValueError("contexts, workloads, and shuffles must be nonempty")
        if len(set(self.contexts)) != len(self.contexts):
            raise ValueError("contexts must be unique")
        if len(set(self.workloads)) != len(self.workloads) or len(set(self.shuffles)) != len(
            self.shuffles
        ):
            raise ValueError("workloads and shuffles must be unique")
        for context in self.contexts:
            if context not in mp.get_all_start_methods():
                raise ValueError(f"unavailable worker context: {context}")
        return self

    def cli_cmd(self) -> None:
        """Keep subprocess JSON protocol separate from human progress on stderr."""
        if self.internal_trial:
            specification = TrialSpec.model_validate_json(sys.stdin.read())
            print(measure_trial(specification).model_dump_json())
            return
        report = run_benchmark(self)
        rendered = report.model_dump_json(indent=2)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(rendered + "\n")
        print(rendered)


class TrialSpec(Record):
    """Only paths and immutable configuration enter the isolated trial process."""

    config: Benchmark
    paths: tuple[Path, ...]
    case: ReaderCase
    shuffle: bool
    workload: Workload
    pair: int


@dataclass
class ResourceObserver:
    """Sample resources independently of batch transport with O(processes) temporary state."""

    stop: threading.Event = field(default_factory=threading.Event)
    peak: Resources = field(default_factory=Resources)
    failures: list[Exception] = field(default_factory=list)

    def run(self) -> None:
        """Include the consuming process, loading workers, and context service processes."""
        process = psutil.Process()
        try:
            while not self.stop.is_set():
                current = resource_snapshot(process)
                self.peak = Resources(
                    rss_bytes=max(self.peak.rss_bytes, current.rss_bytes),
                    threads=max(self.peak.threads, current.threads),
                    file_descriptors=max(self.peak.file_descriptors, current.file_descriptors),
                    processes=max(self.peak.processes, current.processes),
                )
                self.stop.wait(0.05)
        except Exception as error:
            # Thread exceptions must invalidate the trial, not silently produce
            # incomplete resource evidence. One failure ends observation.
            self.failures.append(error)


def resource_snapshot(process: psutil.Process) -> Resources:
    """Skip children that exited between enumeration and observation."""
    rss = threads = descriptors = count = 0
    for member in (process, *process.children(recursive=True)):
        try:
            with member.oneshot():
                rss += member.memory_info().rss
                threads += member.num_threads()
                descriptors += member.num_fds()
                count += 1
        except psutil.NoSuchProcess:
            continue
    return Resources(rss_bytes=rss, threads=threads, file_descriptors=descriptors, processes=count)


def collate(outputs: list[Output]) -> tuple[Output, ...]:
    """Use the same compact batch representation as the direct-native control."""
    return tuple(outputs)


def native_batches(
    dataset: CachedDataset, transform: Transform, size: int
) -> Iterator[tuple[Output, ...]]:
    """Batch transformed outputs only, with O(batch size) Python storage."""
    batch: list[Output] = []
    for sample in dataset:
        batch.append(transform(sample))
        if len(batch) == size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def checked_batch(value: BoundaryT) -> tuple[Output, ...]:
    """Narrow Torch's untyped collator return to a validated bounded output batch."""
    if not isinstance(value, tuple) or not value:
        raise TypeError("expected a nonempty transformed batch")
    for item in value:
        if not isinstance(item, Output) or len(item.digest) != 32:
            raise TypeError("invalid transformed output")
    return cast("tuple[Output, ...]", value)


def measure_pass(
    batches: Iterable[tuple[Output, ...]], config: Benchmark, shuffle: bool
) -> PassMeasurement:
    """Consume exact useful work; discarded prefetched outputs are not counted."""
    observer = ResourceObserver()
    thread = threading.Thread(target=observer.run, name="reader-benchmark-resources")
    thread.start()
    try:
        started = time.perf_counter()
        iterator = iter(batches)
        created = time.perf_counter()
        first = checked_batch(next(iterator))
        delivered = time.perf_counter()
        accepted = len(first)
        byte_count = sum(output.bytes_read for output in first)
        pids = {output.pid for output in first}
        steady_started = time.perf_counter()
        total = config.samples_per_cache * config.caches
        while accepted < total:
            batch = checked_batch(next(iterator))
            accepted += len(batch)
            byte_count += sum(output.bytes_read for output in batch)
            pids.update(output.pid for output in batch)
        finished = time.perf_counter()
        assert accepted == total and byte_count == total * config.payload_bytes
        # Audit after the timed work, before iterator cleanup can reap workers.
        # Short smoke passes may finish between periodic observer samples.
        final_resources = resource_snapshot(psutil.Process())
        if not shuffle:
            assert next(iterator, ()) == (), "sequential loader emitted extra samples"
        del iterator
        gc.collect()
    finally:
        observer.stop.set()
        thread.join(5)
        assert not thread.is_alive(), "resource observer did not stop"
        if observer.failures:
            raise RuntimeError("resource observer failed") from observer.failures[0]
    return PassMeasurement(
        iterator_seconds=created - started,
        first_batch_seconds=delivered - started,
        steady_seconds=finished - steady_started,
        steady_samples=accepted - len(first),
        accepted=accepted,
        bytes_read=byte_count,
        transform_pids=tuple(sorted(pids)),
        resources=Resources(
            rss_bytes=max(observer.peak.rss_bytes, final_resources.rss_bytes),
            threads=max(observer.peak.threads, final_resources.threads),
            file_descriptors=max(observer.peak.file_descriptors, final_resources.file_descriptors),
            processes=max(observer.peak.processes, final_resources.processes),
        ),
    )


def measure_trial(spec: TrialSpec) -> TrialResult:
    """Warm parent native state, then exercise one caller-selected loader configuration."""
    config = spec.config
    torch.set_num_threads(1)
    started = time.perf_counter()
    runtime = DatasetRuntime(num_workers=config.native_workers)
    dataset = runtime.cached_dataset(
        spec.paths,
        reader_config=ReaderConfig(
            shuffle=spec.shuffle,
            seed=derive_seed(config.seed, 0, 0),
            prefetch_size=config.prefetch_size,
        ),
    )
    # The consumer always reconstructs its own state. Warming this source dataset
    # exercises the production case where fork follows existing native activity.
    assert next(iter(dataset)).data
    dataset.set_epoch_len(config.samples_per_cache * config.caches)
    transform = Transform(config.transform_rounds if spec.workload == "heavy" else 0)
    loader: Iterable[tuple[Output, ...]]
    if spec.case.mode == "loader":
        loader = cast(
            "Iterable[tuple[Output, ...]]",
            dataset.to_torch_dataloader(
                shuffle=spec.shuffle,
                seed=config.seed,
                batch_size=config.batch_size,
                num_workers=spec.case.workers,
                sample_transform_fn=transform,
                collate_fn=collate,
                native_num_workers=config.native_workers,
                multiprocessing_context=spec.case.context if spec.case.workers else None,
                prefetch_factor=config.prefetch_factor if spec.case.workers else None,
                persistent_workers=config.persistent_workers and spec.case.workers > 0,
            ),
        )
    else:
        loader = ()
    construction = time.perf_counter() - started
    cold = measure_pass(
        native_batches(dataset, transform, config.batch_size)
        if spec.case.mode == "native"
        else loader,
        config,
        spec.shuffle,
    )
    warm = measure_pass(
        native_batches(dataset, transform, config.batch_size)
        if spec.case.mode == "native"
        else loader,
        config,
        spec.shuffle,
    )
    if spec.case.workers == 0:
        assert cold.transform_pids == warm.transform_pids == (psutil.Process().pid,)
    else:
        assert len(cold.transform_pids) == len(warm.transform_pids) == spec.case.workers
        assert psutil.Process().pid not in cold.transform_pids + warm.transform_pids
        if config.persistent_workers:
            assert cold.transform_pids == warm.transform_pids
    started = time.perf_counter()
    del loader, dataset, runtime
    gc.collect()
    for child in mp.active_children():
        child.join(5)
        assert not child.is_alive(), "loading worker survived trial cleanup"
    cleanup = time.perf_counter() - started
    return TrialResult(
        case=spec.case,
        shuffle=spec.shuffle,
        workload=spec.workload,
        pair=spec.pair,
        construction_seconds=construction,
        cold=cold,
        warm=warm,
        cleanup_seconds=cleanup,
    )


def create_fixture(config: Benchmark, root: Path) -> tuple[Path, ...]:
    """Prepare caches once outside reader timings; sources stream through Rust bounds."""
    runtime = DatasetRuntime(num_workers=config.native_workers)
    sources: list[CacheSource] = [
        Source(f"cache-{index}", index, config.samples_per_cache, config.payload_bytes)
        for index in range(config.caches)
    ]
    paths = []
    for outcome in runtime.write_cache(
        sources, root, writer_config=WriterConfig(show_progress=False)
    ):
        match outcome:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case error:
                raise RuntimeError(error)
    return tuple(paths)


def reader_cases(
    contexts: tuple[Literal["fork", "spawn", "forkserver"], ...],
) -> tuple[ReaderCase, ...]:
    """Include native and zero-worker controls in every paired benchmark round."""
    return (
        ReaderCase(mode="native", workers=0, context="serial"),
        ReaderCase(mode="loader", workers=0, context="serial"),
        *(
            ReaderCase(mode="loader", workers=workers, context=context)
            for context in contexts
            for workers in (1, 2, 4)
        ),
    )


def run_trial(spec: TrialSpec) -> TrialResult:
    """Isolate resources/imports and enforce a finite deadline without native transport."""
    with subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--internal-trial"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(
                spec.model_dump_json(), timeout=spec.config.trial_timeout
            )
        except BaseException:
            # The trial owns a session so timeout and Ctrl+C also stop its workers.
            terminate_trial(process)
            raise
        if process.returncode != 0:
            terminate_trial(process)
            raise RuntimeError(f"benchmark trial failed ({process.returncode}): {stderr}")
    return TrialResult.model_validate_json(stdout)


def terminate_trial(process: subprocess.Popen[str]) -> None:
    """Terminate only the benchmark's owned session and reap its immediate child."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def performance_gates(trials: tuple[TrialResult, ...]) -> tuple[Gate, ...]:
    """Pair trials by workload/shuffle/round and preserve every raw timing ratio."""
    gates = []
    for workload, shuffle in sorted({(trial.workload, trial.shuffle) for trial in trials}):
        selected = [
            trial for trial in trials if trial.workload == workload and trial.shuffle == shuffle
        ]
        native = {trial.pair: trial for trial in selected if trial.case.mode == "native"}
        serial = {
            trial.pair: trial
            for trial in selected
            if trial.case.mode == "loader" and trial.case.workers == 0
        }
        ratios = tuple(
            native[pair].warm.samples_per_second / serial[pair].warm.samples_per_second - 1
            for pair in sorted(native)
        )
        median = statistics.median(ratios)
        gates.append(
            Gate(
                shuffle=shuffle,
                workload=workload,
                metric="serial_regression",
                context="serial",
                paired_ratios=ratios,
                median=median,
                threshold=0.10,
                sufficient_repeats=len(ratios) >= 5,
                passed=len(ratios) >= 5 and median <= 0.10,
            )
        )
        if workload != "heavy":
            continue
        for context in sorted(
            {trial.case.context for trial in selected if trial.case.workers == 4}
        ):
            parallel = {
                trial.pair: trial
                for trial in selected
                if trial.case.workers == 4 and trial.case.context == context
            }
            ratios = tuple(
                parallel[pair].warm.samples_per_second / serial[pair].warm.samples_per_second
                for pair in sorted(serial)
            )
            median = statistics.median(ratios)
            gates.append(
                Gate(
                    shuffle=shuffle,
                    workload=workload,
                    metric="four_worker_speedup",
                    context=context,
                    paired_ratios=ratios,
                    median=median,
                    threshold=2.0,
                    sufficient_repeats=len(ratios) >= 5,
                    passed=len(ratios) >= 5 and median >= 2.0,
                )
            )
    return tuple(gates)


def environment() -> Environment:
    """Report uncommitted benchmark identity separately from the Git base revision."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return Environment(
        recorded_at=datetime.now(UTC),
        revision=revision.stdout.strip(),
        worktree_dirty=bool(dirty.stdout),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reader_sha256=reader_fingerprint(),
        platform=platform.platform(),
        python=platform.python_version(),
        torch=version("torch"),
        psutil=version("psutil"),
        cpus=psutil.cpu_count() or 1,
        memory_bytes=psutil.virtual_memory().total,
    )


def reader_fingerprint() -> str:
    """Identify uncommitted integration code without relying on only a base revision."""
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for relative in (
        "dataset_rt/dataset.py",
        "dataset_rt/integrations/loader.py",
        "dataset_rt/integrations/loading.py",
        "dataset_rt/config.py",
        "dataset_rt/runtime.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def run_benchmark(config: Benchmark) -> Report:
    """Rotate case order per pair; retain measurements only, never a payload population."""
    trials = []
    reader_revision = reader_fingerprint()
    benchmark_revision = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases = reader_cases(config.contexts)
    with tempfile.TemporaryDirectory(prefix="dataset-rt-reader-") as directory:
        paths = create_fixture(config, Path(directory))
        for workload in config.workloads:
            for shuffle in config.shuffles:
                for pair in range(config.repeats):
                    offset = pair % len(cases)
                    for case in cases[offset:] + cases[:offset]:
                        if not config.quiet:
                            print(
                                f"{workload}, shuffle={shuffle}, pair {pair + 1}/{config.repeats}: {case}",
                                file=sys.stderr,
                            )
                        trials.append(
                            run_trial(
                                TrialSpec(
                                    config=config,
                                    paths=paths,
                                    case=case,
                                    shuffle=shuffle,
                                    workload=workload,
                                    pair=pair,
                                )
                            )
                        )
    results = tuple(trials)
    if reader_fingerprint() != reader_revision:
        raise RuntimeError(
            "reader code changed during benchmark; repeat with one consistent revision"
        )
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != benchmark_revision:
        raise RuntimeError("benchmark code changed during measurement; repeat with one revision")
    return Report(
        environment=environment(), settings=config, trials=results, gates=performance_gates(results)
    )


if __name__ == "__main__":
    CliApp.run(Benchmark)
