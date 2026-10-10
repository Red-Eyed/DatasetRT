"""Measure paired serial/spawned source preparation without preloading payloads."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import psutil
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict

from dataset_rt import CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig, WriterConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dataset_rt.records import CacheSource


class SourceObservation(BaseModel):
    """Validate one source's small benchmark artifact at the JSON boundary."""

    started: float
    first_sample: float
    pid: int


@dataclass
class PreparedSource:
    """Prepare records in the source process; files are benchmark observation artifacts."""

    name: str
    samples: int
    payload_bytes: int
    rounds: int
    observations: Path

    def __iter__(self) -> Iterator[CacheInput]:
        """Generate one bounded payload at a time with CPU work on delivered bytes."""
        started = time.monotonic()
        for index in range(self.samples):
            value = index.to_bytes(8, "little")
            digest = value
            for _ in range(self.rounds):
                digest = hashlib.sha256(digest).digest()
            digest = hashlib.sha256(digest).digest()
            payload = (value + digest * ((self.payload_bytes + 31) // 32))[: self.payload_bytes]
            if index == 0:
                observation = SourceObservation(
                    started=started, first_sample=time.monotonic(), pid=os.getpid()
                )
                (self.observations / self.name).write_text(observation.model_dump_json())
            yield CacheInput(payload, {"index": index})


@dataclass
class Resources:
    """Sampled process-tree maxima include the parent and spawned services."""

    peak_rss_bytes: int = 0
    peak_threads: int = 0
    peak_processes: int = 0


def observe_resources(stop: threading.Event, resources: Resources) -> None:
    """Retain only aggregate maxima while writers run; shared RSS may be counted twice."""
    parent = psutil.Process()
    while not stop.is_set():
        rss = threads = processes = 0
        for process in [parent, *parent.children(recursive=True)]:
            try:
                rss += process.memory_info().rss
                threads += process.num_threads()
                processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        resources.peak_rss_bytes = max(resources.peak_rss_bytes, rss)
        resources.peak_threads = max(resources.peak_threads, threads)
        resources.peak_processes = max(resources.peak_processes, processes)
        stop.wait(0.05)


@dataclass(frozen=True)
class Measurement:
    """Write duration includes process startup and shutdown; verification is outside it."""

    pair: int
    workload: Literal["heavy", "cheap"]
    processes: int
    write_seconds: float
    first_source_seconds: float
    first_sample_seconds: float
    accepted_samples: int
    accepted_bytes: int
    writer_pids: tuple[int, ...]
    resources: Resources


class Benchmark(BaseSettings):
    """Explicit sizes and rotating paired controls separate useful work from startup overhead."""

    model_config = SettingsConfigDict(cli_kebab_case=True)
    samples: int = Field(default=2000, gt=0)
    sources: int = Field(default=8, gt=0)
    payload_bytes: int = Field(default=1024, ge=8)
    preparation_rounds: int = Field(default=5000, ge=0)
    native_workers: int = Field(default=1, gt=0)
    prefetch_size: int = Field(default=16, gt=0)
    repeats: int = Field(default=5, ge=1)
    processes: list[int] = Field(default=[0, 1, 2, 4], min_length=1)
    process_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    output: Path = Path("plan/evidence/writer-spawn.json")
    quiet: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Run useful-work controls and preserve source fingerprints with raw paired timings."""
        if any(count < 0 for count in self.processes) or len(set(self.processes)) != len(
            self.processes
        ):
            raise ValueError("process counts must be distinct nonnegative integers")
        measurements: list[Measurement] = []
        source_fingerprints = fingerprints()
        warmup = self.model_copy(update={"samples": 16})
        for workload in ("heavy", "cheap"):
            for processes in self.processes:
                measure(warmup, -1, workload, processes)
        for pair in range(self.repeats):
            order = (
                self.processes[pair % len(self.processes) :]
                + self.processes[: pair % len(self.processes)]
            )
            for workload in ("heavy", "cheap"):
                for processes in order:
                    if not self.quiet:
                        print(
                            f"Writer pair {pair + 1}/{self.repeats}: {workload}, processes={processes}",
                            file=sys.stderr,
                        )
                    measurements.append(measure(self, pair, workload, processes))
        if fingerprints() != source_fingerprints:
            raise RuntimeError("writer or benchmark sources changed during measurement")
        report = WriterReport(
            recorded_at=datetime.now(UTC),
            revision=subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            fingerprints=source_fingerprints,
            platform=platform.platform(),
            python=platform.python_version(),
            cpu_count=psutil.cpu_count() or 1,
            memory_bytes=psutil.virtual_memory().total,
            settings=self,
            measurements=measurements,
            paired_speedups=paired_speedups(measurements),
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(report.model_dump_json(indent=2) + "\n")
        print(json.dumps(report.paired_speedups, indent=2, sort_keys=True))


class WriterReport(BaseModel):
    """Keep workload, provenance and measurements precise when exporting JSON."""

    recorded_at: datetime
    revision: str
    fingerprints: dict[str, str]
    platform: str
    python: str
    cpu_count: int
    memory_bytes: int
    settings: Benchmark
    measurements: list[Measurement]
    paired_speedups: dict[str, dict[int, float]]
    scope: str = "T10 descriptive writer measurements; final combined acceptance is T12"


def fingerprints() -> dict[str, str]:
    """Identify the uncommitted implementation actually measured."""
    root = Path(__file__).resolve().parents[1]
    names = (
        "dataset_rt/writer.py",
        "dataset_rt/config.py",
        "dataset_rt/runtime.py",
        "scripts/bench_writer.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def measure(
    config: Benchmark, pair: int, workload: Literal["heavy", "cheap"], processes: int
) -> Measurement:
    """Verify every physical record after a timed, bounded source-write operation."""
    with tempfile.TemporaryDirectory(prefix="dataset_rt_writer_") as temporary:
        root = Path(temporary)
        observations = root / "observations"
        observations.mkdir()
        sources: list[CacheSource] = [
            PreparedSource(
                f"source-{index}",
                config.samples,
                config.payload_bytes,
                config.preparation_rounds if workload == "heavy" else 0,
                observations,
            )
            for index in range(config.sources)
        ]
        runtime = DatasetRuntime(num_workers=config.native_workers)
        resources = Resources()
        stop = threading.Event()
        observer = threading.Thread(target=observe_resources, args=(stop, resources))
        observer.start()
        started = time.monotonic()
        try:
            results = runtime.write_cache(
                sources,
                root / "cache",
                writer_config=WriterConfig(
                    num_processes=processes,
                    process_timeout_seconds=config.process_timeout_seconds,
                    prefetch_size=config.prefetch_size,
                    show_progress=False,
                ),
            )
            seconds = time.monotonic() - started
        finally:
            stop.set()
            observer.join(timeout=5)
        if observer.is_alive():
            raise RuntimeError("resource observer did not stop")
        paths: list[Path] = []
        for result in results:
            match result:
                case CacheWriteSuccess(path=path):
                    paths.append(path)
                case error:
                    raise RuntimeError(error)
        dataset = runtime.cached_dataset(
            paths, reader_config=ReaderConfig(seed=0, shuffle=False, validate_cache=True)
        )
        count = byte_count = 0
        for sample in dataset:
            assert int.from_bytes(sample.data[:8], "little") == sample.sample_id
            count += 1
            byte_count += len(sample.data)
        assert count == config.samples * config.sources
        assert byte_count == count * config.payload_bytes
        # Each file is one small benchmark record, never an internal payload bridge.
        records = [
            SourceObservation.model_validate_json(path.read_text())
            for path in observations.iterdir()
        ]
        return Measurement(
            pair,
            workload,
            processes,
            seconds,
            min(record.started for record in records) - started,
            min(record.first_sample for record in records) - started,
            count,
            byte_count,
            tuple(sorted({record.pid for record in records})),
            resources,
        )


def paired_speedups(measurements: list[Measurement]) -> dict[str, dict[int, float]]:
    """Use median ratios of paired serial/write times rather than ratios of medians."""
    controls = {
        (item.pair, item.workload): item.write_seconds
        for item in measurements
        if item.processes == 0
    }
    ratios: dict[str, dict[int, list[float]]] = {}
    for item in measurements:
        control = controls.get((item.pair, item.workload))
        if control is not None:
            ratios.setdefault(item.workload, {}).setdefault(item.processes, []).append(
                control / item.write_seconds
            )
    return {
        workload: {processes: statistics.median(values) for processes, values in cases.items()}
        for workload, cases in ratios.items()
    }


if __name__ == "__main__":
    CliApp.run(Benchmark)
