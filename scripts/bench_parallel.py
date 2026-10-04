"""Record reproducible serial baselines before adding Python multiprocessing."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict

from dataset_rt import (
    CacheInput,
    CacheSource,
    CacheWriteSuccess,
    DatasetRuntime,
    ReaderConfig,
    WriterConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class Source:
    """Stream bounded payloads while spending explicit CPU work in Python."""

    name: str
    samples: int
    payload_bytes: int
    rounds: int

    def __iter__(self) -> Iterator[CacheInput]:
        """Keep preparation in the source process, as production serializers do."""
        for index in range(self.samples):
            digest = prepare(index, self.rounds)
            payload = (digest * ((self.payload_bytes + 31) // 32))[: self.payload_bytes]
            yield CacheInput(payload, {"index": index})


def prepare(index: int, rounds: int) -> bytes:
    """Use bounded CPU-heavy Python work without numerical-library thread pools."""
    value = str(index).encode()
    for _ in range(rounds):
        value = hashlib.sha256(value).digest()
    return hashlib.sha256(value).digest()


@dataclass(frozen=True)
class Measurement:
    """One finite run reports accepted work and lifecycle costs separately."""

    write_seconds: float
    startup_seconds: float
    first_sample_seconds: float
    read_transform_seconds: float
    cleanup_seconds: float
    accepted: int
    bytes_read: int
    rss_bytes: int
    threads: int
    file_descriptors: int


class Benchmark(BaseSettings):
    """Explicit workload sizes permit smoke and reference runs from one command."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    samples: int = Field(default=1000, gt=0)
    sources: int = Field(default=4, gt=0)
    payload_bytes: int = Field(default=1024, gt=0)
    preparation_rounds: int = Field(default=1000, ge=0)
    transform_rounds: int = Field(default=1000, ge=0)
    native_workers: int = Field(default=1, gt=0)
    prefetch_size: int = Field(default=16, gt=0)
    repeats: int = Field(default=5, ge=1)
    output: Path = Path("plan/evidence/baseline.json")
    quiet: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Generate fresh fixtures and preserve JSON evidence with environment identity."""
        measurements = []
        for index in range(self.repeats):
            if not self.quiet:
                print(f"Baseline run {index + 1}/{self.repeats}", file=sys.stderr)
            measurements.append(asdict(measure(self)))
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        report = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "revision": revision,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
            "settings": self.model_dump(mode="json"),
            "filesystem_cache": "fresh write then read; warm filesystem cache",
            "measurements": measurements,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        self.output.write_text(rendered + "\n")
        print(rendered)


def measure(config: Benchmark) -> Measurement:
    """Separate fixture writing, native startup, streaming transforms and cleanup."""
    process = psutil.Process()
    temporary = tempfile.TemporaryDirectory(prefix="dataset_rt_baseline_")
    try:
        runtime = DatasetRuntime(num_workers=config.native_workers)
        started = time.perf_counter()
        sources: list[CacheSource] = [
            Source(
                f"source-{index}", config.samples, config.payload_bytes, config.preparation_rounds
            )
            for index in range(config.sources)
        ]
        results = runtime.write_cache(
            sources,
            Path(temporary.name),
            writer_config=WriterConfig(prefetch_size=config.prefetch_size, show_progress=False),
        )
        write_seconds = time.perf_counter() - started
        paths = []
        for result in results:
            match result:
                case CacheWriteSuccess(path=path):
                    paths.append(path)
                case error:
                    raise RuntimeError(error)
        started = time.perf_counter()
        dataset = runtime.cached_dataset(
            paths,
            reader_config=ReaderConfig(seed=42, prefetch_size=config.prefetch_size),
        )
        startup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        first_sample_seconds = 0.0
        accepted = 0
        bytes_read = 0
        for sample in dataset:
            prepare(sample.sample_id, config.transform_rounds)
            accepted += 1
            bytes_read += len(sample.data)
            if accepted == 1:
                first_sample_seconds = time.perf_counter() - started
        read_seconds = time.perf_counter() - started
        assert accepted == config.samples * config.sources
        assert bytes_read == accepted * config.payload_bytes
        rss_bytes = process.memory_info().rss
        threads = process.num_threads()
        descriptors = process.num_fds()
        started = time.perf_counter()
        del dataset, runtime
        temporary.cleanup()
        cleanup_seconds = time.perf_counter() - started
        return Measurement(
            write_seconds,
            startup_seconds,
            first_sample_seconds,
            read_seconds,
            cleanup_seconds,
            accepted,
            bytes_read,
            rss_bytes,
            threads,
            descriptors,
        )
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    CliApp.run(Benchmark)
