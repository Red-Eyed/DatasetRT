"""Paired storage benchmarks: run this same harness with isolated baseline/candidate interpreters."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import psutil
from pydantic import BaseModel, Field, TypeAdapter, computed_field, model_validator
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict

from dataset_rt import (
    CachedDataset,
    CacheInput,
    CacheWriteSuccess,
    DatasetRuntime,
    Lz4Fast,
    NoCompression,
    ReaderConfig,
    WriterConfig,
    Zstd,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dataset_rt.records import Metadata

Profile = Literal["text-64", "text-256", "text-4k", "image-64k", "image-1m"]
Codec = Literal["none", "lz4", "zstd"]
DEPENDENCIES = ("numpy", "polars", "pyarrow", "pydantic", "pydantic-settings", "psutil")


class TrialSpec(BaseModel):
    """One process owns a bounded fixture and all measurements for one matrix cell."""

    profile: Profile
    codec: Codec
    workers: int = Field(ge=1)
    samples_per_cache: int = Field(ge=1)
    caches: int = Field(ge=2)
    prefetch: int = Field(ge=1)
    pair: int = Field(ge=0)
    label: Literal["baseline", "candidate"]


class ReadMeasurement(BaseModel):
    """First-pass means a fresh reader, with no claim of filesystem cache eviction."""

    shuffle: bool
    phase: Literal["first", "warm"]
    construction_seconds: float
    seconds: float = Field(gt=0)
    first_sample_seconds: float
    samples: int
    payload_bytes: int
    order_sha256: str

    @computed_field
    @property
    def samples_per_second(self) -> float:
        """Include iterator setup and first-sample cost in total read throughput."""
        return self.samples / self.seconds


class Measurement(BaseModel):
    """Persist raw timing, size and correctness evidence without timing assertions in pytest."""

    spec: TrialSpec
    package_version: str
    python_version: str
    dependencies: dict[str, str]
    format_versions: tuple[int, ...]
    write_seconds: float = Field(gt=0)
    shard_bytes: int
    cache_bytes: int
    payload_bytes: int
    verified_samples: int
    peak_rss_bytes: int
    initial_rss_bytes: int
    reads: tuple[ReadMeasurement, ...]

    @computed_field
    @property
    def write_samples_per_second(self) -> float:
        """Measure end-to-end source ingestion and durable cache publication."""
        return self.verified_samples / self.write_seconds


class Manifest(BaseModel):
    """Read only the version field needed to prove the baseline actually differs."""

    format_version: int


class Comparison(BaseModel):
    """Paired medians retain raw ratios and require repeated distinct-format evidence."""

    profile: Profile
    codec: Codec
    workers: int
    metric: str
    candidate_over_baseline: float
    ratios: tuple[float, ...]
    sufficient_repeats: bool
    passed: bool


class Report(BaseModel):
    """Keep environment context and every individual trial reviewable in JSON."""

    platform: str
    python: str
    max_read_regression: float
    filesystem_cache: Literal["uncontrolled"] = "uncontrolled"
    measurements: tuple[Measurement, ...]
    comparisons: tuple[Comparison, ...]


class Benchmark(BaseSettings):
    """Run streamed text/PNG cases with identical controls in both installed releases."""

    model_config = SettingsConfigDict(cli_parse_args=False, cli_kebab_case=True)

    baseline_python: Path = Path(sys.executable)
    candidate_python: Path = Path(sys.executable)
    baseline_wheel: Path | None = None
    candidate_wheel: Path | None = None
    profiles: tuple[Profile, ...] = ("text-64", "text-256", "text-4k", "image-64k", "image-1m")
    codecs: tuple[Codec, ...] = ("none", "zstd")
    workers: tuple[int, ...] = (1, 4)
    samples_per_cache: int = Field(default=1024, ge=1)
    caches: int = Field(default=4, ge=2)
    prefetch: int = Field(default=64, ge=1)
    repeats: int = Field(default=5, ge=1)
    max_read_regression: float = Field(default=0.1, ge=0, lt=1)
    enforce: CliImplicitFlag[bool] = False
    quiet: CliImplicitFlag[bool] = False
    output: Path = Path("storage-benchmark.json")
    trial_spec: Path | None = None

    @model_validator(mode="after")
    def distinct_controls(self) -> Benchmark:
        """Reject ambiguous or empty matrices before creating any cache artifacts."""
        for values in (self.profiles, self.codecs, self.workers):
            if not values or len(set(values)) != len(values):
                raise ValueError("matrix controls must be nonempty and distinct")
        if min(self.workers) < 1:
            raise ValueError("workers must be positive")
        return self


def png_chunk(name: bytes, data: bytes) -> bytes:
    """Emit a standard PNG chunk with its required CRC, without an image dependency."""
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))


def image_payload(side: int, seed: int) -> bytes:
    """Encode seeded RGB noise as a valid PNG; already compressed image bytes resist Zstd."""
    rng = random.Random(seed)
    rows = b"".join(b"\0" + rng.randbytes(side * 3) for _ in range(side))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


def payloads(profile: Profile) -> tuple[bytes, ...]:
    """Keep eight deterministic variants in memory, independent of the dataset size."""
    match profile:
        case "image-64k" | "image-1m":
            side = 144 if profile == "image-64k" else 592
            return tuple(image_payload(side, seed) for seed in range(8))
        case "text-64" | "text-256" | "text-4k":
            size = {"text-64": 64, "text-256": 256, "text-4k": 4096}[profile]
            sentences = (
                "The research team collected training examples from several sources. ",
                "Each document contains text and a stable identifier for reproducibility. ",
                "A reader follows the planned order while workers load independent samples. ",
                "The library keeps metadata separate so users can filter their dataset. ",
                "Images and text have different storage and compression characteristics. ",
            )
            rng = random.Random(42)
            return tuple(
                "".join(rng.choices(sentences, k=size // 60 + 2)).encode()[:size] for _ in range(8)
            )


def sample_metadata(sample_id: int, cache: int) -> Metadata:
    """Use representative primitive fields and a path, identical across cache formats."""
    return {
        "index": sample_id,
        "source": cache,
        "split": "train",
        "quality": 0.95,
        "path": f"/training/corpus/partition-{cache:04d}/sample-{sample_id:012d}",
        "description": "A training sample with provenance, quality and split metadata.",
    }


@dataclass
class Source:
    """Stream a fixed pool of domain bytes without prebuilding per-sample objects."""

    name: str
    cache: int
    count: int
    pool: tuple[bytes, ...]

    def __iter__(self) -> Iterator[CacheInput]:
        """Allocate only the current input and its sample metadata."""
        for sample_id in range(self.count):
            yield CacheInput(
                self.pool[sample_id % len(self.pool)], sample_metadata(sample_id, self.cache)
            )


def read_pass(
    dataset: CachedDataset, shuffle: bool, phase: Literal["first", "warm"], construction: float
) -> ReadMeasurement:
    """Consume real returned bytes and identities; full content checks run outside timing."""
    started = time.perf_counter()
    first = 0.0
    samples = 0
    byte_count = 0
    order = hashlib.sha256()
    for sample in dataset:
        if samples == 0:
            first = time.perf_counter() - started
        samples += 1
        byte_count += len(sample.data)
        if sample.metadata["index"] != sample.sample_id:
            raise AssertionError("metadata/sample identity mismatch")
        order.update(struct.pack("<QQ", sample.cache_id, sample.sample_id))
    seconds = time.perf_counter() - started
    if samples != len(dataset):
        raise AssertionError(f"expected {len(dataset)} samples, received {samples}")
    return ReadMeasurement(
        shuffle=shuffle,
        phase=phase,
        construction_seconds=construction,
        seconds=seconds,
        first_sample_seconds=first,
        samples=samples,
        payload_bytes=byte_count,
        order_sha256=order.hexdigest(),
    )


def run_trial(spec: TrialSpec, directory: Path) -> Measurement:
    """Time durable writes and fresh/warm reads; retain a bounded fixture in this process."""
    initial_rss = psutil.Process().memory_info().rss
    pool = payloads(spec.profile)
    runtime = DatasetRuntime(num_workers=spec.workers)
    paths, write_seconds = write_fixture(spec, directory, runtime, pool)
    reads = measure_reads(spec, runtime, paths)
    verified = verify_fixture(spec, runtime, paths, pool)
    files = tuple(file for path in paths for file in path.rglob("*") if file.is_file())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return Measurement(
        spec=spec,
        package_version=version("dataset-rt"),
        python_version=platform.python_version(),
        dependencies={name: version(name) for name in DEPENDENCIES},
        format_versions=tuple(
            Manifest.model_validate_json((path / "manifest.json").read_text()).format_version
            for path in paths
        ),
        write_seconds=write_seconds,
        shard_bytes=sum(file.stat().st_size for file in files if file.parent.name == "shards"),
        cache_bytes=sum(file.stat().st_size for file in files),
        payload_bytes=sum(len(pool[index % len(pool)]) for index in range(spec.samples_per_cache))
        * spec.caches,
        verified_samples=verified,
        peak_rss_bytes=int(peak * (1 if sys.platform == "darwin" else 1024)),
        initial_rss_bytes=initial_rss,
        reads=reads,
    )


def write_fixture(
    spec: TrialSpec, directory: Path, runtime: DatasetRuntime, pool: tuple[bytes, ...]
) -> tuple[list[Path], float]:
    """Include source iteration and durable publication, excluding domain fixture generation."""
    codec = {"none": NoCompression(), "lz4": Lz4Fast(), "zstd": Zstd()}[spec.codec]
    config = WriterConfig(
        shard_compression=codec,
        show_progress=False,
        prefetch_size=spec.prefetch,
        max_shard_bytes=16 * 1024 * 1024,
    )
    started = time.perf_counter()
    paths = []
    for cache in range(spec.caches):
        result = runtime.write_cache(
            Source(f"storage-{cache}", cache, spec.samples_per_cache, pool),
            directory,
            writer_config=config,
        )[0]
        if not isinstance(result, CacheWriteSuccess):
            raise AssertionError(result)
        paths.append(result.path)
    return paths, time.perf_counter() - started


def measure_reads(
    spec: TrialSpec, runtime: DatasetRuntime, paths: list[Path]
) -> tuple[ReadMeasurement, ...]:
    """Reuse each dataset for its warm epoch, with identical seeded sampling across releases."""
    reads = []
    for shuffle in (False, True):
        started = time.perf_counter()
        dataset = runtime.cached_dataset(
            paths, reader_config=ReaderConfig(seed=42, shuffle=shuffle, prefetch_size=spec.prefetch)
        )
        construction = time.perf_counter() - started
        reads.append(read_pass(dataset, shuffle, "first", construction))
        reads.append(read_pass(dataset, shuffle, "warm", 0.0))
    return tuple(reads)


def verify_fixture(
    spec: TrialSpec, runtime: DatasetRuntime, paths: list[Path], pool: tuple[bytes, ...]
) -> int:
    """Compare every physical payload and metadata row outside throughput measurements."""
    dataset = runtime.cached_dataset(
        paths, reader_config=ReaderConfig(seed=42, shuffle=False, prefetch_size=spec.prefetch)
    )
    verified = 0
    for sample in dataset:
        expected = pool[sample.sample_id % len(pool)]
        cache = verified // spec.samples_per_cache
        if sample.data != expected or sample.metadata != sample_metadata(sample.sample_id, cache):
            raise AssertionError("payload or metadata changed")
        verified += 1
    if verified != spec.samples_per_cache * spec.caches:
        raise AssertionError("verification did not cover every physical sample")
    return verified


def compare(measurements: tuple[Measurement, ...], max_regression: float) -> tuple[Comparison, ...]:
    """Pair matching cells; validate controls and work before calculating any performance gate."""
    groups: dict[tuple[Profile, Codec, int], dict[int, dict[str, Measurement]]] = {}
    for measurement in measurements:
        spec = measurement.spec
        cell = groups.setdefault((spec.profile, spec.codec, spec.workers), {}).setdefault(
            spec.pair, {}
        )
        if spec.label in cell:
            raise ValueError("duplicate trial")
        cell[spec.label] = measurement
    comparisons = []
    for (profile, codec, workers), pairs in groups.items():
        metrics: dict[str, list[float]] = {}
        distinct_formats = True
        for pair in pairs.values():
            if set(pair) != {"baseline", "candidate"}:
                raise ValueError("incomplete trial pair")
            baseline, candidate = pair["baseline"], pair["candidate"]
            if (baseline.python_version, baseline.dependencies) != (
                candidate.python_version,
                candidate.dependencies,
            ):
                raise ValueError("paired Python or dependency versions differ")
            if baseline.spec.model_dump(exclude={"label"}) != candidate.spec.model_dump(
                exclude={"label"}
            ):
                raise ValueError("paired controls differ")
            if (baseline.payload_bytes, baseline.verified_samples) != (
                candidate.payload_bytes,
                candidate.verified_samples,
            ):
                raise ValueError("paired work differs")
            distinct_formats &= set(baseline.format_versions) == {4} and set(
                candidate.format_versions
            ) == {5}
            metrics.setdefault("write_throughput", []).append(
                baseline.write_seconds / candidate.write_seconds
            )
            metrics.setdefault("shard_bytes", []).append(
                candidate.shard_bytes / baseline.shard_bytes
            )
            metrics.setdefault("cache_bytes", []).append(
                candidate.cache_bytes / baseline.cache_bytes
            )
            metrics.setdefault("peak_rss_bytes", []).append(
                candidate.peak_rss_bytes / baseline.peak_rss_bytes
            )
            for old, new in zip(baseline.reads, candidate.reads, strict=True):
                if (old.shuffle, old.phase, old.samples, old.payload_bytes, old.order_sha256) != (
                    new.shuffle,
                    new.phase,
                    new.samples,
                    new.payload_bytes,
                    new.order_sha256,
                ):
                    raise ValueError("paired read work or order differs")
                mode = "shuffled" if old.shuffle else "sequential"
                metrics.setdefault(f"read_{mode}_{old.phase}", []).append(
                    new.samples_per_second / old.samples_per_second
                )
        sufficient = len(pairs) >= 5 and distinct_formats
        for metric, ratios in metrics.items():
            median = statistics.median(ratios)
            passed = sufficient and (
                median >= 1 - max_regression if metric.startswith("read_") else True
            )
            comparisons.append(
                Comparison(
                    profile=profile,
                    codec=codec,
                    workers=workers,
                    metric=metric,
                    candidate_over_baseline=median,
                    ratios=tuple(ratios),
                    sufficient_repeats=sufficient,
                    passed=passed,
                )
            )
    return tuple(comparisons)


def run_benchmark(config: Benchmark) -> Report:
    """Alternate release order per pair; launch fresh interpreters with no inherited runtime."""
    measurements = []
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="datasetrt-storage-") as temporary:
        directory = Path(temporary)
        baseline = prepare_interpreter(
            config.baseline_wheel, config.baseline_python, directory / "baseline"
        )
        interpreters = {
            "baseline": baseline,
            "candidate": prepare_interpreter(
                config.candidate_wheel,
                config.candidate_python,
                directory / "candidate",
                dependency_requirements(baseline),
            ),
        }
        for spec in trial_specs(config):
            if not config.quiet:
                print(
                    f"{spec.profile} {spec.codec} workers={spec.workers} pair={spec.pair + 1} {spec.label}",
                    file=sys.stderr,
                    flush=True,
                )
            measurements.append(launch_trial(spec, interpreters[spec.label], script, directory))
    values = tuple(measurements)
    return Report(
        platform=platform.platform(),
        python=platform.python_version(),
        max_read_regression=config.max_read_regression,
        measurements=values,
        comparisons=compare(values, config.max_read_regression),
    )


def trial_specs(config: Benchmark) -> Iterator[TrialSpec]:
    """Enumerate matching controls with alternating release order for every pair."""
    for profile, codec, workers, pair in product(
        config.profiles, config.codecs, config.workers, range(config.repeats)
    ):
        labels: tuple[Literal["baseline", "candidate"], ...] = ("baseline", "candidate")
        if pair % 2:
            labels = tuple(reversed(labels))
        for label in labels:
            yield TrialSpec(
                profile=profile,
                codec=codec,
                workers=workers,
                samples_per_cache=config.samples_per_cache,
                caches=config.caches,
                prefetch=config.prefetch,
                pair=pair,
                label=label,
            )


def launch_trial(spec: TrialSpec, executable: Path, script: Path, directory: Path) -> Measurement:
    """Avoid repository imports leaking into the baseline; preserve venv executable symlinks."""
    spec_path = directory / "trial.json"
    spec_path.write_text(spec.model_dump_json())
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(executable.absolute()), str(script), "--trial-spec", str(spec_path)],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return Measurement.model_validate_json(completed.stdout)


def prepare_interpreter(
    wheel: Path | None,
    executable: Path,
    directory: Path,
    requirements: tuple[str, ...] = ("pydantic-settings", "psutil"),
) -> Path:
    """Resolve baseline wheel dependencies once, then pin the candidate to that environment."""
    if wheel is None:
        return executable
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise ValueError(f"expected an existing wheel: {wheel}")
    subprocess.run(["uv", "venv", "--python", sys.executable, str(directory)], check=True)
    interpreter = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--only-binary=:all:",
            "--python",
            str(interpreter),
            str(wheel.resolve()),
            *requirements,
        ],
        check=True,
    )
    return interpreter


def dependency_requirements(executable: Path) -> tuple[str, ...]:
    """Query the actual baseline rather than assuming the runner's packages have portable wheels."""
    script = (
        "import importlib.metadata,json,sys; "
        "print(json.dumps([name + '==' + importlib.metadata.version(name) for name in sys.argv[1:]]))"
    )
    completed = subprocess.run(
        [str(executable.absolute()), "-c", script, *DEPENDENCIES],
        capture_output=True,
        text=True,
        check=True,
    )
    return TypeAdapter(tuple[str, ...]).validate_json(completed.stdout)


def main() -> None:
    """Emit typed JSON in worker mode or a complete paired report with optional gates."""
    config = CliApp.run(Benchmark)
    if config.trial_spec is not None:
        spec = TrialSpec.model_validate_json(config.trial_spec.read_text())
        with tempfile.TemporaryDirectory(prefix="fixture-") as temporary:
            print(run_trial(spec, Path(temporary)).model_dump_json())
        return
    report = run_benchmark(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(report.model_dump_json(indent=2) + "\n")
    print(config.output)
    if config.enforce and not all(item.passed for item in report.comparisons):
        raise SystemExit("storage performance acceptance failed; inspect JSON comparisons")


if __name__ == "__main__":
    main()
