"""Finite, supervised training measurements over real DatasetRT loader consumers."""

from __future__ import annotations

import gc
import hashlib
import multiprocessing as mp
import platform
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import psutil
import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, get_worker_info

from dataset_rt import (
    CachedDataset,
    CachedSample,
    CacheInput,
    CacheSource,
    CacheWriteSuccess,
    DatasetRuntime,
)
from dataset_rt.benchmarks.config import BenchmarkConfig, Topology
from dataset_rt.benchmarks.resources import (
    ResourceObserver,
    Resources,
    combine_resources,
    resource_snapshot,
)
from dataset_rt.config import ReaderConfig, WriterConfig
from dataset_rt.integrations.loader import ReaderAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator


class Digest(Protocol):
    """Accept a streaming hash without depending on hashlib's private concrete types."""

    def update(self, data: bytes, /) -> None:
        """Consume one bounded identity encoding."""
        ...


class Record(BaseModel):
    """Validate bounded result records at the collective and JSON boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PassResult(Record):
    """First pass is cold-reader startup, not proof of cold operating-system disk cache."""

    batches: int
    samples: int
    elapsed_seconds: float
    first_batch_seconds: float
    batch_wait_seconds: float
    training_seconds: float
    prefix_digest: str
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int
    resources: Resources


class RankResult(Record):
    """Keep aggregate evidence per rank; never gather payloads or complete sample lists."""

    rank: int
    local_rank: int
    hostname: str
    device: str
    dataset_rt_version: str
    torch_version: str
    python_version: str
    platform: str
    reader_sha256: str
    benchmark_sha256: str
    accelerator_build: str
    population: int
    global_requested_samples: int
    global_effective_samples: int
    loader_batches: int
    construction_seconds: float
    replay_verified: bool
    sequential_fixture_order_verified: bool
    parameter_max_difference: float
    cleanup_seconds: float
    passes: tuple[PassResult, ...]


class Report(Record):
    """Global throughput uses the slowest rank's elapsed time for each complete pass."""

    recorded_at: datetime
    settings: BenchmarkConfig
    world_size: int
    backend: str
    synthetic: bool
    ranks: tuple[RankResult, ...]
    global_samples_per_second: tuple[float, ...]


@dataclass
class Source:
    """Generate bounded deterministic fixtures privately per rank, without shared writes."""

    name: str
    cache_position: int
    count: int
    payload_bytes: int
    metadata_bytes: int

    def __iter__(self) -> Iterator[CacheInput]:
        """Encode row positions independently of the loader's partition implementation."""
        for index in range(self.count):
            position = self.cache_position * self.count + index
            header = position.to_bytes(8, "little") + index.to_bytes(8, "little")
            payload = (header * ((self.payload_bytes + 15) // 16))[: self.payload_bytes]
            yield CacheInput(
                payload,
                {
                    "benchmark_position": position,
                    "wide": str(position).ljust(self.metadata_bytes, "x")
                    if self.metadata_bytes
                    else "",
                },
            )


class Identity(NamedTuple):
    """Bounded batch evidence; position is interpreted only for synthetic inputs."""

    cache_id: int
    sample_id: int
    position: int
    worker: int


@dataclass(frozen=True)
class Sample:
    """Return CPU tensors from workers; the training rank alone owns its accelerator."""

    inputs: torch.Tensor
    identity: Identity


@dataclass(frozen=True)
class Batch:
    """Retain only one collated batch and support Torch's custom pinning protocol."""

    inputs: torch.Tensor
    identities: tuple[Identity, ...]

    def pin_memory(self) -> Batch:
        """Pin tensor storage while leaving small immutable identity records alone."""
        return Batch(self.inputs.pin_memory(), self.identities)


@dataclass(frozen=True)
class Transform:
    """Hash domain payloads into fixed-size features; this is synthetic compute, not image decoding."""

    features: int
    rounds: int
    synthetic: bool

    def __call__(self, sample: CachedSample) -> Sample:
        """Verify synthetic payload identity and perform explicitly configured CPU work."""
        position = 0
        if self.synthetic:
            value = sample.metadata["benchmark_position"]
            if type(value) is not int:
                raise ValueError("synthetic position must be an integer")
            position = value
            expected = position.to_bytes(8, "little") + sample.sample_id.to_bytes(8, "little")
            if sample.data[:16] != expected:
                raise ValueError("synthetic payload identity differs from metadata")
        digest = hashlib.sha256(sample.data).digest()
        for _ in range(self.rounds):
            digest = hashlib.sha256(digest).digest()
        inputs = torch.frombuffer(bytearray(digest), dtype=torch.uint8).float()
        inputs = inputs.repeat(self.features // 32).div_(255)
        worker = get_worker_info()
        identity = Identity(
            sample.cache_id, sample.sample_id, position, 0 if worker is None else worker.id
        )
        return Sample(inputs, identity)


def collate(samples: list[Sample]) -> Batch:
    """Stack one bounded batch and preserve its independent identity evidence."""
    return Batch(
        torch.stack([sample.inputs for sample in samples]),
        tuple(sample.identity for sample in samples),
    )


@contextmanager
def input_dataset(config: BenchmarkConfig) -> Iterator[CachedDataset]:
    """Existing caches are immutable; private synthetic fixtures outlive all rank consumers."""
    runtime = DatasetRuntime(num_workers=config.native_num_workers)
    with TemporaryDirectory(prefix="dataset-rt-benchmark-") as directory:
        paths = config.cache_paths
        if not paths:
            sources: list[CacheSource] = [
                Source(
                    f"benchmark-{index}",
                    index,
                    config.samples_per_cache,
                    config.payload_bytes,
                    config.metadata_bytes,
                )
                for index in range(config.caches)
            ]
            built = []
            for result in runtime.write_cache(
                sources, Path(directory), writer_config=WriterConfig(show_progress=False)
            ):
                if not isinstance(result, CacheWriteSuccess):
                    raise RuntimeError(result)
                built.append(result.path)
            paths = tuple(built)
        yield runtime.cached_dataset(
            paths,
            reader_config=ReaderConfig(
                seed=config.seed, shuffle=config.shuffle, prefetch_size=config.prefetch_size
            ),
        )


def make_loader(
    dataset: CachedDataset, config: BenchmarkConfig, synthetic: bool
) -> DataLoader[Batch]:
    """Exercise the public loader with the same topology and sampling policy on every rank."""
    return cast(
        "DataLoader[Batch]",
        dataset.to_torch_dataloader(
            shuffle=config.shuffle,
            seed=config.seed if config.shuffle else None,
            samples_per_epoch=config.samples_per_epoch,
            worker_partition=config.worker_partition,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            native_num_workers=config.native_num_workers,
            drop_last=config.drop_last,
            pin_memory=config.pin_memory,
            timeout=config.worker_timeout if config.num_workers else 0,
            multiprocessing_context=config.worker_context if config.num_workers else None,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
            prefetch_factor=config.prefetch_factor if config.num_workers else None,
            sample_transform_fn=Transform(
                config.input_features, config.transform_rounds, synthetic
            ),
            collate_fn=collate,
        ),
    )


def local_samples(samples: int, ranks: int, batch_size: int, drop_last: bool) -> int:
    """Independently calculate the global-budget contract used for observed-count checks."""
    return (
        samples // (ranks * batch_size) * batch_size
        if drop_last
        else (samples + ranks - 1) // ranks
    )


def file_digest(path: Path) -> str:
    """Fingerprint artifacts in bounded chunks without another full metadata allocation."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def code_digest(paths: tuple[str, ...]) -> str:
    """Identify installed source without requiring Git or expanding file contents in memory."""
    package = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode())
        digest.update(file_digest(package / relative).encode())
    return digest.hexdigest()


def reader_digest() -> str:
    """Fingerprint reconstruction and sampling orchestration, including uncommitted edits."""
    return code_digest(
        (
            "dataset.py",
            "runtime.py",
            "config.py",
            "reconstruction.py",
            "integrations/loading.py",
            "integrations/loader.py",
        )
    )


def benchmark_digest() -> str:
    """Distinguish harness changes even when installed package versions stay unchanged."""
    return code_digest(
        (
            "run_benchmark.py",
            "benchmarks/config.py",
            "benchmarks/distributed.py",
            "benchmarks/resources.py",
        )
    )


def validate_plans(
    adapter: ReaderAdapter[Sample],
    population: int,
    requested: int,
    config: BenchmarkConfig,
    topology: Topology,
) -> None:
    """Independently verify split spans and replicated populations in O(local workers)."""
    quota = local_samples(requested, topology.world_size, config.batch_size, config.drop_last)
    active = tuple(plan for plan in adapter.partitions if plan.sample_count)
    if sum(plan.sample_count for plan in active) != quota:
        raise ValueError("worker plans multiply or lose the local budget")
    if not quota:
        return
    if config.worker_partition == "replicate":
        if any(plan.offset != 0 or plan.population_size != population for plan in active):
            raise ValueError("replicated worker population is incomplete")
        return
    if requested == population:
        expected_offset, expected_length = topology.rank * quota, quota
    else:
        virtual = max(population, topology.world_size)
        size, remainder = divmod(virtual, topology.world_size)
        expected_offset = topology.rank * size + min(topology.rank, remainder)
        expected_length = size + int(topology.rank < remainder)
    offset = expected_offset
    for plan in active:
        if plan.offset != offset or not plan.population_size:
            raise ValueError("split worker spans overlap or omit positions")
        offset += plan.population_size
    if offset != expected_offset + expected_length:
        raise ValueError("split rank span has incorrect coverage")


def input_agreement(
    loader: DataLoader[Batch], config: BenchmarkConfig, topology: Topology
) -> tuple[str, str]:
    """Reject differing workloads or active metadata before ranks enter training collectives."""
    adapter = loader.dataset
    if not isinstance(adapter, ReaderAdapter):
        raise TypeError("expected DatasetRT reader adapter")
    signature = (
        reader_digest(),
        benchmark_digest(),
        file_digest(adapter.config_path.parent / "metadata.parquet"),
        config.model_dump_json(exclude={"ranks", "output", "cache_paths", "quiet"}),
    )
    gathered: list[tuple[str, str, str, str]] = [signature] * topology.world_size
    dist.all_gather_object(gathered, signature)
    if any(value != signature for value in gathered):
        raise ValueError("rank code, workloads, or active metadata differ")
    return signature[0], signature[1]


def prefix_digest(batch: Batch, digest: Digest) -> None:
    """Hash bounded-prefix physical identities without using process-dependent values."""
    for identity in batch.identities:
        digest.update(identity.cache_id.to_bytes(8, "little"))
        digest.update(identity.sample_id.to_bytes(8, "little"))


def check_order(
    batch: Batch, adapter: ReaderAdapter[Sample], counters: list[int], population: int
) -> None:
    """Check every sequential synthetic row against its consumer's cyclic planned span."""
    for identity in batch.identities:
        plan = adapter.partitions[identity.worker]
        expected = (plan.offset + counters[identity.worker] % plan.population_size) % population
        if identity.position != expected:
            raise ValueError("sequential fixture position differs from worker plan")
        counters[identity.worker] += 1


def synchronize(device: torch.device) -> None:
    """Finish accelerator work before reading wall-clock timings or reporting memory."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_pass(
    loader: DataLoader[Batch],
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    config: BenchmarkConfig,
    topology: Topology,
    device: torch.device,
    counters: list[int],
    population: int,
    synthetic: bool,
) -> PassResult:
    """Stream a finite pass; CUDA steps synchronize so reported durations include execution."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    observer = ResourceObserver()
    thread = threading.Thread(target=observer.run, daemon=True)
    synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    digest = hashlib.sha256()
    batches = samples = 0
    waiting = training = first = 0.0
    thread.start()
    iterator: Iterator[Batch] | None = None
    try:
        iterator = iter(loader)
        while True:
            before = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            waiting += time.perf_counter() - before
            if not isinstance(batch, Batch):
                raise TypeError("expected benchmark batch")
            if batches == 0:
                first = time.perf_counter() - started
            if batches < 4:
                prefix_digest(batch, digest)
            if synthetic and not config.shuffle:
                adapter = loader.dataset
                if not isinstance(adapter, ReaderAdapter):
                    raise TypeError("expected DatasetRT adapter")
                check_order(batch, adapter, counters, population)
            before = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            inputs = batch.inputs.to(device, non_blocking=config.pin_memory)
            model(inputs).square().mean().backward()
            optimizer.step()
            synchronize(device)
            training += time.perf_counter() - before
            batches += 1
            samples += len(batch.identities)
            if topology.rank == 0 and not config.quiet and batches % 100 == 0:
                print(f"{batches}/{len(loader)} batches", file=sys.stderr)
    finally:
        del iterator
        observer.stop.set()
        thread.join(5)
        if thread.is_alive():
            raise RuntimeError("resource observer did not stop")
    elapsed = time.perf_counter() - started
    if observer.failures:
        raise RuntimeError("resource observation failed") from observer.failures[0]
    resources = combine_resources(observer.peak, resource_snapshot(psutil.Process()))
    requested = population if config.samples_per_epoch is None else config.samples_per_epoch
    expected = local_samples(requested, topology.world_size, config.batch_size, config.drop_last)
    if samples != expected or batches != len(loader):
        raise ValueError("observed batches or sample count differ from reported loader length")
    counts = torch.tensor([batches, samples], device=device, dtype=torch.int64)
    minimum, maximum = counts.clone(), counts.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise ValueError("rank sample or batch counts differ")
    return PassResult(
        batches=batches,
        samples=samples,
        elapsed_seconds=elapsed,
        first_batch_seconds=first,
        batch_wait_seconds=waiting,
        training_seconds=training,
        prefix_digest=digest.hexdigest(),
        resources=resources,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    )


def verify_replay(
    dataset: CachedDataset, config: BenchmarkConfig, synthetic: bool, expected: str
) -> None:
    """Compare four fresh batches without collecting an epoch or retaining another reader."""
    from itertools import islice

    loader = make_loader(dataset, config, synthetic)
    digest = hashlib.sha256()
    iterator = iter(loader)
    try:
        for batch in islice(iterator, 4):
            if not isinstance(batch, Batch):
                raise TypeError("expected benchmark batch")
            prefix_digest(batch, digest)
    finally:
        del iterator, loader
        gc.collect()
    if digest.hexdigest() != expected:
        raise ValueError("fresh seeded loader did not reproduce its initial prefix")


def parameter_difference(model: DistributedDataParallel, device: torch.device) -> float:
    """Compare against rank zero with at most one parameter-sized temporary tensor."""
    maximum = torch.zeros((), device=device)
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        maximum = torch.maximum(maximum, (parameter.detach() - reference).abs().max())
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    result = float(maximum.item())
    if result > 1e-6:
        raise ValueError("model parameters differ across ranks")
    return result


def device_for(topology: Topology, config: BenchmarkConfig) -> torch.device:
    """Assign one CUDA device per local rank before NCCL object or tensor collectives."""
    if config.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available() or topology.local_rank >= torch.cuda.device_count():
        raise ValueError("requested CUDA topology exceeds available CUDA devices")
    if not dist.is_nccl_available():
        raise ValueError("CUDA benchmark requires a PyTorch build with NCCL")
    torch.cuda.set_device(topology.local_rank)
    return torch.device("cuda", topology.local_rank)


def execute(config: BenchmarkConfig, topology: Topology, device: torch.device) -> RankResult:
    """Own rank resources through replay checks and explicit worker cleanup."""
    synthetic = not config.cache_paths
    with input_dataset(config) as dataset:
        population = len(dataset)
        started = time.perf_counter()
        loader = make_loader(dataset, config, synthetic)
        construction = time.perf_counter() - started
        adapter = loader.dataset
        if not isinstance(adapter, ReaderAdapter):
            raise TypeError("expected DatasetRT adapter")
        requested = population if config.samples_per_epoch is None else config.samples_per_epoch
        validate_plans(adapter, population, requested, config, topology)
        del adapter
        fingerprints = input_agreement(loader, config, topology)
        torch.manual_seed(config.seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(config.input_features, config.model_width),
            torch.nn.ReLU(),
            torch.nn.Linear(config.model_width, 1),
        ).to(device)
        model = DistributedDataParallel(
            network, device_ids=[topology.local_rank] if device.type == "cuda" else None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        counters = [0] * max(1, config.num_workers)
        passes = []
        for repeat in range(config.repeats):
            if repeat and config.num_workers and not config.persistent_workers:
                counters = [0] * config.num_workers
            if topology.rank == 0 and not config.quiet:
                print(
                    f"pass {repeat + 1}/{config.repeats}; {len(loader)} batches per rank",
                    file=sys.stderr,
                )
            passes.append(
                measure_pass(
                    loader,
                    model,
                    optimizer,
                    config,
                    topology,
                    device,
                    counters,
                    population,
                    synthetic,
                )
            )
        difference = parameter_difference(model, device)
        verify_replay(dataset, config, synthetic, passes[0].prefix_digest)
        batches = len(loader)
        started = time.perf_counter()
        del loader, model, optimizer, network
        gc.collect()
        for child in mp.active_children():
            child.join(5)
            if child.is_alive():
                raise RuntimeError("loading worker survived benchmark cleanup")
        cleanup = time.perf_counter() - started
        if fingerprints != (reader_digest(), benchmark_digest()):
            raise RuntimeError("reader or benchmark code changed during measurement")
        requested = population if config.samples_per_epoch is None else config.samples_per_epoch
        effective = (
            local_samples(requested, topology.world_size, config.batch_size, config.drop_last)
            * topology.world_size
        )
        return RankResult(
            rank=topology.rank,
            local_rank=topology.local_rank,
            hostname=socket.gethostname(),
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            dataset_rt_version=version("dataset-rt"),
            torch_version=version("torch"),
            python_version=platform.python_version(),
            platform=platform.platform(),
            reader_sha256=fingerprints[0],
            benchmark_sha256=fingerprints[1],
            accelerator_build=torch.version.cuda
            or torch.version.hip
            or "not reported by this PyTorch build",
            population=population,
            global_requested_samples=requested,
            global_effective_samples=effective,
            loader_batches=batches,
            construction_seconds=construction,
            replay_verified=True,
            sequential_fixture_order_verified=synthetic and not config.shuffle,
            parameter_max_difference=difference,
            cleanup_seconds=cleanup,
            passes=tuple(passes),
        )


def run_rank(config: BenchmarkConfig) -> None:
    """Use torchrun's group identity; only global rank zero writes the complete report."""
    topology = Topology()
    if config.ranks not in (1, topology.world_size):
        raise ValueError(
            "--ranks disagrees with torchrun WORLD_SIZE; omit it under external torchrun"
        )
    torch.set_num_threads(1)
    device = device_for(topology, config)
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        init_method="env://",
        timeout=timedelta(seconds=config.timeout_seconds),
    )
    try:
        result = execute(config, topology, device)
        gathered: list[RankResult] = [result] * topology.world_size
        dist.all_gather_object(gathered, result)
        if topology.rank == 0:
            results = tuple(gathered)
            rates = tuple(
                sum(rank.passes[repeat].samples for rank in results)
                / max(rank.passes[repeat].elapsed_seconds for rank in results)
                for repeat in range(config.repeats)
            )
            report = Report(
                recorded_at=datetime.now(UTC),
                settings=config,
                world_size=topology.world_size,
                backend=dist.get_backend(),
                synthetic=not config.cache_paths,
                ranks=results,
                global_samples_per_second=rates,
            )
            config.output.parent.mkdir(parents=True, exist_ok=True)
            config.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
            print(report.model_dump_json())
    finally:
        dist.destroy_process_group()
