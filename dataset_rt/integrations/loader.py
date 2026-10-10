"""Optional PyTorch boundary; native reader state belongs to the consuming PID."""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from math import isfinite
from multiprocessing import get_all_start_methods
from multiprocessing.context import BaseContext
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataset_rt.integrations.loading import (
    WorkerPlan,
    capture_replica,
    common_seed,
    partition_rows,
    rank_budget,
    reader_seed,
    worker_budgets,
)
from dataset_rt.reconstruction import (
    ConstructionArtifacts,
    DatasetConfiguration,
    load_configuration,
    load_metadata,
    select_population,
)
from dataset_rt.records import CachedSample, RowSpan

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pydantic import DirectoryPath, FilePath

    from dataset_rt.dataset import CachedDataset
    from dataset_rt.integrations.loading import ReplicaIdentity

T = TypeVar("T")
BatchT = TypeVar("BatchT")

# Torch computes ceil(sample_count / batch_size) through floating-point division.
MAX_SAMPLE_BUDGET = min(sys.maxsize, 1 << 53)


@dataclass(frozen=True)
class PendingReader:
    """No native resources exist before first consumption."""


@dataclass(frozen=True)
class EmptyPartition:
    """Setup completed in this PID; no assigned rows require a native reader."""

    pid: int


@dataclass
class LocalReader:
    """A native dataset and retained draw iterator may only be used in their PID."""

    pid: int
    dataset: CachedDataset
    stream: Iterator[CachedSample]


class ReaderAdapter(IterableDataset[CachedSample | T], Generic[T]):
    """A finite sample budget over a prepared process-local population."""

    def __init__(
        self,
        replica: ReplicaIdentity,
        shuffle: bool,
        seed: int | None,
        native_num_workers: int,
        sample_transform_fn: Callable[[CachedSample], T] | None,
        row_count: int,
        partitions: tuple[WorkerPlan, ...],
        config_path: FilePath,
    ) -> None:
        """Store immutable inputs; construction neither reads nor creates native state."""
        self.replica = replica
        self.shuffle = shuffle
        self.seed = seed
        self.native_num_workers = native_num_workers
        self.sample_transform_fn = sample_transform_fn
        self.row_count = row_count
        self.partitions = partitions
        self.config_path = config_path
        self._artifacts: ConstructionArtifacts | None = None
        self._state: PendingReader | EmptyPartition | LocalReader = PendingReader()

    def setup(self) -> None:
        """Create native state once in the consuming PID and reject inherited reuse."""
        match self._state:
            case LocalReader(pid=pid) | EmptyPartition(pid=pid):
                if pid != os.getpid():
                    raise RuntimeError("DatasetRT loader native state belongs to another PID")
                return
            case PendingReader():
                pass
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        worker_count = 1 if info is None else info.num_workers
        if worker_count != len(self.partitions):
            raise RuntimeError("DataLoader worker count differs from prepared partitions")
        recipe = self.partitions[worker_id]
        if recipe.sample_count == 0:
            self._state = EmptyPartition(os.getpid())
            return
        if recipe.rank != self.replica.rank or recipe.worker_id != worker_id:
            raise RuntimeError("worker plan does not match consumer identity")
        from dataset_rt.dataset import CachedDataset

        config = load_configuration(self.config_path)
        metadata = select_population(
            load_metadata(self.config_path, config),
            offset=recipe.offset,
            length=recipe.population_size,
            partition_seed=recipe.partition_seed,
        )
        config = config.model_copy(
            update={
                "num_workers": self.native_num_workers,
                "epoch_len": recipe.sample_count,
                "row_count": metadata.num_rows,
                "reader_config": config.reader_config.model_copy(
                    update={"shuffle": self.shuffle, "seed": recipe.seed}
                ),
            }
        )
        dataset = CachedDataset._from_configuration(config, metadata)
        self._state = LocalReader(os.getpid(), dataset, iter(dataset))

    def __len__(self) -> int:
        """Expose the loader's finite sample budget for Torch's batch-count calculation."""
        return self.row_count

    def __iter__(self) -> Iterator[CachedSample | T]:
        """Emit one worker quota, retaining unfinished native windows between calls."""
        self.setup()
        match self._state:
            case EmptyPartition():
                return
            case PendingReader():
                raise RuntimeError("DatasetRT reader setup did not complete")
            case LocalReader() as state:
                for _ in range(len(state.dataset)):
                    try:
                        sample = next(state.stream)
                    except StopIteration:
                        state.stream = iter(state.dataset)
                        sample = next(state.stream)
                    yield self._transform(sample)

    def _transform(self, sample: CachedSample) -> CachedSample | T:
        """Transform only successfully delivered native samples; errors are terminal."""
        if self.sample_transform_fn is None:
            return sample
        return self.sample_transform_fn(sample)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize reconstruction inputs only, even after zero-worker consumption."""
        return type(self), (
            self.replica,
            self.shuffle,
            self.seed,
            self.native_num_workers,
            self.sample_transform_fn,
            self.row_count,
            self.partitions,
            self.config_path,
        )


@dataclass(frozen=True)
class WorkerInitializer:
    """Compose native setup with an ordinary picklable caller initialization hook."""

    callback: Callable[[int], None] | None

    def __call__(self, worker_id: int) -> None:
        """Initialize native state before user code can inspect its worker dataset."""
        info = get_worker_info()
        if info is None or not isinstance(info.dataset, ReaderAdapter):
            raise RuntimeError("DatasetRT initialization requires its DataLoader worker")
        info.dataset.setup()
        if self.callback is not None:
            self.callback(worker_id)


def prepare_partitions(
    config: DatasetConfiguration,
    replica: ReplicaIdentity,
    *,
    num_workers: int,
    batch_size: int,
    samples_per_epoch: int,
    worker_partition: Literal["split", "replicate"],
    shuffle: bool,
    seed: int | None,
    drop_last: bool,
) -> tuple[WorkerPlan, ...]:
    """Plan global/rank/worker quotas in O(workers), without materializing metadata."""
    count = rank_budget(samples_per_epoch, replica, batch_size, drop_last)
    workers = max(1, num_workers)
    population = config.row_count
    if worker_partition == "replicate":
        rank_span = RowSpan(0, population)
        populations = workers
    else:
        if samples_per_epoch == population:
            population = count * replica.world_size
        elif population < replica.world_size:
            population = (
                (population + replica.world_size - 1) // replica.world_size * replica.world_size
            )
        rank_span = partition_rows(population, parts=replica.world_size, part_id=replica.rank)
        populations = rank_span.length
    budgets = worker_budgets(count, workers=workers, populations=populations, batch_size=batch_size)
    active = sum(budget > 0 for budget in budgets)
    full_pass = sum(budgets) == rank_span.length
    plans = []
    offset = rank_span.offset
    for worker, budget in enumerate(budgets):
        if budget == 0:
            population_size = 0
        elif worker_partition == "replicate":
            population_size = config.row_count
        elif full_pass:
            population_size = budget
        else:
            population_size = partition_rows(rank_span.length, parts=active, part_id=worker).length
        plans.append(
            WorkerPlan(
                rank=replica.rank,
                worker_id=worker,
                offset=0 if worker_partition == "replicate" else offset,
                population_size=population_size,
                sample_count=budget,
                seed=reader_seed(
                    shuffle=shuffle, seed=seed, rank_id=replica.rank, worker_id=worker
                ),
                partition_seed=seed if shuffle and worker_partition == "split" else None,
            )
        )
        offset += population_size
    return tuple(plans)


def _validate_worker_options(
    *,
    num_workers: int,
    native_num_workers: int,
    timeout: float,
    multiprocessing_context: Literal["fork", "spawn", "forkserver"] | BaseContext | None,
    prefetch_factor: int | None,
    persistent_workers: bool,
) -> None:
    """Reject unusable worker settings before exporting dataset-scale metadata."""
    if type(num_workers) is not int or not 0 <= num_workers < 1 << 32:
        raise ValueError("num_workers must be a nonnegative integer smaller than 2**32")
    if type(native_num_workers) is not int or not 1 <= native_num_workers <= sys.maxsize:
        raise ValueError("native_num_workers must be a positive integer fitting sys.maxsize")
    if not isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and nonnegative")
    if num_workers == 0 and timeout != 0:
        raise ValueError("timeout must be zero with num_workers=0")
    if prefetch_factor is not None and (type(prefetch_factor) is not int or prefetch_factor < 1):
        raise ValueError("prefetch_factor must be a positive integer")
    if num_workers == 0 and prefetch_factor is not None:
        raise ValueError("prefetch_factor requires num_workers > 0")
    if num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")
    if multiprocessing_context is None:
        return
    if num_workers == 0:
        raise ValueError("multiprocessing_context requires num_workers > 0")
    if isinstance(multiprocessing_context, str):
        if multiprocessing_context not in get_all_start_methods():
            raise ValueError("multiprocessing_context must be an available start method")
        return
    if not isinstance(multiprocessing_context, BaseContext):
        raise ValueError("multiprocessing_context must be a start method or BaseContext")


class _PickleSink:
    """Discard serialized probe output instead of retaining callback-sized buffers."""

    def write(self, data: bytes) -> int:
        """Satisfy the pickler's stream boundary without keeping its output."""
        return len(data)


def _validate_transform(transform: Callable[[CachedSample], T] | None, num_workers: int) -> None:
    """Fail before export when a multiprocess transform cannot be serialized."""
    if transform is None or num_workers == 0:
        return
    try:
        ForkingPickler(_PickleSink(), pickle.HIGHEST_PROTOCOL).dump(transform)
    except Exception as error:
        raise TypeError("sample_transform_fn must be picklable for multiprocess loading") from error


def make_dataloader(
    dataset: CachedDataset,
    *,
    shuffle: bool,
    seed: int | None,
    samples_per_epoch: int | None,
    worker_partition: Literal["split", "replicate"],
    batch_size: int,
    num_workers: int,
    sample_transform_fn: Callable[[CachedSample], T] | None,
    collate_fn: (Callable[[list[T]], BatchT] | Callable[[list[CachedSample]], BatchT] | None),
    drop_last: bool,
    pin_memory: bool,
    timeout: float,
    native_num_workers: int,
    multiprocessing_context: Literal["fork", "spawn", "forkserver"] | BaseContext | None,
    worker_init_fn: Callable[[int], None] | None,
    prefetch_factor: int | None,
    persistent_workers: bool,
    work_dir: DirectoryPath | None,
) -> DataLoader[CachedSample | T]:
    """Snapshot in the training process; delegate batching and lifecycle to Torch."""
    _validate_worker_options(
        num_workers=num_workers,
        native_num_workers=native_num_workers,
        timeout=timeout,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    if type(batch_size) is not int or not 1 <= batch_size <= sys.maxsize:
        raise ValueError("batch_size must be a positive integer fitting sys.maxsize")
    for name, value in (
        ("shuffle", shuffle),
        ("drop_last", drop_last),
        ("pin_memory", pin_memory),
        ("persistent_workers", persistent_workers),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if samples_per_epoch is not None and (
        type(samples_per_epoch) is not int or not 1 <= samples_per_epoch <= MAX_SAMPLE_BUDGET
    ):
        raise ValueError(
            f"samples_per_epoch must be a positive integer <= {MAX_SAMPLE_BUDGET} or None"
        )
    if worker_partition not in ("split", "replicate"):
        raise ValueError("worker_partition must be 'split' or 'replicate'")
    if not shuffle and seed is not None:
        raise ValueError("seed requires shuffle=True; omit seed for sequential reading")
    if shuffle and seed is not None:
        reader_seed(shuffle=True, seed=seed, rank_id=0, worker_id=0)
    _validate_transform(sample_transform_fn, num_workers)
    recipe = dataset._reader_recipe()
    if samples_per_epoch is None and not 1 <= recipe.sample_count <= MAX_SAMPLE_BUDGET:
        raise ValueError(f"inherited samples_per_epoch must be between 1 and {MAX_SAMPLE_BUDGET}")
    replica = capture_replica()
    seed = common_seed(replica, shuffle, seed)
    artifacts = ConstructionArtifacts.create(work_dir)
    try:
        config_path = dataset.dump_config(artifacts.directory)
        config = load_configuration(config_path)
        partitions = prepare_partitions(
            config,
            replica,
            num_workers=num_workers,
            batch_size=batch_size,
            samples_per_epoch=recipe.sample_count
            if samples_per_epoch is None
            else samples_per_epoch,
            worker_partition=worker_partition,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        adapter = ReaderAdapter(
            replica,
            shuffle,
            seed,
            native_num_workers,
            sample_transform_fn,
            sum(partition.sample_count for partition in partitions),
            partitions,
            config_path,
        )
        adapter._artifacts = artifacts
        return DataLoader(
            adapter,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            timeout=timeout,
            multiprocessing_context=multiprocessing_context,
            worker_init_fn=WorkerInitializer(worker_init_fn),
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    except BaseException:
        artifacts.discard()
        raise
