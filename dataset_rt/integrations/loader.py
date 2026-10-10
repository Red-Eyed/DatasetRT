"""Optional PyTorch boundary; native reader state belongs to the consuming PID."""

from __future__ import annotations

import os
import secrets
import sys
from dataclasses import dataclass, replace
from math import isfinite
from multiprocessing import get_all_start_methods
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataset_rt.integrations.loading import (
    capture_replica,
    partition_rows,
    reader_seed,
    worker_budgets,
)
from dataset_rt.metadata import decode_metadata, encode_metadata
from dataset_rt.records import CachedSample, MetadataSnapshot, ReaderRecipe
from dataset_rt.runtime import DatasetRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import polars as pl

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
        partitions: tuple[ReaderRecipe, ...],
    ) -> None:
        """Store immutable inputs; construction neither reads nor creates native state."""
        self.replica = replica
        self.shuffle = shuffle
        self.seed = seed
        self.native_num_workers = native_num_workers
        self.sample_transform_fn = sample_transform_fn
        self.row_count = row_count
        self.partitions = partitions
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
        config = replace(
            recipe,
            reader_config=recipe.reader_config.model_copy(
                update={
                    "shuffle": self.shuffle,
                    "seed": reader_seed(
                        shuffle=self.shuffle,
                        seed=self.seed,
                        rank_id=self.replica.rank,
                        worker_id=worker_id,
                    ),
                }
            ),
        )
        runtime = DatasetRuntime(num_workers=self.native_num_workers)
        dataset = runtime.cached_dataset(config.cache_paths, reader_config=config.reader_config)
        match config.metadata:
            case MetadataSnapshot() as snapshot:
                dataset._restore_metadata(snapshot)
        dataset.set_epoch_len(config.sample_count)
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
    dataset: CachedDataset,
    recipe: ReaderRecipe,
    replica: ReplicaIdentity,
    *,
    num_workers: int,
    batch_size: int,
    samples_per_epoch: int | None,
    worker_partition: Literal["split", "replicate"],
    shuffle: bool,
    seed: int | None,
) -> tuple[ReaderRecipe, ...]:
    """Snapshot active rows once, then prepare columnar populations and finite quotas.

    Preparation is O((rows + workers) × columns), including an O(rows)
    permutation for shuffled splitting. Replicated workers share one IPC value.
    No payload is read and no dataset row becomes a Python object.
    """
    match recipe.metadata:
        case MetadataSnapshot(ipc=ipc):
            frame = decode_metadata(ipc)
        case _:
            frame = dataset.get_metadata()
    if not shuffle:
        rank_span = partition_rows(frame.height, parts=replica.world_size, part_id=replica.rank)
        frame = frame.slice(rank_span.offset, rank_span.length)
    count = recipe.sample_count if samples_per_epoch is None else samples_per_epoch
    workers = max(1, num_workers)
    populations = frame.height
    if worker_partition == "replicate" and frame.height:
        populations = workers
    budgets = worker_budgets(count, workers=workers, populations=populations, batch_size=batch_size)
    if shuffle and worker_partition == "split":
        if seed is None:
            raise RuntimeError("shuffled partition preparation requires a resolved seed")
        frame = frame.sample(fraction=1.0, shuffle=True, seed=seed)
    if worker_partition == "replicate":
        snapshot = MetadataSnapshot(encode_metadata(frame))
        return tuple(replace(recipe, sample_count=budget, metadata=snapshot) for budget in budgets)
    return split_populations(frame, recipe, budgets)


def split_populations(
    frame: pl.DataFrame, recipe: ReaderRecipe, budgets: tuple[int, ...]
) -> tuple[ReaderRecipe, ...]:
    """Encode disjoint populations; a full-row epoch aligns populations with batch quotas.

    A custom budget divides the population among active consumers independently
    of the draw count, so every row remains available without copying repeats.
    """
    active = sum(budget > 0 for budget in budgets)
    full_pass = sum(budgets) == frame.height
    recipes = []
    offset = 0
    for worker, budget in enumerate(budgets):
        if full_pass:
            population_size = budget
        elif budget:
            population_size = partition_rows(frame.height, parts=active, part_id=worker).length
        else:
            population_size = 0
        snapshot = MetadataSnapshot(encode_metadata(frame.slice(offset, population_size)))
        recipes.append(replace(recipe, sample_count=budget, metadata=snapshot))
        offset += population_size
    return tuple(recipes)


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
    if shuffle:
        if seed is None:
            seed = secrets.randbits(64)
        reader_seed(shuffle=True, seed=seed, rank_id=0, worker_id=0)
    recipe = dataset._reader_recipe()
    if samples_per_epoch is None and not 1 <= recipe.sample_count <= MAX_SAMPLE_BUDGET:
        raise ValueError(f"inherited samples_per_epoch must be between 1 and {MAX_SAMPLE_BUDGET}")
    replica = capture_replica()
    partitions = prepare_partitions(
        dataset,
        recipe,
        replica,
        num_workers=num_workers,
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        worker_partition=worker_partition,
        shuffle=shuffle,
        seed=seed,
    )
    adapter = ReaderAdapter(
        replica,
        shuffle,
        seed,
        native_num_workers,
        sample_transform_fn,
        sum(partition.sample_count for partition in partitions),
        partitions,
    )
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
