"""Optional PyTorch boundary; native reader state belongs to the consuming PID."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, TypeVar

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataset_rt.integrations.loading import capture_replica, reader_seed, worker_rows
from dataset_rt.metadata import decode_metadata, encode_metadata
from dataset_rt.records import CachedSample, MetadataSnapshot, OriginalMetadata, ReaderRecipe
from dataset_rt.runtime import DatasetRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from multiprocessing.context import BaseContext

    from dataset_rt.dataset import CachedDataset
    from dataset_rt.integrations.loading import ReplicaIdentity

T = TypeVar("T")
BatchT = TypeVar("BatchT")


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
    """Initialize once and continue a full-population weighted training stream."""

    def __init__(
        self,
        recipe: ReaderRecipe,
        replica: ReplicaIdentity,
        shuffle: bool,
        seed: int | None,
        native_num_workers: int,
        sample_transform_fn: Callable[[CachedSample], T] | None,
        row_count: int,
        partitions: tuple[ReaderRecipe, ...],
    ) -> None:
        """Store immutable inputs; construction neither reads nor creates native state."""
        self.recipe = recipe
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
        recipe = self.recipe
        if not self.shuffle:
            if worker_count != len(self.partitions):
                raise RuntimeError("DataLoader worker count differs from prepared partitions")
            recipe = self.partitions[worker_id]
        if not self.shuffle and recipe.sample_count == 0:
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

    def __iter__(self) -> Iterator[CachedSample | T]:
        """Reuse native state; keep unfinished shuffled windows across iterator calls."""
        self.setup()
        match self._state:
            case EmptyPartition():
                return
            case PendingReader():
                raise RuntimeError("DatasetRT reader setup did not complete")
            case LocalReader() as state:
                if not self.shuffle:
                    for sample in state.dataset:
                        yield self._transform(sample)
                    return
                while True:
                    try:
                        sample = next(state.stream)
                    except StopIteration:
                        state.stream = iter(state.dataset)
                        continue
                    yield self._transform(sample)

    def _transform(self, sample: CachedSample) -> CachedSample | T:
        """Transform only successfully delivered native samples; errors are terminal."""
        if self.sample_transform_fn is None:
            return sample
        return self.sample_transform_fn(sample)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize reconstruction inputs only, even after zero-worker consumption."""
        return type(self), (
            self.recipe,
            self.replica,
            self.shuffle,
            self.seed,
            self.native_num_workers,
            self.sample_transform_fn,
            self.row_count,
            self.partitions,
        )


class ValidationAdapter(ReaderAdapter[T]):
    """A sized finite rank-local traversal; native state is reused between passes."""

    def __len__(self) -> int:
        """Report the prepared partition count without touching native state."""
        return self.row_count


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


def validation_partitions(
    dataset: CachedDataset, recipe: ReaderRecipe, replica: ReplicaIdentity, num_workers: int
) -> tuple[ReaderRecipe, ...]:
    """Decode once in the parent and encode disjoint slices in O(rows × columns).

    Workers restore the selected IPC directly in Rust. Repeated decoding of the
    full table per worker would scale with workers and could deadlock after fork.
    """
    match recipe.metadata:
        case MetadataSnapshot(ipc=ipc):
            frame = decode_metadata(ipc)
        case _:
            frame = dataset.get_metadata()
    recipes = []
    for worker in range(max(1, num_workers)):
        span = worker_rows(frame.height, replica, num_workers=num_workers, worker_id=worker)
        recipes.append(
            replace(
                recipe,
                sample_count=span.length,
                metadata=MetadataSnapshot(encode_metadata(frame.slice(span.offset, span.length))),
            )
        )
    return tuple(recipes)


def make_dataloader(
    dataset: CachedDataset,
    *,
    shuffle: bool,
    seed: int | None,
    batch_size: int | None,
    num_workers: int,
    sample_transform_fn: Callable[[CachedSample], T] | None,
    collate_fn: (
        Callable[[list[T]], BatchT]
        | Callable[[T], BatchT]
        | Callable[[list[CachedSample]], BatchT]
        | Callable[[CachedSample], BatchT]
        | None
    ),
    drop_last: bool,
    pin_memory: bool,
    timeout: float,
    native_num_workers: int,
    multiprocessing_context: str | BaseContext | None,
    worker_init_fn: Callable[[int], None] | None,
    prefetch_factor: int | None,
    persistent_workers: bool,
) -> DataLoader[CachedSample | T]:
    """Snapshot in the training process; delegate batching and lifecycle to Torch."""
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(native_num_workers) is not int or native_num_workers < 1:
        raise ValueError("native_num_workers must be a positive integer")
    if num_workers == 0 and timeout != 0:
        raise ValueError("timeout must be zero with num_workers=0")
    if prefetch_factor is not None and (type(prefetch_factor) is not int or prefetch_factor < 1):
        raise ValueError("prefetch_factor must be a positive integer")
    if shuffle and seed is not None:
        # Validate an explicit seed now without drawing entropy for omitted seeds.
        reader_seed(shuffle=True, seed=seed, rank_id=0, worker_id=0)
    replica = capture_replica()
    recipe = dataset._reader_recipe()
    row_count = recipe.sample_count
    partitions: tuple[ReaderRecipe, ...] = ()
    adapter_type = ReaderAdapter if shuffle else ValidationAdapter
    if not shuffle:
        partitions = validation_partitions(dataset, recipe, replica, num_workers)
        row_count = sum(partition.sample_count for partition in partitions)
        # Prepared partitions replace the override for validation. Keeping the
        # original global IPC too would duplicate unrelated rank rows in workers.
        recipe = replace(recipe, metadata=OriginalMetadata())
    adapter = adapter_type(
        recipe,
        replica,
        shuffle,
        seed,
        native_num_workers,
        sample_transform_fn,
        row_count,
        partitions,
    )
    return DataLoader(
        adapter,
        batch_size=batch_size,
        num_workers=num_workers,
        # Torch types only batched collation, but passes one sample when batch_size=None.
        collate_fn=collate_fn,  # pyrefly: ignore[bad-argument-type]
        drop_last=drop_last,
        pin_memory=pin_memory,
        timeout=timeout,
        multiprocessing_context=multiprocessing_context,
        worker_init_fn=WorkerInitializer(worker_init_fn),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
