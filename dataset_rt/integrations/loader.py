"""Optional PyTorch boundary; native reader state belongs to the consuming PID."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, TypeVar

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataset_rt.integrations.loading import capture_replica, reader_seed, worker_rows
from dataset_rt.metadata import encode_metadata, slice_metadata
from dataset_rt.records import CachedSample, MetadataSnapshot, ReaderRecipe
from dataset_rt.runtime import DatasetRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from dataset_rt.dataset import CachedDataset
    from dataset_rt.integrations.loading import ReplicaIdentity

T = TypeVar("T")


@dataclass(frozen=True)
class PendingReader:
    """No native resources exist before first consumption."""


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
    ) -> None:
        """Store immutable inputs; construction neither reads nor creates native state."""
        self.recipe = recipe
        self.replica = replica
        self.shuffle = shuffle
        self.seed = seed
        self.native_num_workers = native_num_workers
        self.sample_transform_fn = sample_transform_fn
        self.row_count = row_count
        self._state: PendingReader | LocalReader = PendingReader()

    def setup(self) -> None:
        """Create native state once in the consuming PID and reject inherited reuse."""
        match self._state:
            case LocalReader(pid=pid):
                if pid != os.getpid():
                    raise RuntimeError("DatasetRT loader native state belongs to another PID")
                return
            case PendingReader():
                pass
        if get_worker_info() is not None:
            raise RuntimeError("multiprocess DataLoader support is not enabled yet")
        if not self.shuffle and self.row_count == 0:
            return
        config = replace(
            self.recipe,
            reader_config=self.recipe.reader_config.model_copy(
                update={
                    "shuffle": self.shuffle,
                    "seed": reader_seed(
                        shuffle=self.shuffle,
                        seed=self.seed,
                        rank_id=self.replica.rank,
                        worker_id=0,
                    ),
                }
            ),
        )
        runtime = DatasetRuntime(num_workers=self.native_num_workers)
        dataset = runtime.cached_dataset(config.cache_paths, reader_config=config.reader_config)
        match config.metadata:
            case MetadataSnapshot() as snapshot:
                dataset._restore_metadata(snapshot)
        dataset.set_epoch_len(config.sample_count if self.shuffle else self.row_count)
        self._state = LocalReader(os.getpid(), dataset, iter(dataset))

    def __iter__(self) -> Iterator[CachedSample | T]:
        """Reuse native state; keep unfinished shuffled windows across iterator calls."""
        self.setup()
        match self._state:
            case PendingReader():
                return
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
        )


class ValidationAdapter(ReaderAdapter[T]):
    """A sized finite rank-local traversal; native state is reused between passes."""

    def __len__(self) -> int:
        """Report the prepared partition count without touching native state."""
        return self.row_count


def make_dataloader(
    dataset: CachedDataset,
    *,
    shuffle: bool,
    seed: int | None,
    batch_size: int | None,
    num_workers: int,
    sample_transform_fn: Callable[[CachedSample], T] | None,
    collate_fn: Callable[..., object] | None,
    drop_last: bool,
    pin_memory: bool,
    timeout: float,
    native_num_workers: int,
) -> DataLoader[CachedSample | T]:
    """Snapshot in the training process; delegate batching and lifecycle to Torch."""
    if type(num_workers) is not int or num_workers != 0:
        raise ValueError("to_torch_dataloader currently requires num_workers=0")
    if type(native_num_workers) is not int or native_num_workers < 1:
        raise ValueError("native_num_workers must be a positive integer")
    if timeout != 0:
        raise ValueError("timeout must be zero with num_workers=0")
    if shuffle and seed is not None:
        # Validate an explicit seed now without drawing entropy for omitted seeds.
        reader_seed(shuffle=True, seed=seed, rank_id=0, worker_id=0)
    replica = capture_replica()
    recipe = dataset._reader_recipe()
    row_count = recipe.sample_count
    adapter_type = ReaderAdapter if shuffle else ValidationAdapter
    if not shuffle:
        # Sequential validation requires an explicit columnar snapshot. No such
        # export is needed for ordinary shuffled loaders with original metadata.
        snapshot = recipe.metadata
        if not isinstance(snapshot, MetadataSnapshot):
            snapshot = MetadataSnapshot(encode_metadata(dataset.get_metadata()))
        from dataset_rt.metadata import decode_metadata

        count = decode_metadata(snapshot.ipc).height
        span = worker_rows(count, replica, num_workers=0, worker_id=0)
        recipe = replace(recipe, metadata=slice_metadata(snapshot, span))
        row_count = span.length
    adapter = adapter_type(
        recipe, replica, shuffle, seed, native_num_workers, sample_transform_fn, row_count
    )
    return DataLoader(
        adapter,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
        timeout=timeout,
    )
