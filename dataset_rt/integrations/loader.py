"""Optional PyTorch boundary; native reader state belongs to the consuming PID."""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataset_rt.integrations.loader_logging import loader_event
from dataset_rt.integrations.loader_options import MAX_SAMPLE_BUDGET, LoaderOptions, ResolvedWorkers
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
    population_selection,
    prepare_row_order,
)
from dataset_rt.records import CachedSample, RowSpan

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from multiprocessing.context import BaseContext

    from pydantic import DirectoryPath, FilePath

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
        verbose: bool = True,
        persistent_workers: bool = False,
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
        self.verbose = verbose
        self.persistent_workers = persistent_workers
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
        recipe = self._worker_plan()
        if recipe.sample_count == 0:
            self._state = EmptyPartition(os.getpid())
            self._log_reader_event("reader_empty", recipe)
            return
        dataset = self._create_dataset(recipe)
        self._state = LocalReader(os.getpid(), dataset, iter(dataset))
        self._log_reader_event("reader_started", recipe)

    def _log_reader_event(self, event: str, recipe: WorkerPlan) -> None:
        """Report construction/reset and reuse without touching native sampling progress."""
        if not self.verbose:
            return
        match get_worker_info():
            case None:
                consumer = "main"
            case info:
                consumer = str(info.id)
        loader_event(
            "event=%s pid=%s rank=%s worker=%s seed=%s native_threads=%s "
            "persistent=%s samples=%s population=%s sampling_reset=%s",
            event,
            os.getpid(),
            recipe.rank,
            consumer,
            recipe.seed,
            self.native_num_workers if recipe.sample_count else 0,
            self.persistent_workers,
            recipe.sample_count,
            recipe.population_size,
            event == "reader_started",
        )

    def _worker_plan(self) -> WorkerPlan:
        """Validate the actual consumer identity against its prepared partition."""
        match get_worker_info():
            case None:
                worker_id, worker_count = 0, 1
            case info:
                worker_id, worker_count = info.id, info.num_workers
        if worker_count != len(self.partitions):
            raise RuntimeError("DataLoader worker count differs from prepared partitions")
        recipe = self.partitions[worker_id]
        if recipe.rank != self.replica.rank or recipe.worker_id != worker_id:
            raise RuntimeError("worker plan does not match consumer identity")
        return recipe

    def _create_dataset(self, recipe: WorkerPlan) -> CachedDataset:
        """Build process-owned native state from mapped immutable inputs and validated quotas."""
        from dataset_rt.dataset import CachedDataset

        config = load_configuration(self.config_path)
        metadata = load_metadata(self.config_path, config)
        selection = population_selection(
            self.config_path,
            config.row_count,
            offset=recipe.offset,
            length=recipe.population_size,
            partition_seed=recipe.partition_seed,
        )
        config = config.model_copy(
            update={
                "num_workers": self.native_num_workers,
                "epoch_len": recipe.sample_count,
                "row_count": recipe.population_size,
                "reader_config": config.reader_config.model_copy(
                    update={"shuffle": self.shuffle, "seed": recipe.seed}
                ),
            }
        )
        return CachedDataset._from_configuration(config, metadata, selection)

    def __len__(self) -> int:
        """Expose the loader's finite sample budget for Torch's batch-count calculation."""
        return self.row_count

    def __iter__(self) -> Iterator[CachedSample | T]:
        """Emit one worker quota, retaining unfinished native windows between calls."""
        self.setup()
        if self.verbose:
            self._log_reader_event("reader_pass", self._worker_plan())
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
            self.verbose,
            self.persistent_workers,
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


def _warn_restarting_workers(workers: ResolvedWorkers) -> None:
    """Expose replay risk only when loading processes actually discard their readers."""
    if workers.counts.processes == 0 or workers.persistent:
        return
    warnings.warn(
        "persistent_workers=False restarts DatasetRT readers on every DataLoader pass "
        "with the same sampling seed and reset draw offset or sequential cursor. "
        "With unchanged metadata, each epoch repeats the same sample sequence, "
        "including when seed='random'. Short epochs can repeatedly omit the rest of "
        "the population. Set persistent_workers=True to continue sampling across epochs.",
        UserWarning,
        stacklevel=4,
    )


def _prepare_adapter(
    dataset: CachedDataset,
    options: LoaderOptions,
    workers: ResolvedWorkers,
    replica: ReplicaIdentity,
    seed: int | None,
    sample_count: int,
    transform: Callable[[CachedSample], T] | None,
    artifacts: ConstructionArtifacts,
) -> ReaderAdapter[T]:
    """Publish immutable inputs and prepare compact worker plans without native reader setup."""
    config_path = dataset.dump_config(artifacts.directory)
    config = load_configuration(config_path)
    partition = options.resolved_partition()
    partitions = prepare_partitions(
        config,
        replica,
        num_workers=workers.counts.processes,
        batch_size=options.batch_size,
        samples_per_epoch=sample_count,
        worker_partition=partition,
        shuffle=options.shuffle,
        seed=seed,
        drop_last=options.drop_last,
    )
    if options.shuffle and partition == "split":
        if seed is None:
            raise RuntimeError("shuffled partitions require a resolved seed")
        prepare_row_order(config_path, config.row_count, seed)
    adapter = ReaderAdapter(
        replica,
        options.shuffle,
        seed,
        workers.counts.native_threads,
        transform,
        sum(plan.sample_count for plan in partitions),
        partitions,
        config_path,
        verbose=options.verbose,
        persistent_workers=workers.persistent,
    )
    adapter._artifacts = artifacts
    return adapter


def make_dataloader(
    dataset: CachedDataset,
    *,
    shuffle: bool,
    seed: int | Literal["random"],
    samples_per_epoch: int | None,
    worker_partition: Literal["split", "replicate"] | None,
    batch_size: int,
    num_workers: int | Literal["auto"],
    sample_transform_fn: Callable[[CachedSample], T] | None,
    collate_fn: (Callable[[list[T]], BatchT] | Callable[[list[CachedSample]], BatchT] | None),
    drop_last: bool,
    pin_memory: bool,
    timeout: float,
    native_num_workers: int | Literal["auto"],
    multiprocessing_context: Literal["inherit", "fork", "spawn", "forkserver"] | BaseContext,
    worker_init_fn: Callable[[int], None] | None,
    prefetch_factor: int | Literal["auto"],
    persistent_workers: bool | Literal["auto"],
    verbose: bool,
    work_dir: DirectoryPath | None,
) -> DataLoader[CachedSample | T]:
    """Snapshot in the training process; delegate batching and lifecycle to Torch."""
    options = LoaderOptions(
        shuffle=shuffle,
        seed=seed,
        samples_per_epoch=samples_per_epoch,
        partition=worker_partition,
        batch_size=batch_size,
        num_workers=num_workers,
        native_num_workers=native_num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        timeout=timeout,
        context=multiprocessing_context,
        prefetch=prefetch_factor,
        verbose=verbose,
    )
    replica = capture_replica()
    workers = options.resolve_workers(replica.world_size)
    _validate_transform(sample_transform_fn, workers.counts.processes)
    _warn_restarting_workers(workers)
    recipe = dataset._reader_recipe()
    sample_count = options.samples_per_epoch
    if sample_count is None:
        sample_count = recipe.sample_count
    if not 1 <= sample_count <= MAX_SAMPLE_BUDGET:
        raise ValueError(f"inherited samples_per_epoch must be between 1 and {MAX_SAMPLE_BUDGET}")
    resolved_seed = common_seed(replica, options.shuffle, options.resolved_seed())
    artifacts = ConstructionArtifacts.create(work_dir)
    try:
        adapter = _prepare_adapter(
            dataset,
            options,
            workers,
            replica,
            resolved_seed,
            sample_count,
            sample_transform_fn,
            artifacts,
        )
        loader = DataLoader(
            adapter,
            batch_size=batch_size,
            num_workers=workers.counts.processes,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            timeout=timeout,
            multiprocessing_context=workers.context,
            worker_init_fn=WorkerInitializer(worker_init_fn),
            prefetch_factor=workers.prefetch,
            persistent_workers=workers.persistent,
        )
        if options.verbose:
            seeds = ",".join(f"{plan.worker_id}:{plan.seed}" for plan in adapter.partitions)
            match workers.prefetch:
                case None:
                    prefetch: int | str = "disabled"
                case int() as prefetch:
                    pass
            loader_event(
                "event=loader_ready pid=%s rank=%s seed=%s worker_seeds=%s "
                "num_workers=%s native_threads_per_consumer=%s partition=%s "
                "persistent=%s prefetch_factor=%s native_prefetch=%s samples=%s samples_per_rank=%s "
                "batch_size=%s world_size=%s shuffle=%s multiprocessing_context=%s",
                os.getpid(),
                replica.rank,
                resolved_seed,
                seeds,
                workers.counts.processes,
                workers.counts.native_threads,
                options.resolved_partition(),
                workers.persistent,
                prefetch,
                recipe.reader_config.prefetch_size,
                sample_count,
                adapter.row_count,
                batch_size,
                replica.world_size,
                shuffle,
                workers.context.get_start_method() if workers.context is not None else "disabled",
            )
        return loader
    except BaseException:
        artifacts.discard()
        raise
