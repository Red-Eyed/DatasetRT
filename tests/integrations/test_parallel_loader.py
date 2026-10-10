"""Supervised real DataLoader workers; no replacement of Torch or native APIs."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import pickle
import signal
import time
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Literal, NamedTuple

import polars as pl
import pytest
from torch.utils.data import DataLoader, get_worker_info

from dataset_rt import CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig
from dataset_rt.config import WriterConfig
from dataset_rt.integrations.loader import EmptyPartition, LocalReader, PendingReader, ReaderAdapter
from dataset_rt.integrations.loading import derive_seed
from dataset_rt.records import CachedSample, MetadataSnapshot, ReaderRecipe

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.connection import Connection
    from pathlib import Path

Context = Literal["fork", "spawn", "forkserver"]
Case = Literal[
    "validation",
    "weighted",
    "persistent-validation",
    "persistent-shuffle",
    "random-persistent",
    "replay",
    "callback-error",
    "transform-error",
    "death",
    "timeout",
    "partial",
    "empty",
    "inherited",
    "batch-tail",
    "batch-drop",
    "many",
    "native-error",
]
INIT_CALLS = 0
FIRST_ID = 2_851_758_661_582_890_383
SECOND_ID = 1_600_601_599_791_221_249


class Source:
    """Distinguish two caches while keeping sample serialization dependency-free."""

    def __init__(self, name: str) -> None:
        """Select an independent cache destination."""
        self.name = name

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream physical sample IDs with matching immutable metadata."""
        for index in range(10):
            yield CacheInput(str(index).encode(), {"index": index})


class Observation(NamedTuple):
    """Record real worker identity and initialized native state with each sample."""

    pid: int
    worker: int
    cache: int
    sample: int
    seed: int
    native_id: int
    init_calls: int
    window: int


@pytest.fixture
def recipe(tmp_path: Path) -> ReaderRecipe:
    """Retain duplicates/weights in an accepted snapshot without passing native state."""
    runtime = DatasetRuntime(num_workers=1)
    paths = []
    for result in runtime.write_cache(
        [Source("first"), Source("second")],
        tmp_path,
        writer_config=WriterConfig(show_progress=False),
    ):
        match result:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case failure:
                raise AssertionError(failure)
    dataset = runtime.cached_dataset(paths, reader_config=ReaderConfig(seed=17, prefetch_size=2))
    frame = dataset.get_metadata()
    selected = pl.concat([frame.slice(7, 1), frame.slice(12, 1), frame.slice(7, 1)])
    dataset.update_metadata(
        selected.with_columns(pl.Series("weight", [3.0, 1.0, 6.0]), pl.lit("kept").alias("extra"))
    )
    return dataset._reader_recipe()


def initialize_worker(worker_id: int) -> None:
    """Prove that caller initialization sees native setup already complete."""
    global INIT_CALLS
    info = get_worker_info()
    assert info is not None and info.id == worker_id
    assert isinstance(info.dataset, ReaderAdapter)
    match info.dataset._state:
        case LocalReader(pid=pid):
            assert pid == os.getpid()
        case EmptyPartition(pid=pid):
            assert pid == os.getpid()
            assert info.dataset.partitions[worker_id].sample_count == 0
        case PendingReader():
            raise AssertionError("caller callback ran before setup")
    INIT_CALLS += 1


def observe(sample: CachedSample) -> Observation:
    """Read process-local state without any Polars or distributed-group access."""
    info = get_worker_info()
    assert info is not None and isinstance(info.dataset, ReaderAdapter)
    state = info.dataset._state
    assert isinstance(state, LocalReader)
    assert state.pid == os.getpid()
    assert sample.data == str(sample.sample_id).encode()
    return Observation(
        os.getpid(),
        info.id,
        sample.cache_id,
        sample.sample_id,
        state.dataset.reader_config.seed,
        id(state.dataset._inner),
        INIT_CALLS,
        len(state.dataset),
    )


def identity(value: Observation) -> Observation:
    """Keep typed observations intact through unbatched Torch collation."""
    return value


def callback_error(worker_id: int) -> None:
    """Fail in actual worker initialization after internal native setup."""
    raise ValueError(f"callback failed worker={worker_id}")


def transform_error(sample: CachedSample) -> Observation:
    """Fail an ordinary transform in the consuming worker."""
    raise ValueError(f"transform failed sample={sample.sample_id}")


def worker_death(sample: CachedSample) -> Observation:
    """Terminate an actual loading worker to test Torch's failure supervision."""
    os._exit(23)


def slow_transform(sample: CachedSample) -> Observation:
    """Block long enough for the caller-selected DataLoader timeout to fire."""
    time.sleep(5)
    return observe(sample)


def batch_identity(values: list[Observation]) -> list[Observation]:
    """Keep actual worker-local batches visible without tensor conversion."""
    return values


def run_batch_case(recipe: ReaderRecipe, context: Context, drop_last: bool) -> None:
    """Demonstrate real worker tails differing from Torch's estimated batch count."""
    runtime = DatasetRuntime(num_workers=1)
    dataset = runtime.cached_dataset(recipe.cache_paths, reader_config=recipe.reader_config)
    match recipe.metadata:
        case MetadataSnapshot() as snapshot:
            dataset._restore_metadata(snapshot)
    frame = pl.concat([dataset.get_metadata()] * 3).slice(0, 6 if drop_last else 8)
    dataset.update_metadata(frame)
    loader = dataset.to_torch_dataloader(
        shuffle=False,
        batch_size=2 if drop_last else 3,
        num_workers=2,
        multiprocessing_context=context,
        drop_last=drop_last,
        timeout=15,
        worker_init_fn=initialize_worker,
        sample_transform_fn=observe,
        collate_fn=batch_identity,
    )
    assert len(loader) == 3
    assert loader.prefetch_factor == 2
    batches = list(loader)
    assert len(batches) == (2 if drop_last else 4)
    assert sum(len(batch) for batch in batches) == (4 if drop_last else 8)


def run_many_case(recipe: ReaderRecipe, context: Context) -> None:
    """Validate ten thousand row positions without materializing emitted observations."""
    runtime = DatasetRuntime(num_workers=1)
    dataset = runtime.cached_dataset(recipe.cache_paths, reader_config=recipe.reader_config)
    match recipe.metadata:
        case MetadataSnapshot() as snapshot:
            dataset._restore_metadata(snapshot)
    dataset.update_metadata(pl.concat([dataset.get_metadata()] * 3334).slice(0, 10000))
    loader = dataset.to_torch_dataloader(
        shuffle=False,
        batch_size=None,
        num_workers=2,
        multiprocessing_context=context,
        timeout=15,
        prefetch_factor=1,
        worker_init_fn=initialize_worker,
        sample_transform_fn=observe,
        collate_fn=identity,
    )
    counts = [0, 0]
    expected = [(FIRST_ID, 7), (SECOND_ID, 2), (FIRST_ID, 7)]
    for row in loader:
        position = row.worker * 5000 + counts[row.worker]
        assert (row.cache, row.sample) == expected[position % 3]
        counts[row.worker] += 1
    assert counts == [5000, 5000]


def run_case(recipe: ReaderRecipe, context: Context, case: Case) -> None:
    """Warm the real parent before launching each isolated DataLoader scenario."""
    if case in ("batch-tail", "batch-drop"):
        run_batch_case(recipe, context, case == "batch-drop")
        return
    if case == "many":
        run_many_case(recipe, context)
        return
    runtime = DatasetRuntime(num_workers=1)
    dataset = runtime.cached_dataset(recipe.cache_paths, reader_config=recipe.reader_config)
    match recipe.metadata:
        case MetadataSnapshot() as snapshot:
            dataset._restore_metadata(snapshot)
    # Exercise both native waits and Polars pools before forked loading workers.
    assert next(iter(dataset)).data
    assert dataset.get_metadata().height == 3
    if case == "empty":
        dataset.update_metadata(dataset.get_metadata().slice(0, 1))
    if case == "native-error":
        (recipe.cache_paths[0] / "manifest.json").unlink()
    workers = 4 if case == "empty" else 2
    shuffle = case not in ("validation", "persistent-validation", "empty")
    persistent = case in ("persistent-validation", "persistent-shuffle", "random-persistent")
    seed = None if case == "random-persistent" else 7
    transform = observe
    match case:
        case "transform-error":
            transform = transform_error
        case "death":
            transform = worker_death
        case "timeout":
            transform = slow_transform
        case _:
            pass
    loader = dataset.to_torch_dataloader(
        shuffle=shuffle,
        seed=seed,
        batch_size=None,
        num_workers=workers,
        multiprocessing_context=mp.get_context(context) if case == "validation" else context,
        persistent_workers=persistent,
        prefetch_factor=1,
        timeout=0.25 if case == "timeout" else 15,
        worker_init_fn=callback_error if case == "callback-error" else initialize_worker,
        sample_transform_fn=transform,
        collate_fn=identity,
    )
    assert isinstance(loader, DataLoader)
    assert loader.num_workers == workers and loader.prefetch_factor == 1
    assert loader.persistent_workers == persistent
    assert loader.multiprocessing_context is not None
    assert loader.multiprocessing_context.get_start_method() == context
    adapter = loader.dataset
    assert isinstance(adapter, ReaderAdapter)
    serialized = pickle.dumps(adapter)
    assert isinstance(adapter._state, PendingReader)
    if case in ("callback-error", "transform-error", "death", "timeout", "native-error"):
        pattern = {
            "callback-error": "callback failed",
            "transform-error": "transform failed",
            "death": "exited unexpectedly",
            "timeout": "timed out",
            "native-error": "manifest",
        }[case]
        iterator = None
        try:
            # Abrupt death may be reported during startup, before iter() returns.
            iterator = iter(loader)
            next(iterator)
        except (ValueError, RuntimeError) as error:
            assert pattern in str(error), str(error)
        else:
            raise AssertionError("worker failure did not propagate")
        finally:
            # Release while Torch is still alive, before multiprocessing's exit
            # finalizer terminates daemon workers. Torch's reraised exception can
            # retain the iterator in a traceback cycle until cyclic collection.
            del iterator
            gc.collect()
            for child in mp.active_children():
                child.join(5)
                assert not child.is_alive(), "worker survived iterator collection"
        return
    if case == "inherited":
        adapter.setup()
        child = mp.get_context("fork").Process(target=reject_inherited, args=(adapter,))
        child.start()
        child.join(15)
        try:
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.kill()
                child.join(5)
        return
    if case == "partial":
        iterator = iter(loader)
        kept = next(iterator)
        time.sleep(0.1)
        following = list(islice(iterator, 10))
        assert kept.sample in (2, 7) and len(following) == 10
        del iterator
        return
    first = (
        list(loader) if not shuffle else list(islice(loader, 1600 if case == "weighted" else 40))
    )
    assert all(row.pid != os.getpid() and row.init_calls == 1 for row in first)
    assert isinstance(adapter._state, PendingReader)
    assert pickle.dumps(adapter) == serialized
    active_workers = 1 if case == "empty" else 2
    assert len({row.pid for row in first}) == active_workers
    if not shuffle:
        assert len(loader) == len(first) == (1 if case == "empty" else 3)
        verify_validation(first, empty=case == "empty")
    else:
        verify_shuffled(first, workers, seed, weighted=case == "weighted")
    if case == "replay":
        second = list(islice(loader, 40))
        assert [(r.worker, r.cache, r.sample, r.seed) for r in first] == [
            (r.worker, r.cache, r.sample, r.seed) for r in second
        ]
        assert {row.pid for row in first}.isdisjoint({row.pid for row in second})
    if persistent:
        second = list(loader) if not shuffle else list(islice(loader, 40))
        assert {(r.pid, r.native_id, r.seed) for r in first} == {
            (r.pid, r.native_id, r.seed) for r in second
        }
        assert all(row.init_calls == 1 for row in second)
        if not shuffle:
            assert first == second
        elif seed is not None:
            assert [(r.worker, r.sample) for r in first] != [(r.worker, r.sample) for r in second]


def verify_validation(rows: list[Observation], *, empty: bool) -> None:
    """Check assigned row order including intentional duplicate physical IDs."""
    assert [(r.cache, r.sample) for r in rows if r.worker == 0] == (
        [(FIRST_ID, 7)] if empty else [(FIRST_ID, 7), (SECOND_ID, 2)]
    )
    assert [(r.cache, r.sample) for r in rows if r.worker == 1] == (
        [] if empty else [(FIRST_ID, 7)]
    )
    assert all(row.seed == 0 for row in rows)


def verify_shuffled(
    observations: list[Observation], workers: int, seed: int | None, *, weighted: bool
) -> None:
    """Check real seeds and full-population weighted frequencies without short-prefix assumptions."""
    for worker in range(workers):
        rows = [row for row in observations if row.worker == worker]
        assert rows and all(row.window == 3 for row in rows)
        assert {(row.cache, row.sample) for row in rows} <= {(FIRST_ID, 7), (SECOND_ID, 2)}
        if weighted:
            assert {(row.cache, row.sample) for row in rows} == {(FIRST_ID, 7), (SECOND_ID, 2)}
        assert len({row.seed for row in rows}) == 1
        if seed is not None:
            assert rows[0].seed == derive_seed(seed, 0, worker)
    assert len({row.seed for row in observations}) == workers
    if weighted:
        counts = Counter((row.cache, row.sample) for row in observations)
        assert 0.85 < counts[(FIRST_ID, 7)] / len(observations) < 0.95


def reject_inherited(adapter: ReaderAdapter[Observation]) -> None:
    """Reject fork inheritance before any inherited native operation is invoked."""
    try:
        adapter.setup()
    except RuntimeError as error:
        assert "another PID" in str(error)
    else:
        raise AssertionError("inherited native state was reused")


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
@pytest.mark.parametrize(
    "case",
    [
        "validation",
        "weighted",
        "persistent-validation",
        "persistent-shuffle",
        "random-persistent",
        "replay",
        "callback-error",
        "transform-error",
        "death",
        "timeout",
        "partial",
        "empty",
        "batch-tail",
        "batch-drop",
        "many",
        "native-error",
    ],
)
def test_real_parallel_loader(recipe: ReaderRecipe, context: Context, case: Case) -> None:
    """Supervise each real parent/worker tree with an independent finite deadline."""
    process = mp.get_context("spawn").Process(target=run_case, args=(recipe, context, case))
    process.start()
    try:
        process.join(50)
        assert not process.is_alive(), f"{context}/{case} exceeded deadline"
        assert process.exitcode == 0, f"{context}/{case}: exit code {process.exitcode}"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()


def test_inherited_initialized_loader_rejected(recipe: ReaderRecipe) -> None:
    """Use an actual fork to prove initialized parent state cannot be reused."""
    process = mp.get_context("spawn").Process(target=run_case, args=(recipe, "fork", "inherited"))
    process.start()
    try:
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()


class InterruptEvent(NamedTuple):
    """Typed acknowledgements from real workers and their supervised loading parent."""

    stage: Literal["ready", "interrupted", "reaped"]
    pid: int


@dataclass(frozen=True)
class BlockingInitialization:
    """Expose an in-progress caller initialization callback to the SIGINT supervisor."""

    connection: Connection

    def __call__(self, worker_id: int) -> None:
        """Acknowledge actual native setup, then keep worker initialization pending."""
        initialize_worker(worker_id)
        self.connection.send(InterruptEvent("ready", os.getpid()))
        time.sleep(30)


@dataclass(frozen=True)
class BlockingTransform:
    """Expose actual transform execution before interrupting the waiting consumer."""

    connection: Connection

    def __call__(self, sample: CachedSample) -> Observation:
        """Notify only after native sample delivery, then block the worker's output."""
        observation = observe(sample)
        self.connection.send(InterruptEvent("ready", os.getpid()))
        time.sleep(30)
        return observation


def run_interrupt_case(
    recipe: ReaderRecipe,
    context: Context,
    initializing: bool,
    persistent: bool,
    connection: Connection,
) -> None:
    """Receive real terminal-style group SIGINT and release a complete worker tree."""
    os.setsid()
    runtime = DatasetRuntime(num_workers=1)
    dataset = runtime.cached_dataset(recipe.cache_paths, reader_config=recipe.reader_config)
    match recipe.metadata:
        case MetadataSnapshot() as snapshot:
            dataset._restore_metadata(snapshot)
    assert next(iter(dataset)).data
    loader = dataset.to_torch_dataloader(
        seed=7,
        batch_size=None,
        num_workers=2,
        multiprocessing_context=context,
        persistent_workers=persistent,
        prefetch_factor=1,
        worker_init_fn=BlockingInitialization(connection) if initializing else initialize_worker,
        sample_transform_fn=observe if initializing else BlockingTransform(connection),
        collate_fn=identity,
    )
    iterator = None
    try:
        iterator = iter(loader)
        next(iterator)
    except KeyboardInterrupt:
        connection.send(InterruptEvent("interrupted", os.getpid()))
    else:
        raise AssertionError("real SIGINT did not interrupt DataLoader consumption")
    finally:
        del iterator, loader
        gc.collect()
        for child in mp.active_children():
            child.join(5)
            assert not child.is_alive(), "worker survived Ctrl+C cleanup"
    connection.send(InterruptEvent("reaped", os.getpid()))


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
@pytest.mark.parametrize("initializing", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_real_ctrl_c_reaps_workers(
    recipe: ReaderRecipe,
    context: Context,
    initializing: bool,
    persistent: bool,
) -> None:
    """Signal an isolated process group only after both real workers reach the target stage."""
    launch = mp.get_context("spawn")
    receive, send = launch.Pipe(duplex=False)
    process = launch.Process(
        target=run_interrupt_case, args=(recipe, context, initializing, persistent, send)
    )
    process.start()
    send.close()
    try:
        worker_pids: set[int] = set()
        for _ in range(2):
            assert receive.poll(20), "workers did not reach interrupt stage"
            event = receive.recv()
            assert isinstance(event, InterruptEvent) and event.stage == "ready"
            worker_pids.add(event.pid)
        assert len(worker_pids) == 2
        assert process.pid is not None
        os.killpg(process.pid, signal.SIGINT)
        for stage in ("interrupted", "reaped"):
            assert receive.poll(20), f"Ctrl+C did not reach {stage} within deadline"
            event = receive.recv()
            assert isinstance(event, InterruptEvent) and event.stage == stage
        process.join(5)
        assert process.exitcode == 0
        for pid in worker_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.is_alive():
            # The loading parent created its own group; reap descendants too
            # if an actual interrupt/cleanup regression exceeds the deadline.
            assert process.pid is not None
            os.killpg(process.pid, signal.SIGKILL)
            process.join(5)
        process.close()
        receive.close()
