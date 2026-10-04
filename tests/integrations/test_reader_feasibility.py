"""Prove process-local streaming reconstruction against unchanged native code."""

from __future__ import annotations

import copy
import io
import multiprocessing as mp
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import polars as pl
import pytest
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

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
    from multiprocessing.process import BaseProcess
    from pathlib import Path

ParentState = Literal["cold", "runtime", "reader"]
RUNTIME_CONSTRUCTIONS = 0


def trace_stage(stage: str) -> None:
    """Flush opt-in milestones directly, without inheriting logging-handler locks."""
    if os.environ.get("DATASETRT_TRACE_FORK") == "1":
        os.write(2, f"DatasetRT pid={os.getpid()} ppid={os.getppid()} {stage}\n".encode())


class ObservedRuntime(DatasetRuntime):
    """Record the actual constructor PID rather than infer ownership from output PID."""

    def __init__(self, *, num_workers: int) -> None:
        """Observe construction while leaving native behavior unchanged."""
        global RUNTIME_CONSTRUCTIONS
        trace_stage("runtime.construct.begin")
        super().__init__(num_workers=num_workers)
        self.constructor_pid = os.getpid()
        RUNTIME_CONSTRUCTIONS += 1
        trace_stage("runtime.construct.end")


class ReaderObservation(NamedTuple):
    """Small fixture-only observations carry constructor and consumer identities."""

    pid: int
    worker_id: int
    sample_id: int
    population_size: int
    constructor_pid: int


class PopulationSource:
    """Small distinct payloads make physical identities observable in workers."""

    name = "population"

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream records without retaining payload or metadata row lists."""
        for index in range(4):
            yield CacheInput(str(index).encode(), {"index": index})


@dataclass(frozen=True)
class ReaderRecipe:
    """Only immutable construction values cross the DataLoader boundary."""

    paths: tuple[Path, ...]
    metadata_ipc: bytes
    samples: int = 800
    seed: int = 17


class RecipeReader(IterableDataset[ReaderObservation]):
    """Test-only adapter; all native objects live in the consuming iterator."""

    def __init__(self, recipe: ReaderRecipe) -> None:
        """Retain a serializable recipe without constructing native state."""
        self.recipe = recipe

    def __len__(self) -> int:
        """Report a local budget without reading caches or starting threads."""
        return self.recipe.samples

    def __iter__(self) -> Iterator[ReaderObservation]:
        """Reconstruct the full weighted population after worker identity is known."""
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        worker_count = 1 if info is None else info.num_workers
        quota = self.recipe.samples // worker_count
        trace_stage(f"adapter.iter.begin worker={worker_id}")
        runtime = ObservedRuntime(num_workers=1)
        trace_stage("dataset.load.begin")
        dataset = runtime.cached_dataset(
            self.recipe.paths,
            reader_config=ReaderConfig(seed=self.recipe.seed + worker_id, prefetch_size=4),
        )
        trace_stage("dataset.load.end metadata.decode.begin")
        metadata = pl.read_ipc(io.BytesIO(self.recipe.metadata_ipc))
        trace_stage("metadata.decode.end metadata.update.begin")
        dataset.update_metadata(metadata)
        trace_stage("metadata.update.end")
        dataset.set_epoch_len(quota)
        population_size = len(dataset.get_metadata())
        trace_stage("native.iter.begin")
        for sample in dataset:
            assert sample.data == str(sample.sample_id).encode()
            yield ReaderObservation(
                os.getpid(), worker_id, sample.sample_id, population_size, runtime.constructor_pid
            )
        trace_stage("native.iter.end")


@pytest.fixture
def recipe(tmp_path: Path) -> ReaderRecipe:
    """Use a validated duplicate/reordered columnar population as reconstruction input."""
    runtime = DatasetRuntime(num_workers=1)
    results = runtime.write_cache(
        PopulationSource(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )
    match results[0]:
        case CacheWriteSuccess(path=path):
            paths = (path,)
        case error:
            raise AssertionError(error)
    dataset = runtime.cached_dataset(paths, reader_config=ReaderConfig(seed=17))
    frame = dataset.get_metadata()
    frame = pl.concat([frame.slice(2, 1), frame.slice(0, 1), frame.slice(2, 1)])
    frame = frame.with_columns(pl.Series("weight", [3.0, 1.0, 3.0]), pl.lit("kept").alias("extra"))
    dataset.update_metadata(frame)
    buffer = frame.write_ipc(None)
    assert buffer is not None
    return ReaderRecipe(paths, buffer.getvalue())


@pytest.mark.parametrize("object_kind", ["runtime", "dataset"])
def test_native_state_cannot_be_serialized_or_copied(
    recipe: ReaderRecipe, object_kind: str
) -> None:
    """Native objects cannot be reconstruction inputs; only recipes may cross workers."""
    runtime = DatasetRuntime(num_workers=1)
    dataset = runtime.cached_dataset(recipe.paths, reader_config=ReaderConfig(seed=3))
    value = runtime if object_kind == "runtime" else dataset
    with pytest.raises(TypeError):
        pickle.dumps(value)
    with pytest.raises(TypeError):
        copy.copy(value._inner)
    with pytest.raises(TypeError):
        copy.deepcopy(value._inner)


def run_reader_case(recipe: ReaderRecipe, context: str, parent_state: ParentState) -> None:
    """Isolate the loading parent from pytest's fixture and other native activity."""
    trace_stage(f"case.begin context={context} parent={parent_state}")
    assert RUNTIME_CONSTRUCTIONS == 0
    parent = None
    parent_dataset = None
    if parent_state != "cold":
        parent = ObservedRuntime(num_workers=1)
    if parent_state == "reader":
        assert parent is not None
        parent_dataset = parent.cached_dataset(recipe.paths, reader_config=ReaderConfig(seed=3))
        assert next(iter(parent_dataset)).data
    constructions_before = RUNTIME_CONSTRUCTIONS
    adapter = RecipeReader(recipe)
    assert vars(adapter) == {"recipe": recipe}
    assert len(adapter) == 800
    assert pickle.loads(pickle.dumps(adapter)).recipe == recipe
    workers = 0 if context == "serial" else 2
    loader = DataLoader(
        adapter,
        batch_size=None,
        num_workers=workers,
        multiprocessing_context=None if workers == 0 else context,
        timeout=0 if workers == 0 else 15,
    )
    assert len(loader) == 800
    assert RUNTIME_CONSTRUCTIONS == constructions_before
    trace_stage("loader.first.begin")
    first = list(loader)
    trace_stage("loader.first.end loader.second.begin")
    second = list(loader)
    trace_stage("loader.second.end")
    assert len(first) == len(loader) == 800
    assert [(row[1], row[2]) for row in first] == [(row[1], row[2]) for row in second]
    assert all(row[3] == 3 for row in first)
    assert all(row[0] == row[4] for row in first + second)
    counts = Counter(row[2] for row in first)
    assert set(counts) == {0, 2}
    assert 0.80 < counts[2] / 800 < 0.91
    pids = {row[0] for row in first}
    assert len(pids) == max(workers, 1)
    assert (os.getpid() in pids) == (workers == 0)
    if workers:
        assert RUNTIME_CONSTRUCTIONS == constructions_before
        streams = [[row[2] for row in first if row[1] == worker] for worker in range(workers)]
        assert streams[0] != streams[1]
    else:
        assert RUNTIME_CONSTRUCTIONS == constructions_before + 2
    assert pickle.loads(pickle.dumps(adapter)).recipe == recipe
    assert vars(adapter) == {"recipe": recipe}
    del parent_dataset, parent
    trace_stage("case.end")


@pytest.mark.parametrize("context", ["serial", "fork", "spawn", "forkserver"])
@pytest.mark.parametrize("parent_state", ["cold", "runtime", "reader"])
def test_process_local_weighted_streams(
    recipe: ReaderRecipe, context: str, parent_state: ParentState
) -> None:
    """Exercise the real __iter__ path without inheriting the pytest parent's native state."""
    if context != "serial" and context not in mp.get_all_start_methods():
        pytest.skip(f"{context} unavailable on this platform")
    case = mp.get_context("spawn").Process(
        target=run_reader_case, args=(recipe, context, parent_state)
    )
    verify_process(case, f"{context}/{parent_state}")


def verify_process(case: BaseProcess, label: str) -> None:
    """Reap a supervised case with a finite deadline even when native code fails."""
    case.start()
    try:
        case.join(60)
        assert not case.is_alive(), f"{label} exceeded its deadline"
        assert case.exitcode == 0, f"{label} failed: exit code {case.exitcode}"
    finally:
        if case.is_alive():
            case.terminate()
            case.join(5)
        if case.is_alive():
            case.kill()
            case.join(5)
        case.close()


@dataclass
class ForkWriteSource:
    """Enough streamed inputs to exercise bounded writer credits and shard rotation."""

    name: str
    samples: int = 1000

    def __iter__(self) -> Iterator[CacheInput]:
        """Keep fixture payloads independent of worker launch and parent native state."""
        for index in range(self.samples):
            yield CacheInput(b"x" * 1024, {"index": index})


def write_and_read_in_fork_child(destination: Path, list_mode: bool) -> None:
    """Fresh child-owned runtimes cover both writer pipelines and direct result waits."""
    trace_stage(f"writer.child.begin list={list_mode}")
    runtime = DatasetRuntime(num_workers=2)
    sources: CacheSource | list[CacheSource] = ForkWriteSource("one")
    if list_mode:
        sources = [ForkWriteSource("one"), ForkWriteSource("two")]
    results = runtime.write_cache(
        sources,
        destination,
        writer_config=WriterConfig(prefetch_size=2, max_shard_bytes=64 * 1024, show_progress=False),
    )
    paths = []
    for result in results:
        match result:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case error:
                raise AssertionError(error)
    dataset = runtime.cached_dataset(
        paths, reader_config=ReaderConfig(seed=17, prefetch_size=2, validate_cache=True)
    )
    assert dataset.get_item(0, 999).data == b"x" * 1024
    assert sum(len(sample.data) for sample in dataset) == len(paths) * 1000 * 1024
    del dataset, runtime
    trace_stage("writer.child.end")


def run_fork_writer_case(recipe: ReaderRecipe, destination: Path, list_mode: bool) -> None:
    """Warm the parent before launching a child that receives only paths and flags."""
    parent = DatasetRuntime(num_workers=1)
    parent_dataset = parent.cached_dataset(recipe.paths, reader_config=ReaderConfig(seed=3))
    assert next(iter(parent_dataset)).data
    child = mp.get_context("fork").Process(
        target=write_and_read_in_fork_child, args=(destination, list_mode)
    )
    verify_process(child, f"fork writer list={list_mode}")


@pytest.mark.parametrize("list_mode", [False, True])
def test_fresh_fork_writer_after_parent_read(
    recipe: ReaderRecipe, tmp_path: Path, list_mode: bool
) -> None:
    """Verify forked single/multi-source writing plus fresh streaming/direct readers."""
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork unavailable on this platform")
    case = mp.get_context("spawn").Process(
        target=run_fork_writer_case, args=(recipe, tmp_path / "fork-writer", list_mode)
    )
    verify_process(case, f"fork writer supervisor list={list_mode}")
