"""Real CPU DDP ranks and loading workers, supervised outside their collectives."""

from __future__ import annotations

import fcntl
import gc
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from itertools import islice
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import polars as pl
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import get_worker_info

from dataset_rt import CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig, WriterConfig
from dataset_rt.integrations.loader import LocalReader, ReaderAdapter
from dataset_rt.integrations.loading import ReplicaIdentity, derive_seed
from dataset_rt.reconstruction import load_configuration, load_metadata, select_population
from dataset_rt.records import CachedSample, MetadataSnapshot, ReaderRecipe

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from dataset_rt import CachedDataset

Context = Literal["fork", "spawn", "forkserver"]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "darwin", reason="Mac CPU Gloo acceptance matrix"),
]


class Source:
    """Stream a small immutable cache whose payload identifies the physical row."""

    name = "ddp"

    def __iter__(self) -> Iterator[CacheInput]:
        """Keep fixture serialization independent of Torch and NumPy."""
        for index in range(24):
            yield CacheInput(str(index).encode(), {"index": index})


class Observation(NamedTuple):
    """Attach actual process-local reader identity to a delivered native sample."""

    sample: int
    worker: int
    pid: int
    seed: int
    native_id: int


@dataclass(frozen=True)
class Batch:
    """Preserve u64 seed evidence without Torch's signed-integer collation."""

    inputs: torch.Tensor
    observations: tuple[Observation, ...]


def collate(samples: list[Observation]) -> Batch:
    """Turn bounded training samples into tensors and retain exact seed integers."""
    return Batch(torch.tensor([[item.sample / 12] for item in samples]), tuple(samples))


def checked_batches(loader: Iterable[Batch]) -> Iterator[Batch]:
    """Validate the custom collator's output at Torch's loosely typed boundary."""
    for batch in loader:
        assert isinstance(batch, Batch)
        yield batch


@dataclass(frozen=True)
class Case:
    """Separate rank launch topology from the DataLoader's worker context."""

    ranks: int
    workers: int
    context: Context
    shuffle: bool
    rows: int = 24
    fail_rank: int = -1
    persistent: bool = False
    drop_last: bool = False
    random_seed: bool = False
    split: bool = False


@dataclass(frozen=True)
class RankReport:
    """Bounded test evidence; no native object crosses the rank boundary."""

    rank: int
    pid: int
    observations: tuple[Observation, ...]
    replay: tuple[Observation, ...]
    continuation: tuple[Observation, ...]
    parameters: tuple[float, ...]
    initial_parameters: tuple[float, ...]
    steps: int
    local_length: int
    positions: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RankFailure:
    """Signal terminal rank failure without serializing arbitrary exceptions."""

    rank: int
    message: str


@pytest.fixture(autouse=True)
def ddp_slot() -> Iterator[None]:
    """Serialize multi-rank matrices across xdist to bound Mac process/RAM usage."""
    with (Path(gettempdir()) / "dataset-rt-ddp-tests.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@pytest.fixture
def recipe(tmp_path: Path) -> ReaderRecipe:
    """Publish one real cache and retain a weighted, duplicate-row snapshot."""
    runtime = DatasetRuntime(num_workers=1)
    outcomes = tuple(
        runtime.write_cache([Source()], tmp_path, writer_config=WriterConfig(show_progress=False))
    )
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert isinstance(outcome, CacheWriteSuccess)
    dataset = runtime.cached_dataset(
        [outcome.path], reader_config=ReaderConfig(seed=17, prefetch_size=2)
    )
    frame = dataset.get_metadata()
    # Distinct active positions may intentionally reference the same physical row.
    selected = pl.concat([frame.slice(0, 12), frame.slice(0, 12)])
    dataset.update_metadata(
        selected.with_columns(
            pl.Series("active_position", range(24)),
            pl.when(pl.col("sample_id") == 0).then(9.0).otherwise(1.0).alias("weight"),
        )
    )
    dataset.set_epoch_len(5)
    return dataset._reader_recipe()


def observe(sample: CachedSample) -> Observation:
    """Inspect setup without querying inherited distributed or Polars state."""
    info = get_worker_info()
    if info is None:
        return Observation(sample.sample_id, 0, os.getpid(), 0, 0)
    assert isinstance(info.dataset, ReaderAdapter)
    state = info.dataset._state
    assert isinstance(state, LocalReader) and state.pid == os.getpid()
    assert sample.data == str(sample.sample_id).encode()
    return Observation(
        sample.sample_id,
        info.id,
        os.getpid(),
        state.dataset._recipe.reader_config.seed,
        id(state.dataset._inner),
    )


def fail_transform(sample: CachedSample) -> Observation:
    """Exercise an ordinary application transform exception in a real worker."""
    raise ValueError("deliberate DDP transform failure")


def parameters(model: DistributedDataParallel) -> tuple[float, ...]:
    """Extract the tiny fixture model for cross-rank consistency assertions."""
    return tuple(
        float(value.item()) for parameter in model.parameters() for value in parameter.flatten()
    )


def run_training(
    model: DistributedDataParallel, loader: Iterable[Batch], shuffle: bool
) -> tuple[tuple[Observation, ...], int]:
    """Run ordinary DDP optimizer steps with identical finite rank quotas."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    iterator = checked_batches(loader)
    batches = islice(iterator, 128) if shuffle else iterator
    observations: list[Observation] = []
    steps = 0
    try:
        for batch in batches:
            observations.extend(batch.observations)
            optimizer.zero_grad()
            loss = model(batch.inputs).square().mean()
            loss.backward()
            optimizer.step()
            steps += 1
    finally:
        del batches, iterator
        gc.collect()
    return tuple(observations), steps


def restore_dataset(recipe: ReaderRecipe, rows: int) -> CachedDataset:
    """Restore accepted IPC and warm parent pools before loading workers launch."""
    runtime = DatasetRuntime(num_workers=1)
    dataset = runtime.cached_dataset(recipe.cache_paths, reader_config=recipe.reader_config)
    assert isinstance(recipe.metadata, MetadataSnapshot)
    dataset._restore_metadata(recipe.metadata)
    if rows != 24:
        dataset.update_metadata(dataset.get_metadata().head(rows))
    dataset.set_epoch_len(recipe.sample_count)
    assert next(iter(dataset)).data
    return dataset


def make_loader(
    dataset: CachedDataset, case: Case, *, failing: bool = False
) -> torch.utils.data.DataLoader[Batch]:
    """Apply one caller-owned configuration consistently to fresh/replayed loaders."""
    return cast(
        "torch.utils.data.DataLoader[Batch]",
        dataset.to_torch_dataloader(
            shuffle=case.shuffle,
            seed=7 if case.shuffle and not case.random_seed else None,
            samples_per_epoch=512 if case.shuffle else case.rows,
            worker_partition="replicate" if case.shuffle and not case.split else "split",
            batch_size=4,
            drop_last=case.drop_last,
            num_workers=case.workers,
            multiprocessing_context=case.context if case.workers else None,
            sample_transform_fn=fail_transform if failing else observe,
            collate_fn=collate,
            prefetch_factor=1 if case.workers else None,
            persistent_workers=case.persistent,
        ),
    )


def prefix(loader: Iterable[Batch]) -> tuple[Observation, ...]:
    """Drop a real partial iterator while retaining only bounded test evidence."""
    iterator = checked_batches(loader)
    try:
        return tuple(item for batch in islice(iterator, 4) for item in batch.observations)
    finally:
        del iterator
        gc.collect()


def partition_positions(adapter: ReaderAdapter[Observation]) -> tuple[tuple[int, ...], ...]:
    """Inspect tiny parent-prepared IPC slices independently of the partition helper.

    Unique fixture labels distinguish active positions that reference identical
    physical samples. Decode in the rank before workers start, never after fork.
    """
    positions = []
    table = load_metadata(adapter.config_path, load_configuration(adapter.config_path))
    for recipe in adapter.partitions:
        frame = pl.from_arrow(
            select_population(
                table,
                offset=recipe.offset,
                length=recipe.population_size,
                partition_seed=recipe.partition_seed,
            )
        )
        assert isinstance(frame, pl.DataFrame)
        positions.append(tuple(int(value) for value in frame["active_position"]))
    return tuple(positions)


def rank_main(
    rank: int, case: Case, recipe: ReaderRecipe, rendezvous: str, pipe: Connection
) -> None:
    """Construct native state inside a spawned rank and own its worker lifetime."""
    os.setsid()
    # These are single-Mac acceptance tests. Automatic hostname routing can select
    # an interface whose Gloo collectives time out before any loader is constructed.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=case.ranks,
            timeout=timedelta(seconds=30),
        )
        torch.manual_seed(11)
        model = DistributedDataParallel(torch.nn.Linear(1, 1))
        initial = parameters(model)
        dataset = restore_dataset(recipe, case.rows)
        loader = make_loader(dataset, case, failing=rank == case.fail_rank)
        assert isinstance(loader.dataset, ReaderAdapter)
        assert loader.dataset.replica == ReplicaIdentity(rank=rank, world_size=case.ranks)
        local_length = -1
        if not case.shuffle:
            local_length = len(loader.dataset)
        positions = partition_positions(loader.dataset)
        observations, steps = run_training(model, loader, case.shuffle)
        if case.workers == 0 and observations and not case.random_seed:
            state = loader.dataset._state
            assert isinstance(state, LocalReader)
            expected_seed = derive_seed(7, rank, 0) if case.shuffle else 0
            assert state.dataset._recipe.reader_config.seed == expected_seed
        # A newly initialized loader must replay explicit seeds, while rank/worker IDs differ.
        continuation = prefix(loader) if case.persistent else ()
        replay_loader = make_loader(dataset, case)
        replay = prefix(replay_loader)
        del replay_loader, loader
        gc.collect()
        for child in mp.active_children():
            child.join(5)
            assert not child.is_alive(), "DataLoader worker survived rank cleanup"
        pipe.send(
            RankReport(
                rank,
                os.getpid(),
                observations,
                replay,
                continuation,
                parameters(model),
                initial,
                steps,
                local_length,
                positions,
            )
        )
    except Exception as error:
        pipe.send(RankFailure(rank, f"{type(error).__name__}: {error}"[:2000]))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        pipe.close()


def stop_rank(process: BaseProcess) -> None:
    """Kill a supervised rank's isolated session, including blocked loading workers."""
    if process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
    process.join(5)


def launch(
    case: Case, recipe: ReaderRecipe, tmp_path: Path
) -> tuple[RankReport | RankFailure, ...]:
    """Bound collective hangs and terminate peers when a terminal failure is reported."""
    context = mp.get_context("spawn")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes: list[BaseProcess] = []
    pipes: list[Connection] = []
    reports: list[RankReport | RankFailure] = []
    deadline = time.monotonic() + 90
    try:
        for rank in range(case.ranks):
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=rank_main, args=(rank, case, recipe, rendezvous, child)
            )
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
        pending = set(range(case.ranks))
        while pending:
            assert time.monotonic() < deadline, "DDP launcher exceeded supervised deadline"
            for rank in tuple(pending):
                if not pipes[rank].poll(0.05):
                    assert processes[rank].exitcode is None, f"rank {rank} exited without a report"
                    continue
                report = pipes[rank].recv()
                assert isinstance(report, RankReport | RankFailure)
                reports.append(report)
                pending.remove(rank)
                if isinstance(report, RankFailure):
                    return tuple(reports)
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        return tuple(sorted(reports, key=lambda report: report.rank))
    finally:
        for process in processes:
            stop_rank(process)
        for pipe in pipes:
            pipe.close()


def verify_report(report: RankReport, case: Case) -> None:
    """Check native streams and real worker identity independently of DDP collectives."""
    assert report.parameters != report.initial_parameters
    if not case.shuffle:
        length = (case.rows + case.ranks - 1) // case.ranks
        start = report.rank * length
        assert tuple(position for worker in report.positions for position in worker) == tuple(
            position % case.rows for position in range(start, start + length)
        )
        assert report.local_length == length
    if case.shuffle:
        assert report.steps == 512 // case.ranks // 4 and report.local_length == -1
        assert [(item.sample, item.worker) for item in report.replay] == [
            (item.sample, item.worker) for item in report.observations[: len(report.replay)]
        ]
    for worker in range(max(1, case.workers)):
        observed = tuple(item for item in report.observations if item.worker == worker)
        if case.workers and observed:
            assert len({item.pid for item in observed}) == 1
            assert observed[0].pid != report.pid
            expected_seed = derive_seed(7, report.rank, worker) if case.shuffle else 0
            assert {item.seed for item in observed} == {expected_seed}
        if case.persistent:
            continued = tuple(item for item in report.continuation if item.worker == worker)
            assert continued and observed
            assert {(item.pid, item.native_id, item.seed) for item in continued} == {
                (item.pid, item.native_id, item.seed) for item in observed
            }
            if not case.shuffle:
                assert [item.sample for item in continued] == [
                    item.sample for item in observed[: len(continued)]
                ]
        if case.shuffle:
            counts = Counter(item.sample for item in observed)
            assert set(counts) <= set(range(12))
            assert counts[0] == max(counts.values())
            assert 0.25 < counts[0] / len(observed) < 0.70
            continue
        rows = report.positions[worker]
        assert [item.sample for item in observed] == [row % 12 for row in rows]


@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize(
    "workers,context",
    [
        (0, "spawn"),
        (1, "fork"),
        (1, "spawn"),
        (1, "forkserver"),
        (2, "fork"),
        (2, "spawn"),
        (2, "forkserver"),
    ],
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_real_ddp(
    recipe: ReaderRecipe, tmp_path: Path, ranks: int, workers: int, context: Context, shuffle: bool
) -> None:
    """Cross real rank counts and loading contexts through forward/backward/optimizer."""
    case = Case(ranks, workers, context, shuffle)
    reports = launch(case, recipe, tmp_path)
    assert len(reports) == ranks
    for report in reports:
        assert isinstance(report, RankReport), report
        verify_report(report, case)
    assert len({report.parameters for report in reports if isinstance(report, RankReport)}) == 1
    if not shuffle:
        assert (
            sum(report.local_length for report in reports if isinstance(report, RankReport)) == 24
        )
        assert Counter(
            item.sample
            for report in reports
            if isinstance(report, RankReport)
            for item in report.observations
        ) == Counter({sample: 2 for sample in range(12)})


@pytest.mark.parametrize("rows", [3, 13])
@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
def test_uneven_ddp(recipe: ReaderRecipe, tmp_path: Path, rows: int, context: Context) -> None:
    """Equal padded quotas keep rank lengths and optimizer step counts identical."""
    case = Case(4, 2, context, False, rows)
    reports = launch(case, recipe, tmp_path)
    assert len(reports) == 4
    for report in reports:
        assert isinstance(report, RankReport), report
        verify_report(report, case)
    assert (
        sum(len(report.observations) for report in reports if isinstance(report, RankReport))
        == (rows + 3) // 4 * 4
    )
    assert len({report.parameters for report in reports if isinstance(report, RankReport)}) == 1


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
def test_terminal_ddp_failure(recipe: ReaderRecipe, tmp_path: Path, context: Context) -> None:
    """The launcher terminates blocked peers after a real worker transform failure."""
    reports = launch(Case(2, 1, context, True, fail_rank=0), recipe, tmp_path)
    assert any(
        isinstance(report, RankFailure) and "deliberate DDP transform failure" in report.message
        for report in reports
    )


@pytest.mark.parametrize("rows", [3, 19])
@pytest.mark.parametrize("drop_last", [False, True])
def test_global_remainder_policy(
    recipe: ReaderRecipe, tmp_path: Path, rows: int, drop_last: bool
) -> None:
    """Padding/discarding gives every real rank identical sample and optimizer counts."""
    case = Case(4, 2, "spawn", False, rows, drop_last=drop_last)
    reports = launch(case, recipe, tmp_path)
    expected = rows // 16 * 4 if drop_last else (rows + 3) // 4
    assert len(reports) == 4
    for report in reports:
        assert isinstance(report, RankReport), report
        assert report.local_length == len(report.observations) == expected
        assert report.steps == (expected + 3) // 4
    assert len({report.parameters for report in reports if isinstance(report, RankReport)}) == 1


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
def test_random_common_partition_seed(
    recipe: ReaderRecipe, tmp_path: Path, context: Context
) -> None:
    """An omitted seed still gives disjoint shuffled positions and distinct rank streams."""
    reports = launch(Case(2, 1, context, True, random_seed=True, split=True), recipe, tmp_path)
    assert len(reports) == 2
    positions = []
    seeds = set()
    for report in reports:
        assert isinstance(report, RankReport), report
        assert report.steps == 64
        positions.extend(position for population in report.positions for position in population)
        seeds.update(item.seed for item in report.observations)
    assert sorted(positions) == list(range(24))
    assert positions != list(range(24))
    assert len(seeds) == 2


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_persistent_ddp(
    recipe: ReaderRecipe, tmp_path: Path, context: Context, shuffle: bool
) -> None:
    """Real persistent workers retain native identity and seed across DDP passes."""
    case = Case(2, 2, context, shuffle, persistent=True)
    reports = launch(case, recipe, tmp_path)
    assert len(reports) == 2
    for report in reports:
        assert isinstance(report, RankReport), report
        verify_report(report, case)
