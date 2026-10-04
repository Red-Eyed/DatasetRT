"""Prove process-local streaming reconstruction against unchanged native code."""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import pickle
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataset_rt import CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig, WriterConfig


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


class RecipeReader(IterableDataset[tuple[int, int, int, int]]):
    """Test-only adapter; all native objects live in the consuming iterator."""

    def __init__(self, recipe: ReaderRecipe) -> None:
        """Retain a serializable recipe without constructing native state."""
        self.recipe = recipe

    def __len__(self) -> int:
        """Report a local budget without reading caches or starting threads."""
        return self.recipe.samples

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        """Reconstruct the full weighted population after worker identity is known."""
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        worker_count = 1 if info is None else info.num_workers
        quota = self.recipe.samples // worker_count
        runtime = DatasetRuntime(num_workers=1)
        dataset = runtime.cached_dataset(
            self.recipe.paths,
            reader_config=ReaderConfig(seed=self.recipe.seed + worker_id, prefetch_size=4),
        )
        dataset.update_metadata(pl.read_ipc(io.BytesIO(self.recipe.metadata_ipc)))
        dataset.set_epoch_len(quota)
        population_size = len(dataset.get_metadata())
        for sample in dataset:
            assert sample.data == str(sample.sample_id).encode()
            yield os.getpid(), worker_id, sample.sample_id, population_size


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


@pytest.mark.parametrize("context", ["serial", "spawn", "fork", "forkserver"])
def test_process_local_weighted_streams(recipe: ReaderRecipe, context: str) -> None:
    """Fresh workers retain full probability mass and reproducible distinct streams."""
    if context != "serial" and context not in mp.get_all_start_methods():
        pytest.skip(f"{context} unavailable on this platform")
    # Keep initialized parent native objects alive to exercise the real fork hazard.
    parent = DatasetRuntime(num_workers=1)
    parent_dataset = parent.cached_dataset(recipe.paths, reader_config=ReaderConfig(seed=3))
    assert next(iter(parent_dataset)).data
    adapter = RecipeReader(recipe)
    assert len(adapter) == 800
    assert pickle.loads(pickle.dumps(adapter)).recipe == recipe
    workers = 0 if context == "serial" else 2
    loader = DataLoader(
        adapter,
        batch_size=None,
        num_workers=workers,
        multiprocessing_context=None if workers == 0 else context,
        timeout=0 if workers == 0 else 30,
    )
    first = list(loader)
    second = list(loader)
    assert len(first) == len(loader) == 800
    assert [(row[1], row[2]) for row in first] == [(row[1], row[2]) for row in second]
    assert all(row[3] == 3 for row in first)
    counts = Counter(row[2] for row in first)
    assert set(counts) == {0, 2}
    assert 0.80 < counts[2] / 800 < 0.91
    pids = {row[0] for row in first}
    assert len(pids) == max(workers, 1)
    assert (os.getpid() in pids) == (workers == 0)
    if workers:
        streams = [[row[2] for row in first if row[1] == worker] for worker in range(workers)]
        assert streams[0] != streams[1]
    assert pickle.loads(pickle.dumps(adapter)).recipe == recipe
