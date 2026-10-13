"""Fast behavioral contracts for defaults, weighted streams, and finite epochs."""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, NamedTuple

import polars as pl
import pytest
from torch.utils.data import get_worker_info

from dataset_rt import CachedDataset, CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig
from dataset_rt.config import WriterConfig
from dataset_rt.integrations import loading
from dataset_rt.integrations.loading import derive_seed

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dataset_rt.records import CachedSample


class Source:
    """Keep the population larger than each epoch and distinguish physical samples."""

    name = "correctness"

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream bounded fixture payloads with numeric weighting metadata."""
        for index in range(32):
            yield CacheInput(str(index).encode(), {"index": index})


class Draw(NamedTuple):
    """Observe worker lifetime and physical identity without serializing native state."""

    worker: int
    pid: int
    cache: int
    sample: int


def observe(sample: CachedSample) -> Draw:
    """Return real process identity with the sample delivered by the native reader."""
    info = get_worker_info()
    return Draw(info.id if info is not None else 0, os.getpid(), sample.cache_id, sample.sample_id)


def collate(samples: list[Draw]) -> list[Draw]:
    """Keep observations unmodified by Torch's tensor collation."""
    return samples


@pytest.fixture
def dataset(tmp_path: Path) -> CachedDataset:
    """Build one tiny cache with strongly nonuniform, positive weights."""
    runtime = DatasetRuntime(num_workers=1)
    result = runtime.write_cache(
        Source(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )[0]
    match result:
        case CacheWriteSuccess(path=path):
            dataset = runtime.cached_dataset(
                [path], reader_config=ReaderConfig(seed=17, prefetch_size=2)
            )
        case failure:
            raise AssertionError(failure)
    dataset.update_metadata(
        dataset.get_metadata().with_columns(
            ((pl.col("index").cast(pl.Float64) + 1) ** 3).alias("weight")
        )
    )
    return dataset


def test_default_workers_continue_exact_weighted_stream(
    dataset: CachedDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real default workers retain streams; independent loaders replay the same full trace."""
    monkeypatch.setattr(loading, "available_cpu_count", lambda: 3)
    traces: list[list[Draw]] = []
    for _ in range(2):
        loader = dataset.to_torch_dataloader(
            seed=7,
            samples_per_epoch=8,
            batch_size=2,
            sample_transform_fn=observe,
            collate_fn=collate,
            timeout=15,
            verbose=False,
        )
        assert loader.num_workers == 2
        assert loader.persistent_workers
        assert loader.multiprocessing_context is not None
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        first = [draw for batch in loader for draw in batch]
        second = [draw for batch in loader for draw in batch]
        assert len(first) == len(second) == 8
        assert {(draw.worker, draw.pid) for draw in first} == {
            (draw.worker, draw.pid) for draw in second
        }
        traces.append(first + second)
        del loader
        gc.collect()

    assert [(d.worker, d.cache, d.sample) for d in traces[0]] == [
        (d.worker, d.cache, d.sample) for d in traces[1]
    ]
    recipe = dataset._reader_recipe()
    runtime = DatasetRuntime(num_workers=1)
    for worker in range(2):
        reference = runtime.cached_dataset(
            recipe.cache_paths,
            reader_config=ReaderConfig(seed=derive_seed(7, 0, worker), prefetch_size=2),
        )
        reference.update_metadata(dataset.get_metadata())
        reference.set_epoch_len(8)
        expected = [(sample.cache_id, sample.sample_id) for sample in reference]
        actual = [(draw.cache, draw.sample) for draw in traces[0] if draw.worker == worker]
        assert actual == expected
        assert actual[:4] != actual[4:]


@pytest.mark.parametrize(
    ("samples", "batch_size", "drop_last", "expected"),
    [
        (7, 3, False, [3, 3, 1]),
        (7, 3, True, [3, 3]),
        (1, 3, False, [1]),
        (1, 3, True, []),
    ],
)
def test_finite_epoch_accounting(
    dataset: CachedDataset, samples: int, batch_size: int, drop_last: bool, expected: list[int]
) -> None:
    """Finite short epochs have exact counts and do not shrink the active population."""
    with pytest.warns(UserWarning) if drop_last else nullcontext():
        loader = dataset.to_torch_dataloader(
            num_workers=0,
            seed=7,
            samples_per_epoch=samples,
            batch_size=batch_size,
            drop_last=drop_last,
            sample_transform_fn=observe,
            collate_fn=collate,
            verbose=False,
        )
    assert len(loader) == len(expected)
    assert [len(batch) for batch in loader] == expected
    assert dataset.get_metadata().height == 32


def test_quiet_nonpersistent_workers_warn(dataset: CachedDataset) -> None:
    """A replay-producing configuration cannot silently pass even with diagnostics disabled."""
    with pytest.warns(UserWarning, match="each epoch repeats the same sample sequence"):
        dataset.to_torch_dataloader(num_workers=1, persistent_workers=False, verbose=False)
