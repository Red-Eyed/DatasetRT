"""Fast behavioral contracts for defaults, weighted streams, and finite epochs."""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal, NamedTuple
from unittest.mock import patch

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


@pytest.mark.parametrize("workers", [1, "auto"])
@pytest.mark.parametrize("partition", [None, "split", "replicate"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("seed", [7, "random"])
def test_nonpersistent_workers_rejected_before_export(
    dataset: CachedDataset,
    monkeypatch: pytest.MonkeyPatch,
    workers: int | Literal["auto"],
    partition: Literal["split", "replicate"] | None,
    shuffle: bool,
    seed: int | Literal["random"],
) -> None:
    """Reject resolved replay settings before snapshots, RNG selection, or worker startup."""
    monkeypatch.setattr(loading, "available_cpu_count", lambda: 3)

    def unexpected_metadata_access(self: CachedDataset) -> None:
        """Catch dataset preparation before rejecting invalid worker settings."""
        raise AssertionError("replay settings reached metadata preparation")

    monkeypatch.setattr(CachedDataset, "_reader_recipe", unexpected_metadata_access)
    with pytest.raises(ValueError) as caught:
        dataset.to_torch_dataloader(
            num_workers=workers,
            worker_partition=partition,
            shuffle=shuffle,
            seed=seed if shuffle else "random",
            persistent_workers=False,
            verbose=False,
        )
    message = str(caught.value)
    for detail in (
        "persistent_workers=False",
        "replays",
        "seed='random'",
        "persistent_workers=True",
        "num_workers=0",
    ):
        assert detail in message


@pytest.mark.parametrize("cpus", [1, 3])
@pytest.mark.parametrize("workers", [0, "auto"])
def test_nonpersistent_workers_allow_only_main_process(
    dataset: CachedDataset,
    monkeypatch: pytest.MonkeyPatch,
    cpus: int,
    workers: int | Literal["auto"],
) -> None:
    """Automatic counts reject actual processes and allow a retained main-process reader."""
    monkeypatch.setattr(loading, "available_cpu_count", lambda: cpus)
    if workers == "auto" and cpus > 1:
        with pytest.raises(ValueError, match="persistent_workers=False"):
            dataset.to_torch_dataloader(num_workers=workers, persistent_workers=False)
        return
    loader = dataset.to_torch_dataloader(
        num_workers=workers,
        seed=7,
        samples_per_epoch=8,
        persistent_workers=False,
        sample_transform_fn=observe,
        collate_fn=collate,
        verbose=False,
    )
    assert loader.num_workers == 0
    assert not loader.persistent_workers
    first = [draw for batch in loader for draw in batch]
    second = [draw for batch in loader for draw in batch]
    assert len(first) == len(second) == 8
    assert [(draw.cache, draw.sample) for draw in first] != [
        (draw.cache, draw.sample) for draw in second
    ]


@pytest.mark.parametrize(("samples", "batch_size", "world_size"), [(1, 3, 1), (7, 2, 4)])
def test_zero_batch_epoch_rejected_before_export(
    dataset: CachedDataset,
    monkeypatch: pytest.MonkeyPatch,
    samples: int,
    batch_size: int,
    world_size: int,
) -> None:
    """Impossible batch budgets fail before metadata export and random-seed collectives."""

    def unexpected_export(self: CachedDataset, path: Path | None = None) -> None:
        """Detect expensive artifact preparation on an invalid epoch budget."""
        raise AssertionError("invalid epoch reached metadata export")

    monkeypatch.setattr(CachedDataset, "dump_config", unexpected_export)
    replica = loading.ReplicaIdentity(world_size=world_size)
    with patch("dataset_rt.integrations.loader.capture_replica", return_value=replica):
        with pytest.raises(ValueError, match="zero batches") as caught:
            dataset.to_torch_dataloader(
                num_workers=0,
                samples_per_epoch=samples,
                batch_size=batch_size,
                drop_last=True,
                verbose=False,
            )
    assert f"require at least {batch_size * world_size} samples" in str(caught.value)


@pytest.mark.parametrize(
    ("workers", "samples", "batch_size", "world_size", "reject"),
    [
        (0, 8, 1, 1, False),
        (4, 1, 1, 1, False),
        (4, 3, 2, 1, False),
        (4, 4, 2, 1, True),
        (0, 8, 1, 2, True),
    ],
)
def test_sequential_replication_counts_active_consumers(
    dataset: CachedDataset,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    samples: int,
    batch_size: int,
    world_size: int,
    reject: bool,
) -> None:
    """Only multiple active replicated cursors duplicate sequential prefixes."""

    def unexpected_export(self: CachedDataset, path: Path) -> None:
        """Detect metadata export even if the loader would clean up afterward."""
        raise AssertionError(f"invalid replication reached metadata export at {path}")

    if reject:
        monkeypatch.setattr(CachedDataset, "dump_config", unexpected_export)
    replica = loading.ReplicaIdentity(world_size=world_size)
    with patch("dataset_rt.integrations.loader.capture_replica", return_value=replica):
        with (
            pytest.raises(ValueError, match="duplicates sequential samples")
            if reject
            else nullcontext()
        ):
            loader = dataset.to_torch_dataloader(
                shuffle=False,
                worker_partition="replicate",
                num_workers=workers,
                samples_per_epoch=samples,
                verbose=False,
                batch_size=batch_size,
            )
            assert len(loader) == (samples + batch_size - 1) // batch_size
