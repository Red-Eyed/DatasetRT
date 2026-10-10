"""Artifact reconstruction, compact transport, and parent-owned output lifetimes."""

from __future__ import annotations

import gc
import json
import pickle
import warnings
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import polars as pl
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from dataset_rt import (
    CachedDataset,
    CacheInput,
    CacheWriteSuccess,
    DatasetRuntime,
    ReaderConfig,
    reconstruction,
)
from dataset_rt.config import WriterConfig
from dataset_rt.integrations.loader import ReaderAdapter
from dataset_rt.integrations.loading import ReplicaIdentity, rank_budget

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dataset_rt.records import CachedSample


class Source:
    """Stream physical identities without holding a sample collection."""

    name = "reconstruction"

    def __iter__(self) -> Iterator[CacheInput]:
        """Supply stable payloads for independent reconstructed readers."""
        for index in range(10):
            yield CacheInput(str(index).encode(), {"index": index})


@pytest.fixture
def dataset(tmp_path: Path) -> CachedDataset:
    """Retain native source state while constructing independent artifacts."""
    runtime = DatasetRuntime(num_workers=2)
    result = runtime.write_cache(
        Source(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    return runtime.cached_dataset([result.path], reader_config=ReaderConfig(seed=7, shuffle=False))


def collate(samples: list[CachedSample]) -> list[int]:
    """Expose bounded batch payloads without Torch's default record collation."""
    return [int(sample.data) for sample in samples]


def test_round_trip_preserves_snapshot_and_epoch(dataset: CachedDataset, tmp_path: Path) -> None:
    """Restore duplicate rows, extras, weights, and epoch overrides with fresh cursors."""
    original = dataset.get_metadata()
    frame = pl.concat([original.slice(8, 1), original.slice(1, 1), original.slice(8, 1)])
    frame = frame.with_columns(pl.Series("weight", [3.0, 1.0, 6.0]), pl.lit("kept").alias("extra"))
    dataset.update_metadata(frame)
    dataset.set_epoch_len(7)
    export = tmp_path / "export"
    export.mkdir()
    config = dataset.dump_config(export)
    assert set(path.name for path in export.iterdir()) == {"dataset.json", "metadata.parquet"}
    dataset.update_metadata(original)
    restored = CachedDataset.from_config(config)
    assert_frame_equal(restored.get_metadata(), frame)
    assert len(restored) == 7
    assert restored._runtime_num_workers == 2
    assert [sample.sample_id for sample in restored] == [8, 1, 8, 8, 1, 8, 8]
    assert [sample.sample_id for sample in CachedDataset.from_config(config)] == [
        8,
        1,
        8,
        8,
        1,
        8,
        8,
    ]
    relocated = tmp_path / "moved"
    export.rename(relocated)
    assert_frame_equal(CachedDataset.from_config(relocated / config.name).get_metadata(), frame)


def test_export_refuses_overwrite(dataset: CachedDataset, tmp_path: Path) -> None:
    """Existing snapshots remain byte-for-byte unchanged after a rejected export."""
    directory = tmp_path / "export"
    directory.mkdir()
    config = dataset.dump_config(directory)
    before = config.read_bytes()
    with pytest.raises(FileExistsError):
        dataset.dump_config(directory)
    assert config.read_bytes() == before


def test_extra_column_types_survive_selection(dataset: CachedDataset, tmp_path: Path) -> None:
    """Primitive and nested extras retain their logical types through shuffled selection."""
    frame = dataset.get_metadata().with_columns(
        pl.lit(date(2026, 10, 10)).alias("day"),
        pl.lit(datetime(2026, 10, 10, 12)).alias("timestamp"),
        pl.lit(["a", "b"]).alias("labels"),
        pl.struct(pl.col("index"), pl.lit("kept").alias("text")).alias("record"),
    )
    dataset.update_metadata(frame)
    directory = tmp_path / "export"
    directory.mkdir()
    assert_frame_equal(
        CachedDataset.from_config(dataset.dump_config(directory)).get_metadata(), frame
    )
    loader = dataset.to_torch_dataloader(shuffle=True, seed=7, collate_fn=collate)
    assert len(list(loader)) == 10


def test_native_validation_rejects_bad_saved_weights(
    dataset: CachedDataset, tmp_path: Path
) -> None:
    """Editing a saved artifact cannot bypass Rust's authoritative weight validation."""
    directory = tmp_path / "export"
    directory.mkdir()
    config = dataset.dump_config(directory)
    frame = dataset.get_metadata().with_columns(pl.lit(-1.0).alias("weight"))
    pq.write_table(frame.to_arrow(), directory / "metadata.parquet")
    with pytest.raises(ValueError, match="weight"):
        CachedDataset.from_config(config)


def test_file_directory_boundaries(dataset: CachedDataset, tmp_path: Path) -> None:
    """DirectoryPath and FilePath contracts fail before reader construction."""
    with pytest.raises(ValidationError):
        dataset.dump_config(tmp_path / "missing")
    with pytest.raises(ValidationError):
        CachedDataset.from_config(tmp_path)


@pytest.mark.parametrize("loader", [False, True])
def test_failed_export_cleans_only_owned_files(
    dataset: CachedDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: bool
) -> None:
    """A failed artifact write removes its partial output while preserving caller files."""
    directory = tmp_path / "export"
    directory.mkdir()
    sentinel = directory / "caller.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def fail_sync(path: Path) -> None:
        """Simulate storage failure after a complete temporary metadata write."""
        raise OSError(f"cannot synchronize {path.name}")

    monkeypatch.setattr(reconstruction, "_sync_file", fail_sync)
    with pytest.raises(OSError, match="cannot synchronize"):
        if loader:
            dataset.to_torch_dataloader(work_dir=directory)
        else:
            dataset.dump_config(directory)
    assert tuple(directory.iterdir()) == (sentinel,)


@pytest.mark.parametrize("corruption", ["metadata", "config", "rows", "version"])
def test_broken_artifacts_fail_clearly(
    dataset: CachedDataset, tmp_path: Path, corruption: str
) -> None:
    """Incomplete, malformed, and incompatible snapshots never become usable readers."""
    directory = tmp_path / "export"
    directory.mkdir()
    config = dataset.dump_config(directory)
    if corruption == "metadata":
        (directory / "metadata.parquet").unlink()
    elif corruption == "config":
        config.write_text("invalid", encoding="utf-8")
    else:
        data = json.loads(config.read_bytes())
        data["row_count" if corruption == "rows" else "version"] = 11
        config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        CachedDataset.from_config(config)


def test_transform_validation_precedes_export(dataset: CachedDataset, tmp_path: Path) -> None:
    """Reject local callbacks before creating output, while permitting serial callbacks."""
    with pytest.raises(TypeError, match="sample_transform_fn must be picklable"):
        dataset.to_torch_dataloader(
            num_workers=1, work_dir=tmp_path, sample_transform_fn=lambda sample: sample.data
        )
    assert not tuple(tmp_path.glob("dataset-rt-*"))
    loader = dataset.to_torch_dataloader(
        shuffle=False, sample_transform_fn=lambda sample: int(sample.data), batch_size=2
    )
    assert next(iter(loader)).tolist() == [0, 1]


def test_compact_pickle_does_not_include_metadata(dataset: CachedDataset) -> None:
    """Wide metadata increases artifact size without increasing worker control payloads."""
    first = dataset.to_torch_dataloader(shuffle=False, num_workers=4)
    assert isinstance(first.dataset, ReaderAdapter)
    baseline = len(pickle.dumps(first.dataset))
    frame = dataset.get_metadata().with_columns(pl.lit("x" * 1_000_000).alias("wide"))
    dataset.update_metadata(frame)
    second = dataset.to_torch_dataloader(shuffle=False, num_workers=4)
    assert isinstance(second.dataset, ReaderAdapter)
    assert abs(len(pickle.dumps(second.dataset)) - baseline) < 64
    assert len(pickle.dumps(second.dataset)) < 4096


def test_temporary_artifacts_follow_active_iterator(dataset: CachedDataset) -> None:
    """Deleting the loader cannot invalidate an iterator that still owns its dataset."""
    loader = dataset.to_torch_dataloader(shuffle=False, collate_fn=collate)
    assert isinstance(loader.dataset, ReaderAdapter)
    directory = loader.dataset.config_path.parent
    iterator = iter(loader)
    del loader
    gc.collect()
    assert directory.exists()
    assert next(iterator) == [0]
    del iterator
    gc.collect()
    assert not directory.exists()


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_temporary_artifacts_follow_worker_iterator(
    dataset: CachedDataset, context: Literal["fork", "spawn"]
) -> None:
    """Real persistent children finish before the final parent owner releases artifacts."""
    loader = dataset.to_torch_dataloader(
        shuffle=False,
        num_workers=1,
        multiprocessing_context=context,
        persistent_workers=True,
        collate_fn=collate,
        timeout=15,
    )
    assert isinstance(loader.dataset, ReaderAdapter)
    directory = loader.dataset.config_path.parent
    iterator = iter(loader)
    assert next(iterator) == [0]
    del loader
    gc.collect()
    assert directory.exists()
    assert len(list(iterator)) == 9
    del iterator
    gc.collect()
    assert not directory.exists()


def test_explicit_work_dir_is_isolated_and_retained(dataset: CachedDataset, tmp_path: Path) -> None:
    """Independent loaders publish distinct snapshots under caller-owned output."""
    first = dataset.to_torch_dataloader(shuffle=False, work_dir=tmp_path)
    second = dataset.to_torch_dataloader(shuffle=False, work_dir=tmp_path)
    assert isinstance(first.dataset, ReaderAdapter) and isinstance(second.dataset, ReaderAdapter)
    paths = (first.dataset.config_path, second.dataset.config_path)
    assert paths[0] != paths[1]
    del first, second
    gc.collect()
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize("drop_last,expected", [(False, 334), (True, 320)])
def test_global_budget_warns_only_in_rank_zero(drop_last: bool, expected: int) -> None:
    """The 1000/3/32 example gives identical quotas with one adjustment warning."""
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        quotas = [
            rank_budget(1000, ReplicaIdentity(rank=rank, world_size=3), 32, drop_last)
            for rank in range(3)
        ]
    assert quotas == [expected] * 3
    assert len(emitted) == 1
    assert f"effective {expected * 3}" in str(emitted[0].message)
