"""Minimal reader recipe, seed, and columnar validation-partition contracts."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from dataset_rt import CachedDataset, CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig
from dataset_rt.config import WriterConfig
from dataset_rt.integrations.loading import (
    ReplicaIdentity,
    capture_replica,
    derive_seed,
    partition_rows,
    reader_seed,
    worker_rows,
)
from dataset_rt.metadata import decode_metadata, slice_metadata
from dataset_rt.records import MetadataSnapshot, OriginalMetadata, ReaderRecipe

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.connection import Connection


class ContractSource:
    """Stream enough physical rows to exercise multiple cache identities."""

    def __init__(self, name: str) -> None:
        """Use distinct destinations while sharing an immutable metadata schema."""
        self.name = name

    def __iter__(self) -> Iterator[CacheInput]:
        """Keep fixture generation streaming rather than storing payload collections."""
        for index in range(1000):
            yield CacheInput(f"{self.name}/{index}".encode(), {"index": index})


@pytest.fixture
def dataset(tmp_path: Path) -> CachedDataset:
    """Keep parent-native state alive to exercise warmed-fork reconstruction."""
    runtime = DatasetRuntime(num_workers=1)
    paths: list[Path] = []
    for result in runtime.write_cache(
        [ContractSource("first"), ContractSource("second")],
        tmp_path,
        writer_config=WriterConfig(show_progress=False),
    ):
        match result:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case failure:
                raise AssertionError(failure)
    return runtime.cached_dataset(paths, reader_config=ReaderConfig(seed=7, shuffle=False))


@pytest.fixture
def metadata(dataset: CachedDataset) -> pl.DataFrame:
    """Intentional duplicate physical identities must survive positional partitioning."""
    frame = dataset.get_metadata()
    return pl.concat([frame.slice(1001, 1), frame.slice(0, 1), frame.slice(1001, 1)]).with_columns(
        pl.Series("weight", [3.0, 1.0, 6.0]), pl.lit("kept").alias("extra")
    )


def test_original_recipe_never_accesses_native_state(
    dataset: CachedDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exports are immutable inputs, even if public compatibility attributes change."""
    expected = dataset._reader_recipe()
    dataset.cache_paths.reverse()
    dataset.reader_config = ReaderConfig(seed=99)
    monkeypatch.setattr(dataset, "_inner", object())
    assert dataset._reader_recipe() is expected
    assert isinstance(expected.metadata, OriginalMetadata)
    assert expected.sample_count == 2000
    assert pickle.loads(pickle.dumps(expected)) == expected


@pytest.mark.parametrize("method", ["update_metadata", "set_samples_metadata"])
def test_snapshot_follows_successful_native_update(
    dataset: CachedDataset, metadata: pl.DataFrame, method: str
) -> None:
    """Rejected updates cannot replace the immutable reconstruction snapshot."""
    original = dataset._reader_recipe()
    getattr(dataset, method)(metadata)
    accepted = dataset._reader_recipe()
    match accepted.metadata:
        case MetadataSnapshot(ipc=ipc):
            assert_frame_equal(decode_metadata(ipc), dataset.get_metadata())
        case missing:
            raise AssertionError(missing)
    assert isinstance(original.metadata, OriginalMetadata)
    assert accepted.sample_count == 3
    with pytest.raises(ValueError, match="weight"):
        dataset.update_metadata(metadata.with_columns(pl.lit(-1.0).alias("weight")))
    assert dataset._reader_recipe() is accepted
    dataset.set_epoch_len(11)
    assert dataset._reader_recipe().sample_count == 11
    assert accepted.sample_count == 3
    assert dataset._reader_recipe().metadata is accepted.metadata


@pytest.mark.parametrize("count", [0, 1, 3, 10, 61])
@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("workers", [0, 1, 3, 20])
def test_partition_coverage(count: int, ranks: int, workers: int) -> None:
    """Rank then worker spans cover every position exactly once without padding."""
    position = 0
    for rank in range(ranks):
        replica = ReplicaIdentity(rank=rank, world_size=ranks)
        for worker in range(max(1, workers)):
            span = worker_rows(count, replica, num_workers=workers, worker_id=worker)
            assert span.offset == position
            position += span.length
    assert position == count


def test_metadata_slices_preserve_duplicate_rows(
    dataset: CachedDataset, metadata: pl.DataFrame
) -> None:
    """Columnar slices preserve schema, weights, extras, order, and physical IDs."""
    dataset.update_metadata(metadata)
    match dataset._reader_recipe().metadata:
        case MetadataSnapshot() as snapshot:
            pieces = [
                decode_metadata(slice_metadata(snapshot, partition_rows(3, parts=4, part_id=i)).ipc)
                for i in range(4)
            ]
            assert_frame_equal(pl.concat(pieces), metadata)
        case missing:
            raise AssertionError(missing)


def test_seed_separation_and_optional_randomness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit seeds separate rank/worker pairs; omission draws only when shuffled."""
    values = {derive_seed(7, rank, worker) for rank in range(4) for worker in range(4)}
    assert len(values) == 16
    assert derive_seed(7, 0, 1) != derive_seed(7, 1, 0)
    assert derive_seed(7, 0, 0) == derive_seed(7, 0, 0)
    calls: list[int] = []

    def random_bits(bits: int) -> int:
        """Observe the entropy boundary without probabilistic equality assertions."""
        calls.append(bits)
        return 123

    monkeypatch.setattr("dataset_rt.integrations.loading.secrets.randbits", random_bits)
    assert reader_seed(shuffle=False, seed=None, rank_id=0, worker_id=0) == 0
    assert reader_seed(shuffle=False, seed=99, rank_id=0, worker_id=0) == 0
    assert reader_seed(shuffle=True, seed=7, rank_id=0, worker_id=0) == derive_seed(7, 0, 0)
    assert calls == []
    assert reader_seed(shuffle=True, seed=None, rank_id=2, worker_id=1) == derive_seed(123, 2, 1)
    assert calls == [64]


def test_version_one_seed_vectors() -> None:
    """Freeze seed derivation across refactors rather than silently replaying new streams."""
    assert [derive_seed(7, rank, worker) for rank, worker in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [
        4009395924408594412,
        14809830920954445264,
        6356427855144527917,
        2535241347958328860,
    ]


def test_many_row_partitions_stay_columnar(
    dataset: CachedDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise replicated physical IDs without per-row Python metadata expansion."""
    frame = pl.concat([dataset.get_metadata()] * 8).with_row_index("position")
    dataset.update_metadata(frame)

    def forbidden(*args: object, **kwargs: object) -> None:
        """Make accidental row-object bridges fail in the scalable partition path."""
        raise AssertionError("metadata rows expanded into Python")

    monkeypatch.setattr(pl.DataFrame, "iter_rows", forbidden)
    monkeypatch.setattr(pl.Series, "to_list", forbidden)
    match dataset._reader_recipe().metadata:
        case MetadataSnapshot() as snapshot:
            slices = [
                decode_metadata(
                    slice_metadata(snapshot, partition_rows(frame.height, parts=7, part_id=i)).ipc
                )
                for i in range(7)
            ]
            assert_frame_equal(pl.concat(slices), frame)
        case missing:
            raise AssertionError(missing)


@pytest.mark.parametrize("invalid", [-1, True, 1 << 64])
def test_invalid_seed_domain(invalid: int) -> None:
    """Seed validation rejects bool coercion and unencodable identities."""
    with pytest.raises(ValueError):
        derive_seed(invalid, 0, 0)
    with pytest.raises(ValueError):
        derive_seed(7, invalid, 0)


@pytest.mark.parametrize("rank", [-1, 2, True])
def test_invalid_replica_identity(rank: int) -> None:
    """Captured group identity is validated once before it crosses processes."""
    with pytest.raises(ValidationError):
        ReplicaIdentity(rank=rank, world_size=2)


@pytest.mark.parametrize("initialized", [False, True])
def test_capture_replica_in_training_process(
    monkeypatch: pytest.MonkeyPatch, initialized: bool
) -> None:
    """Use initialized-group identity with single-rank fallback, never worker discovery."""
    import torch.distributed as distributed
    import torch.utils.data as torch_data

    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
    expected = ReplicaIdentity(rank=1, world_size=2) if initialized else ReplicaIdentity()
    assert capture_replica() == expected
    monkeypatch.setattr(torch_data, "get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="before DataLoader workers"):
        capture_replica()


def consume_recipe(recipe: ReaderRecipe, output: Connection) -> None:
    """Test process entry reconstructs fresh native state from accepted inputs only."""
    try:
        runtime = DatasetRuntime(num_workers=1)
        dataset = runtime.cached_dataset(recipe.cache_paths, reader_config=recipe.reader_config)
        match recipe.metadata:
            case MetadataSnapshot() as snapshot:
                dataset._restore_metadata(snapshot)
            case OriginalMetadata():
                pass
        dataset.set_epoch_len(recipe.sample_count)
        output.send(
            (os.getpid(), [(s.cache_id, s.sample_id) for s in dataset], derive_seed(7, 1, 0))
        )
    finally:
        output.close()


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
def test_recipe_reconstruction_in_actual_process(
    dataset: CachedDataset, metadata: pl.DataFrame, context: Literal["fork", "spawn", "forkserver"]
) -> None:
    """Fresh native readers reconstruct metadata in each Mac process context."""
    if context not in mp.get_all_start_methods():
        pytest.skip(f"{context} unavailable")
    dataset.update_metadata(metadata)
    list(dataset)
    receive, send = mp.get_context(context).Pipe(duplex=False)
    match context:
        case "fork":
            child = mp.get_context("fork").Process(
                target=consume_recipe, args=(dataset._reader_recipe(), send)
            )
        case "spawn":
            child = mp.get_context("spawn").Process(
                target=consume_recipe, args=(dataset._reader_recipe(), send)
            )
        case "forkserver":
            child = mp.get_context("forkserver").Process(
                target=consume_recipe, args=(dataset._reader_recipe(), send)
            )
    try:
        child.start()
        send.close()
        assert receive.poll(30), f"{context} child did not report"
        pid, identities, seed = receive.recv()
        assert pid != os.getpid()
        assert identities == [(1, 1), (0, 0), (1, 1)]
        assert seed == derive_seed(7, 1, 0)
        child.join(30)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        if child.is_alive():
            child.kill()
            child.join(5)
        receive.close()
        send.close()


@pytest.mark.parametrize("invalid", [False, True])
def test_checker_rejects_unknown_recipe_fields(tmp_path: Path, invalid: bool) -> None:
    """Use the configured checker to prove a precise recipe field contract."""
    field = "missing_field" if invalid else "sample_count"
    path = tmp_path / "recipe_contract.py"
    path.write_text(
        "from dataset_rt.records import ReaderRecipe\n"
        "def count(recipe: ReaderRecipe) -> int:\n"
        f"    return recipe.{field}\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--config",
            str(Path(__file__).resolve().parents[1] / "pyproject.toml"),
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    if invalid:
        assert result.returncode != 0 and "missing-attribute" in output, output
    else:
        assert result.returncode == 0, output
