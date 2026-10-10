"""Exercise the public helper against actual Torch and native DatasetRT state."""

from __future__ import annotations

import multiprocessing as mp
import pickle
import subprocess
import sys
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Literal

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from torch.utils.data import DataLoader

from dataset_rt import CachedDataset, CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig
from dataset_rt.config import WriterConfig
from dataset_rt.integrations.loader import LocalReader, PendingReader, ReaderAdapter
from dataset_rt.metadata import decode_metadata
from dataset_rt.records import MetadataSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dataset_rt.records import CachedSample


class LoaderSource:
    """Provide distinguishable physical rows without any tensor serialization."""

    def __init__(self, name: str = "serial-loader", count: int = 10) -> None:
        """Configure independent cache fixtures while keeping generation streaming."""
        self.name = name
        self.count = count

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream a small cache suitable for repeated native window tests."""
        for index in range(self.count):
            yield CacheInput(str(index).encode(), {"index": index})


@dataclass(frozen=True)
class InvalidOptions:
    """Only the integer options exercised by serial boundary rejection tests."""

    num_workers: int = 0
    native_num_workers: int = 1
    timeout: int = 0
    seed: int = 7


@dataclass(frozen=True)
class InvalidWorkerOptions:
    """Worker configurations that must fail before metadata preparation."""

    num_workers: int = 0
    native_num_workers: int = 1
    timeout: float = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    multiprocessing_context: Literal["fork", "spawn", "forkserver", "invalid"] | None = None


@dataclass(frozen=True)
class InvalidBooleanOptions:
    """Typed invalid boolean inputs that must not become truthy adapter modes."""

    shuffle: bool | int = False
    drop_last: bool | int = False
    pin_memory: bool | int = False
    persistent_workers: bool | int = False


@pytest.fixture
def dataset(tmp_path: Path) -> CachedDataset:
    """Keep a parent native dataset alive while the loader constructs its own."""
    runtime = DatasetRuntime(num_workers=1)
    result = runtime.write_cache(
        LoaderSource(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )[0]
    match result:
        case CacheWriteSuccess(path=path):
            return runtime.cached_dataset(
                [path], reader_config=ReaderConfig(seed=17, shuffle=False, prefetch_size=2)
            )
        case failure:
            raise AssertionError(failure)


def sample_identity(samples: list[CachedSample]) -> CachedSample:
    """Expose one record while exercising Torch's mandatory list collation."""
    return samples[0]


@pytest.fixture
def many_cache_dataset(tmp_path: Path) -> CachedDataset:
    """Exercise ten thousand rows across independent caches without row-list setup."""
    runtime = DatasetRuntime(num_workers=2)
    paths = []
    for result in runtime.write_cache(
        [LoaderSource("first", 5000), LoaderSource("second", 5000)],
        tmp_path,
        writer_config=WriterConfig(show_progress=False),
    ):
        match result:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case failure:
                raise AssertionError(failure)
    return runtime.cached_dataset(
        paths, reader_config=ReaderConfig(seed=17, shuffle=False, prefetch_size=2)
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_many_samples_and_caches_stream_without_output_materialization(
    many_cache_dataset: CachedDataset, shuffle: bool
) -> None:
    """Smoke the scalable adapter path with bounded native prefetch and streaming checks."""
    loader = many_cache_dataset.to_torch_dataloader(
        shuffle=shuffle, seed=7 if shuffle else None, batch_size=1, collate_fn=sample_identity
    )
    count = 0
    seen_caches: set[int] = set()
    for sample in islice(loader, 10000):
        assert sample.data == str(sample.sample_id).encode()
        if not shuffle:
            position, sample_id = divmod(count, 5000)
            assert (sample.cache_id, sample.sample_id) == (
                (2_851_758_661_582_890_383, 1_600_601_599_791_221_249)[position],
                sample_id,
            )
        seen_caches.add(sample.cache_id)
        count += 1
    assert count == 10000
    assert seen_caches == {2_851_758_661_582_890_383, 1_600_601_599_791_221_249}


def sample_index(sample: CachedSample) -> int:
    """Decode domain data in an ordinary sample transformation."""
    return int(sample.data)


def first_index(indices: list[int]) -> int:
    """Keep batch-size-one assertions scalar while using ordinary batched collation."""
    return indices[0]


def indices(batch: list[CachedSample]) -> list[int]:
    """Use caller collation to expose actual worker-independent batching."""
    return [int(sample.data) for sample in batch]


def fail_transform(sample: CachedSample) -> int:
    """Exercise terminal callback propagation after successful native delivery."""
    raise ValueError(f"bad sample {sample.sample_id}")


@pytest.mark.parametrize("shuffle", [False, True])
def test_setup_is_lazy_and_reuses_native_state(dataset: CachedDataset, shuffle: bool) -> None:
    """Real readers are created only on consumption and reused on setup/iteration."""
    loader = dataset.to_torch_dataloader(
        shuffle=shuffle, seed=7 if shuffle else None, batch_size=1, collate_fn=sample_identity
    )
    assert isinstance(loader, DataLoader)
    adapter = loader.dataset
    assert isinstance(adapter, ReaderAdapter)
    assert isinstance(adapter._state, PendingReader)
    assert len(loader) == 10
    assert isinstance(adapter._state, PendingReader)
    first = list(islice(loader, 3))
    state = adapter._state
    assert isinstance(state, LocalReader)
    assert state.dataset is not dataset
    adapter.setup()
    assert adapter._state is state
    second = list(islice(loader, 3))
    assert adapter._state is state
    assert first[0].data == str(first[0].metadata["index"]).encode()
    assert len(second) == 3


def test_shuffle_continues_windows_and_partial_iterators(dataset: CachedDataset) -> None:
    """Partial Python iteration never loses pending draws or resets native sampling."""
    dataset.set_epoch_len(3)
    first = dataset.to_torch_dataloader(
        seed=7,
        samples_per_epoch=31,
        batch_size=1,
        sample_transform_fn=sample_index,
        collate_fn=first_index,
    )
    second = dataset.to_torch_dataloader(
        seed=7,
        samples_per_epoch=31,
        batch_size=1,
        sample_transform_fn=sample_index,
        collate_fn=first_index,
    )
    prefix = list(islice(first, 2))
    continuation = list(islice(first, 29))
    assert prefix + continuation == list(islice(second, 31))
    assert set(prefix + continuation) == set(range(10))


def test_sequential_inherits_source_epoch_length(dataset: CachedDataset) -> None:
    """A loader snapshots the source epoch length and continues sequentially."""
    dataset.set_epoch_len(3)
    loader = dataset.to_torch_dataloader(
        shuffle=False, batch_size=1, sample_transform_fn=sample_index, collate_fn=first_index
    )
    assert len(loader) == 3
    dataset.set_epoch_len(8)
    assert len(loader) == 3
    assert list(loader) == [0, 1, 2]
    assert list(loader) == [3, 4, 5]
    assert list(dataset)[0].sample_id == 0


@pytest.mark.parametrize("seed", [7, -1])
def test_sequential_rejects_explicit_seed(dataset: CachedDataset, seed: int) -> None:
    """Unshuffled loading rejects a setting that would otherwise be ignored."""
    with pytest.raises(ValueError, match="seed requires shuffle=True"):
        dataset.to_torch_dataloader(shuffle=False, seed=seed)


@pytest.mark.parametrize(
    "drop_last,expected",
    [(False, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]), (True, [[0, 1, 2, 3], [4, 5, 6, 7]])],
)
def test_native_dataloader_owns_batching(
    dataset: CachedDataset, drop_last: bool, expected: list[list[int]]
) -> None:
    """Caller collation and drop_last behave exactly as in ordinary Torch loading."""
    loader = dataset.to_torch_dataloader(
        shuffle=False, batch_size=4, collate_fn=indices, drop_last=drop_last
    )
    assert list(loader) == expected


def test_metadata_snapshot_preserves_duplicates_and_parent_isolation(
    dataset: CachedDataset,
) -> None:
    """Native reconstruction freezes accepted weights, filtering, order, and duplicates."""
    frame = dataset.get_metadata()
    dataset.update_metadata(pl.concat([frame.slice(8, 1), frame.slice(1, 1), frame.slice(8, 1)]))
    loader = dataset.to_torch_dataloader(shuffle=False, batch_size=1, collate_fn=sample_identity)
    dataset.update_metadata(frame)
    assert [sample.sample_id for sample in loader] == [8, 1, 8]
    assert [sample.sample_id for sample in loader] == [8, 1, 8]


def test_weighted_sampling_keeps_full_active_population(dataset: CachedDataset) -> None:
    """Actual native draws respect changed weights rather than a permutation sampler."""
    frame = dataset.get_metadata().slice(0, 2).with_columns(pl.Series("weight", [1.0, 9.0]))
    dataset.update_metadata(frame)
    loader = dataset.to_torch_dataloader(
        seed=37,
        samples_per_epoch=1000,
        batch_size=1,
        sample_transform_fn=sample_index,
        collate_fn=first_index,
    )
    draws = list(islice(loader, 1000))
    assert set(draws) == {0, 1}
    assert 850 <= draws.count(1) <= 950


def test_pickle_excludes_initialized_native_state(dataset: CachedDataset) -> None:
    """An initialized adapter serializes fresh reconstruction inputs, never handles."""
    loader = dataset.to_torch_dataloader(
        seed=7, batch_size=1, sample_transform_fn=sample_index, collate_fn=first_index
    )
    adapter = loader.dataset
    assert isinstance(adapter, ReaderAdapter)
    before = pickle.dumps(adapter)
    list(islice(loader, 7))
    assert pickle.dumps(adapter) == before
    restored = pickle.loads(before)
    assert isinstance(restored, ReaderAdapter)
    assert isinstance(restored._state, PendingReader)
    assert list(islice(restored, 20)) == list(islice(pickle.loads(before), 20))


def test_omitted_seed_is_fixed_after_setup(dataset: CachedDataset) -> None:
    """Actual entropy is selected once, with a stable reader across iterator calls."""
    loader = dataset.to_torch_dataloader(
        batch_size=1, sample_transform_fn=sample_index, collate_fn=first_index
    )
    list(islice(loader, 1))
    adapter = loader.dataset
    assert isinstance(adapter, ReaderAdapter)
    state = adapter._state
    assert isinstance(state, LocalReader)
    seed = state.dataset.reader_config.seed
    list(islice(loader, 13))
    assert adapter._state is state
    assert state.dataset.reader_config.seed == seed


def test_transform_errors_propagate(dataset: CachedDataset) -> None:
    """No implicit replacement or cross-rank recovery hides domain callback errors."""
    loader = dataset.to_torch_dataloader(batch_size=1, sample_transform_fn=fail_transform)
    with pytest.raises(ValueError, match="bad sample"):
        next(iter(loader))


@pytest.mark.parametrize(
    "options",
    [
        InvalidOptions(num_workers=-1),
        InvalidOptions(native_num_workers=0),
        InvalidOptions(timeout=1),
        InvalidOptions(seed=-1),
    ],
)
def test_invalid_loader_options_fail_before_consumption(
    dataset: CachedDataset, options: InvalidOptions
) -> None:
    """Unsupported serial settings are rejected before creating a consuming reader."""
    with pytest.raises(ValueError):
        dataset.to_torch_dataloader(
            num_workers=options.num_workers,
            native_num_workers=options.native_num_workers,
            timeout=options.timeout,
            seed=options.seed,
        )


def test_package_import_keeps_torch_optional() -> None:
    """A fresh interpreter can import the package without loading Torch integration."""
    result = subprocess.run(
        [sys.executable, "-c", "import dataset_rt, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("samples", [1, 3, 10, 23])
@pytest.mark.parametrize("drop_last", [False, True])
def test_explicit_finite_budget_wraps_sequentially(
    dataset: CachedDataset, samples: int, drop_last: bool
) -> None:
    """Sample budgets remain finite, preserve order, and report actual batch counts."""
    loader = dataset.to_torch_dataloader(
        shuffle=False,
        samples_per_epoch=samples,
        batch_size=3,
        drop_last=drop_last,
        collate_fn=indices,
    )
    expected_count = samples // 3 if drop_last else (samples + 2) // 3
    batches = list(loader)
    assert len(loader) == len(batches) == expected_count
    emitted = samples - samples % 3 if drop_last else samples
    assert [value for batch in batches for value in batch] == [i % 10 for i in range(emitted)]


@pytest.mark.parametrize(
    "samples", [0, -1, True, 1.5, "10", (1 << 53) + 1, sys.maxsize + 1, 1 << 64]
)
def test_invalid_sample_budget(
    dataset: CachedDataset, samples: int | float | str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject coercion and nonpositive budgets before constructing a consuming reader."""

    def unexpected_metadata_access(self: CachedDataset) -> None:
        """Detect preparation reached before rejecting an explicit budget."""
        pytest.fail("invalid sample budget reached metadata preparation")

    monkeypatch.setattr(CachedDataset, "get_metadata", unexpected_metadata_access)
    monkeypatch.setattr(CachedDataset, "_reader_recipe", unexpected_metadata_access)
    with pytest.raises(ValueError, match="samples_per_epoch"):
        dataset.to_torch_dataloader(samples_per_epoch=samples)  # pyrefly: ignore[no-matching-overload]


@pytest.mark.parametrize(
    "options",
    [
        InvalidWorkerOptions(num_workers=-1),
        InvalidWorkerOptions(num_workers=1 << 32),
        InvalidWorkerOptions(native_num_workers=0),
        InvalidWorkerOptions(native_num_workers=sys.maxsize + 1),
        InvalidWorkerOptions(timeout=1),
        InvalidWorkerOptions(num_workers=1, timeout=-1),
        InvalidWorkerOptions(num_workers=1, timeout=float("nan")),
        InvalidWorkerOptions(num_workers=1, timeout=float("inf")),
        InvalidWorkerOptions(prefetch_factor=2),
        InvalidWorkerOptions(num_workers=1, prefetch_factor=0),
        InvalidWorkerOptions(persistent_workers=True),
        InvalidWorkerOptions(multiprocessing_context="spawn"),
        InvalidWorkerOptions(num_workers=1, multiprocessing_context="invalid"),
    ],
)
def test_invalid_workers_fail_before_metadata(
    dataset: CachedDataset, options: InvalidWorkerOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject invalid process options without touching parent metadata or recipes."""

    def unexpected_metadata_access(self: CachedDataset) -> None:
        """Expose expensive preparation reached before option validation."""
        pytest.fail("invalid options reached metadata preparation")

    monkeypatch.setattr(CachedDataset, "get_metadata", unexpected_metadata_access)
    monkeypatch.setattr(CachedDataset, "_reader_recipe", unexpected_metadata_access)
    with pytest.raises(ValueError):
        dataset.to_torch_dataloader(  # pyrefly: ignore[no-matching-overload]
            shuffle=False,
            num_workers=options.num_workers,
            native_num_workers=options.native_num_workers,
            timeout=options.timeout,
            prefetch_factor=options.prefetch_factor,
            persistent_workers=options.persistent_workers,
            multiprocessing_context=options.multiprocessing_context,
        )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 1024])
@pytest.mark.parametrize("drop_last", [False, True])
def test_largest_supported_sample_budget(
    dataset: CachedDataset, batch_size: int, drop_last: bool
) -> None:
    """The maximum accepted length is observable without consuming a huge epoch."""
    budget = min(sys.maxsize, 1 << 53)
    loader = dataset.to_torch_dataloader(
        shuffle=False, samples_per_epoch=budget, batch_size=batch_size, drop_last=drop_last
    )
    expected = budget // batch_size if drop_last else (budget + batch_size - 1) // batch_size
    assert len(loader) == expected
    assert isinstance(loader.dataset, ReaderAdapter)
    assert len(loader.dataset) == budget
    assert isinstance(loader.dataset._state, PendingReader)


@pytest.mark.parametrize(
    "options",
    [
        InvalidBooleanOptions(shuffle=1),
        InvalidBooleanOptions(drop_last=1),
        InvalidBooleanOptions(pin_memory=1),
        InvalidBooleanOptions(persistent_workers=1),
    ],
)
def test_boolean_options_reject_truthy_integers(
    dataset: CachedDataset, options: InvalidBooleanOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not coerce integer flags or reach metadata preparation on invalid input."""

    def unexpected_metadata_access(self: CachedDataset) -> None:
        """Expose preparation reached before validating adapter modes."""
        pytest.fail("invalid boolean option reached metadata preparation")

    monkeypatch.setattr(CachedDataset, "get_metadata", unexpected_metadata_access)
    with pytest.raises(ValueError, match="must be a boolean"):
        dataset.to_torch_dataloader(  # pyrefly: ignore[no-matching-overload]
            num_workers=1,
            shuffle=options.shuffle,
            drop_last=options.drop_last,
            pin_memory=options.pin_memory,
            persistent_workers=options.persistent_workers,
        )


def test_inherited_budget_limit(dataset: CachedDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inherited oversized epochs are rejected before exporting source metadata."""
    dataset.set_epoch_len((1 << 53) + 1)

    def unexpected_metadata_access(self: CachedDataset) -> None:
        """Detect a metadata read made before inherited budget validation."""
        pytest.fail("oversized inherited budget reached metadata preparation")

    monkeypatch.setattr(CachedDataset, "get_metadata", unexpected_metadata_access)
    with pytest.raises(ValueError, match="inherited samples_per_epoch"):
        dataset.to_torch_dataloader(shuffle=False)
    monkeypatch.undo()
    loader = dataset.to_torch_dataloader(shuffle=False, samples_per_epoch=10)
    assert len(loader) == 10


def test_invalid_worker_partition(dataset: CachedDataset) -> None:
    """Unknown partition policies fail explicitly at the public boundary."""
    with pytest.raises(ValueError, match="worker_partition"):
        dataset.to_torch_dataloader(worker_partition="other")  # pyrefly: ignore[no-matching-overload]


def test_prepared_topology_must_match_consumer(dataset: CachedDataset) -> None:
    """A two-worker adapter cannot silently consume both partitions in its parent."""
    loader = dataset.to_torch_dataloader(shuffle=False, num_workers=2)
    assert isinstance(loader.dataset, ReaderAdapter)
    with pytest.raises(RuntimeError, match="worker count differs"):
        loader.dataset.setup()
    assert isinstance(loader.dataset._state, PendingReader)


def test_explicit_base_context(dataset: CachedDataset) -> None:
    """An explicit multiprocessing context starts a real process-local reader."""
    loader = dataset.to_torch_dataloader(
        shuffle=False,
        batch_size=4,
        num_workers=1,
        multiprocessing_context=mp.get_context("spawn"),
        timeout=15,
        collate_fn=indices,
    )
    batches = list(loader)
    assert len(loader) == len(batches) == 3
    assert [value for batch in batches for value in batch] == list(range(10))


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
@pytest.mark.parametrize("partition,heavy_count", [("split", 50), ("replicate", 100)])
def test_partition_policy_exposes_weight_distribution(
    dataset: CachedDataset,
    context: Literal["fork", "spawn", "forkserver"],
    partition: Literal["split", "replicate"],
    heavy_count: int,
) -> None:
    """Random splitting cannot turn equal worker quotas into global weighted draws."""
    dataset.update_metadata(
        dataset.get_metadata().slice(0, 2).with_columns(pl.Series("weight", [1.0, 1e12]))
    )
    loader = dataset.to_torch_dataloader(
        shuffle=True,
        seed=37,
        samples_per_epoch=100,
        worker_partition=partition,
        batch_size=25,
        num_workers=2,
        multiprocessing_context=context,
        timeout=15,
        collate_fn=indices,
    )
    batches = list(loader)
    assert len(loader) == len(batches) == 4
    assert sum(batch.count(1) for batch in batches) == heavy_count


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("partition", ["split", "replicate"])
def test_prepared_worker_populations(
    dataset: CachedDataset, shuffle: bool, partition: Literal["split", "replicate"]
) -> None:
    """Parent-side Polars preserves full records while shuffling only split populations."""
    frame = dataset.get_metadata().with_columns(pl.col("sample_id").alias("extra"))
    dataset.update_metadata(frame)
    loader = dataset.to_torch_dataloader(
        shuffle=shuffle,
        seed=7 if shuffle else None,
        worker_partition=partition,
        num_workers=2,
        batch_size=3,
    )
    adapter = loader.dataset
    assert isinstance(adapter, ReaderAdapter)
    pieces = []
    for recipe in adapter.partitions:
        assert isinstance(recipe.metadata, MetadataSnapshot)
        pieces.append(decode_metadata(recipe.metadata.ipc))
    assert [recipe.sample_count for recipe in adapter.partitions] == [6, 4]
    if partition == "replicate":
        for piece in pieces:
            assert_frame_equal(piece, frame)
        assert adapter.partitions[0].metadata is adapter.partitions[1].metadata
    else:
        prepared = pl.concat(pieces)
        assert_frame_equal(prepared.sort("sample_id"), frame)
        if shuffle:
            assert not prepared.equals(frame)
        else:
            assert_frame_equal(prepared, frame)


@pytest.mark.parametrize("batch_size", [None, 0, -1, True, 1.5, sys.maxsize + 1])
def test_invalid_batch_size(dataset: CachedDataset, batch_size: int | float | None) -> None:
    """Require batching explicitly, without coercing None, booleans, or floats."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        dataset.to_torch_dataloader(batch_size=batch_size)  # pyrefly: ignore[no-matching-overload]


@pytest.mark.parametrize("shuffle", [False, True])
def test_inherited_and_explicit_epoch_budgets(dataset: CachedDataset, shuffle: bool) -> None:
    """Honor source epoch overrides in both modes and let a loader override them."""
    dataset.set_epoch_len(17)
    inherited = dataset.to_torch_dataloader(
        shuffle=shuffle, seed=7 if shuffle else None, batch_size=3, collate_fn=indices
    )
    explicit = dataset.to_torch_dataloader(
        shuffle=shuffle,
        seed=7 if shuffle else None,
        samples_per_epoch=5,
        batch_size=3,
        collate_fn=indices,
    )
    assert len(inherited) == len(list(inherited)) == 6
    assert len(explicit) == len(list(explicit)) == 2
    assert sum(len(batch) for batch in inherited) == 17
    assert sum(len(batch) for batch in explicit) == 5
