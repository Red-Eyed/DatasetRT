"""Exercise the public helper against actual Torch and native DatasetRT state."""

from __future__ import annotations

import pickle
import subprocess
import sys
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

import polars as pl
import pytest
from torch.utils.data import DataLoader

from dataset_rt import CachedDataset, CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig
from dataset_rt.config import WriterConfig
from dataset_rt.integrations.loader import LocalReader, PendingReader, ReaderAdapter

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


def sample_identity(sample: CachedSample) -> CachedSample:
    """Keep unbatched samples intact across Torch's collation boundary."""
    return sample


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
        shuffle=shuffle, seed=7, batch_size=None, collate_fn=sample_identity
    )
    count = 0
    seen_caches: set[int] = set()
    for sample in islice(loader, 10000):
        assert sample.data == str(sample.sample_id).encode()
        if not shuffle:
            assert (sample.cache_id, sample.sample_id) == divmod(count, 5000)
        seen_caches.add(sample.cache_id)
        count += 1
    assert count == 10000
    assert seen_caches == {0, 1}


def sample_index(sample: CachedSample) -> int:
    """Decode domain data in an ordinary sample transformation."""
    return int(sample.data)


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
        shuffle=shuffle, seed=7, batch_size=None, collate_fn=sample_identity
    )
    assert isinstance(loader, DataLoader)
    adapter = loader.dataset
    assert isinstance(adapter, ReaderAdapter)
    assert isinstance(adapter._state, PendingReader)
    if shuffle:
        with pytest.raises(TypeError):
            len(loader)
    else:
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
    first = dataset.to_torch_dataloader(seed=7, batch_size=None, sample_transform_fn=sample_index)
    second = dataset.to_torch_dataloader(seed=7, batch_size=None, sample_transform_fn=sample_index)
    prefix = list(islice(first, 2))
    continuation = list(islice(first, 29))
    assert prefix + continuation == list(islice(second, 31))
    assert set(prefix + continuation) == set(range(10))


@pytest.mark.parametrize("seed", [None, 7, -1])
def test_sequential_repeats_full_partition_and_ignores_seed(
    dataset: CachedDataset, seed: int | None
) -> None:
    """Finite validation uses active rows even when source window length differs."""
    dataset.set_epoch_len(3)
    loader = dataset.to_torch_dataloader(
        shuffle=False, seed=seed, batch_size=None, sample_transform_fn=sample_index
    )
    assert len(loader) == 10
    assert list(loader) == list(range(10))
    assert list(loader) == list(range(10))
    assert list(dataset)[0].sample_id == 0


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
    loader = dataset.to_torch_dataloader(shuffle=False, batch_size=None, collate_fn=sample_identity)
    dataset.update_metadata(frame)
    assert [sample.sample_id for sample in loader] == [8, 1, 8]
    assert [sample.sample_id for sample in loader] == [8, 1, 8]


def test_weighted_sampling_keeps_full_active_population(dataset: CachedDataset) -> None:
    """Actual native draws respect changed weights rather than a permutation sampler."""
    frame = dataset.get_metadata().slice(0, 2).with_columns(pl.Series("weight", [1.0, 9.0]))
    dataset.update_metadata(frame)
    loader = dataset.to_torch_dataloader(seed=37, batch_size=None, sample_transform_fn=sample_index)
    draws = list(islice(loader, 1000))
    assert set(draws) == {0, 1}
    assert 850 <= draws.count(1) <= 950


def test_pickle_excludes_initialized_native_state(dataset: CachedDataset) -> None:
    """An initialized adapter serializes fresh reconstruction inputs, never handles."""
    loader = dataset.to_torch_dataloader(seed=7, batch_size=None, sample_transform_fn=sample_index)
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
    loader = dataset.to_torch_dataloader(batch_size=None, sample_transform_fn=sample_index)
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
    loader = dataset.to_torch_dataloader(batch_size=None, sample_transform_fn=fail_transform)
    with pytest.raises(ValueError, match="bad sample"):
        next(iter(loader))


@pytest.mark.parametrize(
    "options",
    [
        InvalidOptions(num_workers=1),
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
