"""Delegation to authoritative Rust dataset state and native iterators."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl

from dataset_rt._dataset_rt import CachedDataset as _RustCachedDataset
from dataset_rt._dataset_rt import DatasetRuntime as _RustDatasetRuntime
from dataset_rt.config import ReaderConfig
from dataset_rt.integrations.torch import to_torch_iterable_dataset
from dataset_rt.metadata import decode_metadata, encode_metadata
from dataset_rt.records import CachedSample, SizedTorchIterableDataset


class CachedDataset:
    """Synchronous iterable view over Rust-owned dataset state.

    Users do not construct this class directly; use
    `DatasetRuntime.cached_dataset` or `DatasetRuntime.from_cache_sources`.
    Rust owns cache validation, active metadata state, epoch planning, sampling,
    bounded prefetching, and iterator cancellation.
    """

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    cache_paths: list[Path]
    """Immutable cache directories loaded by this dataset, in `cache_id` order."""

    reader_config: ReaderConfig
    """Reader configuration used when this dataset was loaded."""

    _inner: _RustCachedDataset

    def __init__(self) -> None:
        """Reject direct construction because every dataset requires a runtime."""
        raise TypeError("use DatasetRuntime.cached_dataset() to create a CachedDataset")

    @classmethod
    def _load(
        cls,
        runtime: _RustDatasetRuntime,
        paths: Sequence[str | Path],
        reader_config: ReaderConfig,
    ) -> CachedDataset:
        """Construct a dataset bound to an already-created native runtime."""
        dataset = cls.__new__(cls)
        dataset.cache_paths = [Path(path) for path in paths]
        dataset.reader_config = reader_config
        dataset._inner = _RustCachedDataset(
            runtime,
            [str(path) for path in dataset.cache_paths],
            reader_config.seed,
            reader_config.prefetch_size,
            reader_config.shuffle,
            reader_config.validate_cache,
        )
        return dataset

    def __iter__(self) -> Iterator[CachedSample]:
        """Create an iterator from the current active metadata table and epoch length.

        Iterator construction snapshots Rust runtime state. Later `set_epoch_len`
        or `update_metadata` calls affect future iterators, not this iterator.

        With `ReaderConfig.shuffle=True`, Rust creates a deterministic weighted
        multinomial stream over active metadata rows. With `shuffle=False`,
        iteration follows active metadata table order from the current cyclic
        cursor, including duplicate rows.
        """
        for data, metadata, cache_id, sample_id in self._inner:
            yield CachedSample(data, metadata, cache_id, sample_id)

    def __len__(self) -> int:
        """Return the number of samples emitted by each future iterator.

        This value changes after `set_epoch_len` or `update_metadata`. Existing
        iterators keep their own snapshot even if this value changes mid-epoch.
        """
        return len(self._inner)

    def get_item(self, cache_id: int, sample_id: int) -> CachedSample:
        """Read a physical sample without advancing an iterator or sampling cursor.

        IDs refer to immutable cache contents, so filtered or duplicated active
        metadata rows do not change which sample this method returns.

        Raises:
            TypeError: If either ID is not a plain integer.
            IndexError: If either ID is outside the loaded cache range.
        """
        if type(cache_id) is not int or type(sample_id) is not int:
            raise TypeError("cache_id and sample_id must be integers")
        if not 0 <= cache_id <= (1 << 64) - 1:
            raise IndexError(f"cache_id {cache_id} is out of range")
        if not 0 <= sample_id <= (1 << 64) - 1:
            raise IndexError(f"sample_id {sample_id} is out of range")

        data, metadata, resolved_cache_id, resolved_sample_id = self._inner.get_item(
            cache_id, sample_id
        )
        return CachedSample(data, metadata, resolved_cache_id, resolved_sample_id)

    def set_epoch_len(self, epoch_len: int) -> None:
        """Set how many samples each future iterator emits before stopping.

        `epoch_len` must be at least 1. The active metadata table remains the
        sampling population; this method changes only the finite window length.
        Existing iterators keep their snapshot.

        With `ReaderConfig.shuffle=False`, future iterators continue from the
        current cyclic active-row cursor. With `ReaderConfig.shuffle=True`,
        future iterators continue from the current multinomial draw stream.
        `update_metadata` resets the cursor or draw stream and sets `epoch_len`
        to the new table row count. Call `set_epoch_len` afterward to override it.
        """
        if epoch_len < 1:
            raise ValueError("epoch_len must be at least 1")
        self._inner.set_epoch_len(epoch_len)

    def to_torch_iterable_dataset(self) -> SizedTorchIterableDataset:
        """Return a sized `torch.utils.data.IterableDataset` view.

        The adapter yields the same `CachedSample` objects as DatasetRT's
        normal iterator and implements `__len__`, so PyTorch consumers can use
        it with `DataLoader` while keeping domain decoding in Python.

        Raises:
            ImportError: If PyTorch is not installed in the active environment.
        """
        return to_torch_iterable_dataset(self)

    def samples_metadata(self) -> pl.DataFrame:
        """Compatibility alias for `get_metadata`.

        Returns the same active metadata table as `get_metadata`. Prefer
        `get_metadata` in new code.
        """
        return self.get_metadata()

    def get_metadata(self) -> pl.DataFrame:
        """Return the in-memory metadata table that controls future iterators.

        Contract:

        - Returns a Polars `DataFrame` copy; editing it does not mutate the
          dataset until the whole frame is passed to `update_metadata`.
        - Columns are `cache_id`, `sample_id`, every metadata column stored in
          the cache, `weight`, and any extra columns preserved from the previous
          `update_metadata` call.
        - `cache_id` is the cache path position passed to
          `DatasetRuntime.cached_dataset`; `sample_id` is the physical row
          inside that cache.
        - Each row is one active sampling row. Duplicate `(cache_id, sample_id)`
          rows are allowed and represent repeated entries for the same physical
          sample.
        - `len(dataset)` equals the configured epoch length. By default it
          matches the active row count, including duplicate rows.
        - Cache files are not read or rewritten by this method beyond exporting
          the current Rust-owned in-memory table.
        """
        return decode_metadata(self._inner.metadata_ipc())

    def set_samples_metadata(self, metadata: pl.DataFrame) -> None:
        """Compatibility alias for `update_metadata`.

        Applies the same validation and runtime-only update semantics as
        `update_metadata`. Prefer `update_metadata` in new code.
        """
        self.update_metadata(metadata)

    def update_metadata(self, metadata: pl.DataFrame) -> None:
        """Replace the Rust-owned active metadata table for future iterators.

        Input contract:

        - `metadata` must be a Polars `DataFrame`.
        - Required columns are `cache_id`, `sample_id`, every metadata column
          stored in the cache, and `weight`.
        - `cache_id` and `sample_id` must be non-null integer columns that map
          to known physical cache samples.
        - Stored metadata columns must be present with the same Arrow types as
          the immutable cache metadata schema.
        - `weight` must be a non-null numeric column, and every value must be
          positive and finite.
        - Extra columns are allowed and preserved in runtime memory; they are
          returned by the next `get_metadata` call.

        Row semantics:

        - Rows absent from `metadata` are removed from the active sampling
          space and excluded from future iterators.
        - Duplicate `(cache_id, sample_id)` rows are allowed. Each duplicate is
          a separate active row that points to the same immutable physical
          sample, useful for row-duplication balancing or OHEM.
        - `len(dataset)` becomes the new table row count, replacing any previous
          `set_epoch_len` override.
        - With `ReaderConfig(shuffle=False)`, future iterators emit active rows
          exactly in table order, including duplicates.
        - With `ReaderConfig.shuffle=True`, future iterators sample with
          replacement over active rows using each row's `weight`.

        Mutation boundary:

        - Rust validates the full table before replacing runtime state; a
          validation error leaves the previous active table intact.
        - The update is runtime-only and does not rewrite `metadata.arrow`,
          `index.bin`, shards, or manifests.
        - Iterators created before this call keep their existing snapshot;
          iterators created after this call use the new active table.
        """
        self._inner.update_metadata_ipc(encode_metadata(metadata))
