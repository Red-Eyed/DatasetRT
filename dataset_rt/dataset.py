"""Delegation to authoritative Rust dataset state and native iterators."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, overload

from dataset_rt._dataset_rt import CachedDataset as _RustCachedDataset
from dataset_rt._dataset_rt import DatasetRuntime as _RustDatasetRuntime
from dataset_rt.integrations.torch import to_torch_iterable_dataset
from dataset_rt.metadata import decode_metadata, encode_metadata
from dataset_rt.records import (
    AbsentManifestTarget,
    CachedSample,
    Err,
    ManifestUpdateEntry,
    ManifestUpdateError,
    ManifestUpdateReport,
    MetadataSnapshot,
    Ok,
    OriginalMetadata,
    ReaderRecipe,
    Result,
    SizedTorchIterableDataset,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from multiprocessing.context import BaseContext

    import polars as pl
    from pydantic import PositiveInt
    from torch.utils.data import DataLoader

    from dataset_rt.config import ReaderConfig

T = TypeVar("T")
BatchT = TypeVar("BatchT")


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
    """Cache directories in physical traversal order; only legacy IDs depend on it."""

    reader_config: ReaderConfig
    """Reader configuration used when this dataset was loaded."""

    _inner: _RustCachedDataset
    _recipe: ReaderRecipe

    def update_manifests(
        self, version: Literal[3]
    ) -> Result[ManifestUpdateReport, ManifestUpdateError]:
        """Persist current cache IDs when upgrading v2 manifests to v3.

        Construct legacy datasets in the original cache order used by saved
        metadata tables. This explicit migration preserves those IDs; it does
        not replace them with name hashes. Already-v3 manifests are unchanged.
        Rust preflights all targets, then atomically replaces each manifest.
        Partial failures retain completed entries; retry in the original order.
        Caller owns manifest mutation exclusively. Sample files, active metadata,
        weights and iterator state are unchanged.
        """
        if type(version) is not int or version != 3:
            return Err(
                ManifestUpdateError(
                    AbsentManifestTarget("unsupported migration request"),
                    "unsupported manifest migration target; expected integer 3",
                )
            )
        success, records, failed_path, message = self._inner.update_manifests(version)
        entries: list[ManifestUpdateEntry] = []
        for path, cache_id, status in records:
            match status:
                case "updated" | "unchanged" | "durability_unknown":
                    entries.append(ManifestUpdateEntry(Path(path), cache_id, status))
                case _:
                    raise ValueError(f"invalid native manifest update status: {status}")
        if success:
            return Ok(ManifestUpdateReport(tuple(entries)))
        target = (
            Path(failed_path)
            if failed_path is not None
            else AbsentManifestTarget("request has no filesystem target")
        )
        return Err(ManifestUpdateError(target, message, tuple(entries)))

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
        # Freeze original construction inputs: public path lists can be edited,
        # but a future worker must reconstruct the same physical cache IDs.
        dataset._recipe = ReaderRecipe(
            tuple(dataset.cache_paths), reader_config, len(dataset._inner), OriginalMetadata()
        )
        return dataset

    def _reader_recipe(self) -> ReaderRecipe:
        """Snapshot accepted reconstruction inputs without exporting default metadata.

        The immutable recipe can cross process boundaries. It neither creates a
        new native reader nor advances the existing native sampling cursor.
        """
        return self._recipe

    def _restore_metadata(self, snapshot: MetadataSnapshot) -> None:
        """Validate IPC directly in Rust without entering Polars in forked workers.

        Snapshots are already columnar. Decoding and re-encoding them in a child
        is unnecessary and can wait on Polars thread-pool state inherited by fork.
        """
        self._inner.update_metadata_ipc(snapshot.ipc)
        self._recipe = replace(self._recipe, sample_count=len(self._inner), metadata=snapshot)

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
        self._recipe = replace(self._recipe, sample_count=len(self._inner))

    def to_torch_iterable_dataset(self) -> SizedTorchIterableDataset:
        """Return a sized `torch.utils.data.IterableDataset` view.

        The adapter yields the same `CachedSample` objects as DatasetRT's
        normal iterator and implements `__len__`, so PyTorch consumers can use
        it with `DataLoader` while keeping domain decoding in Python.

        Raises:
            ImportError: If PyTorch is not installed in the active environment.
        """
        return to_torch_iterable_dataset(self)

    @overload
    def to_torch_dataloader(
        self,
        *,
        batch_size: PositiveInt = 1,
        sample_transform_fn: Callable[[CachedSample], T],
        collate_fn: Callable[[list[T]], BatchT],
        shuffle: bool = True,
        seed: int | None = None,
        samples_per_epoch: PositiveInt | None = None,
        worker_partition: Literal["split", "replicate"] = "split",
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        timeout: float = 0,
        native_num_workers: int = 1,
        multiprocessing_context: Literal["fork", "spawn", "forkserver"] | BaseContext | None = None,
        worker_init_fn: Callable[[int], None] | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> DataLoader[CachedSample | T]:
        """Collate lists of transformed samples into a caller-defined result."""
        ...

    @overload
    def to_torch_dataloader(
        self,
        *,
        batch_size: PositiveInt = 1,
        sample_transform_fn: None = None,
        collate_fn: Callable[[list[CachedSample]], BatchT],
        shuffle: bool = True,
        seed: int | None = None,
        samples_per_epoch: PositiveInt | None = None,
        worker_partition: Literal["split", "replicate"] = "split",
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        timeout: float = 0,
        native_num_workers: int = 1,
        multiprocessing_context: Literal["fork", "spawn", "forkserver"] | BaseContext | None = None,
        worker_init_fn: Callable[[int], None] | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> DataLoader[CachedSample | T]:
        """Collate lists of cache records without a sample transform."""
        ...

    @overload
    def to_torch_dataloader(
        self,
        *,
        batch_size: PositiveInt = 1,
        sample_transform_fn: Callable[[CachedSample], T] | None = None,
        collate_fn: None = None,
        shuffle: bool = True,
        seed: int | None = None,
        samples_per_epoch: PositiveInt | None = None,
        worker_partition: Literal["split", "replicate"] = "split",
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        timeout: float = 0,
        native_num_workers: int = 1,
        multiprocessing_context: Literal["fork", "spawn", "forkserver"] | BaseContext | None = None,
        worker_init_fn: Callable[[int], None] | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> DataLoader[CachedSample | T]:
        """Delegate to Torch default collation when no custom callback is supplied."""
        ...

    def to_torch_dataloader(
        self,
        *,
        shuffle: bool = True,
        seed: int | None = None,
        samples_per_epoch: PositiveInt | None = None,
        worker_partition: Literal["split", "replicate"] = "split",
        batch_size: PositiveInt = 1,
        num_workers: int = 0,
        sample_transform_fn: Callable[[CachedSample], T] | None = None,
        collate_fn: (
            Callable[[list[T]], BatchT] | Callable[[list[CachedSample]], BatchT] | None
        ) = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        timeout: float = 0,
        native_num_workers: int = 1,
        multiprocessing_context: Literal["fork", "spawn", "forkserver"] | BaseContext | None = None,
        worker_init_fn: Callable[[int], None] | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> DataLoader[CachedSample | T]:
        """Build a finite PyTorch DataLoader for training or evaluation.

        ``samples_per_epoch`` counts samples for this DataLoader before batching,
        not batches or optimizer steps. None inherits ``len(self)`` at loader
        construction, including any ``set_epoch_len`` override. A positive integer
        overrides that inherited count for this loader. Sequential readers wrap
        around their population when necessary. Later changes to the source
        dataset do not change an existing loader's sample budget.
        The inherited or explicit count must be between 1 and
        min(sys.maxsize, 2**53), keeping PyTorch's batch-length calculation exact.

        ``len(loader)`` is exact: with batch size B and sample budget N it is
        ceil(N / B), or floor(N / B) with ``drop_last=True``. For example, N=10
        and B=3 yields four batches, or three when dropping the incomplete batch.
        ``batch_size`` must be a positive integer no larger than sys.maxsize;
        None is not supported.
        Worker quotas consist of whole batches plus at most one incomplete batch.
        Increasing ``num_workers`` neither multiplies N nor adds dropped tails.
        Workers without an assigned quota yield nothing.

        ``worker_partition="split"`` gives workers disjoint metadata populations.
        With ``shuffle=False``, metadata stays in its existing order and workers
        read their populations sequentially. With ``shuffle=True``, metadata rows
        are shuffled in the parent before splitting, then each worker samples
        its own population by weight with replacement. Randomized splits
        improve the weight mix but do not guarantee the global weighted frequency:
        worker quotas follow batch counts, not partition weight sums.

        ``worker_partition="replicate"`` gives every worker the full population
        while sharing the same total sample budget. With shuffle=True,
        each worker samples by weight with replacement; with shuffle=False, each
        starts at the beginning, so different workers can emit duplicate samples.
        With one consumer (num_workers=0 or 1), both modes use the full population.
        Torch interleaves worker batches; multiworker output need not match global
        metadata order. Weights do not affect sequential reads.

        ``seed`` is accepted only with shuffle=True; otherwise ValueError avoids
        silently ignoring it. An explicit seed reproduces fresh loaders with the
        same worker topology. None selects randomness once during construction.
        Repeated passes reuse native state in the main process or persistent
        workers and continue their draw streams/cursors. Nonpersistent workers
        reconstruct fresh readers each pass. Split populations are fixed for the
        loader's lifetime, including when persistent workers are used.

        Construction snapshots metadata columnarly without loading payloads or
        creating a consuming native reader. Parent edits do not change snapshots.
        Native setup happens on first consumption, or before worker_init_fn in
        each worker. num_workers controls Torch processes; native_num_workers
        controls Rust reader threads per consuming process. Queues remain bounded
        by ReaderConfig.prefetch_size. Callbacks must be picklable for spawn and
        forkserver; errors propagate rather than silently replacing samples.

        Collation always receives a list, including when batch_size=1. The list
        contains transformed samples if sample_transform_fn is supplied, otherwise
        CachedSample values. multiprocessing_context selects how workers start:
        "fork", "spawn", "forkserver", or a multiprocessing BaseContext object.
        None uses PyTorch's platform default. Pinning, timeout, prefetch, and worker
        lifetime follow ordinary PyTorch semantics. PyTorch is optional until called.
        """
        from dataset_rt.integrations.loader import make_dataloader

        return make_dataloader(
            self,
            shuffle=shuffle,
            seed=seed,
            samples_per_epoch=samples_per_epoch,
            worker_partition=worker_partition,
            batch_size=batch_size,
            num_workers=num_workers,
            sample_transform_fn=sample_transform_fn,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            timeout=timeout,
            native_num_workers=native_num_workers,
            multiprocessing_context=multiprocessing_context,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

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
        - `cache_id` is persisted in v3 manifests. Legacy v2 caches use their
          supplied path positions until explicit manifest migration. `sample_id`
          is the physical row inside that cache.
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
        ipc = encode_metadata(metadata)
        # Retain the exact accepted IPC without another export or row expansion.
        # Rejected updates leave both native state and this snapshot unchanged.
        self._restore_metadata(MetadataSnapshot(ipc))
