"""Public Python API for DatasetRT.

The Python layer is intentionally thin. It defines ergonomic types, forwards
cache construction and dataset state to Rust, and converts returned outcomes
or rows into familiar Python objects.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    cast,
)

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from dataset_rt._dataset_rt import CachedDataset as _RustCachedDataset
from dataset_rt._dataset_rt import DatasetRuntime as _RustDatasetRuntime
from dataset_rt._dataset_rt import write_cache as _write_cache

if TYPE_CHECKING:
    from dataset_rt._dataset_rt import CacheWriteRecord as _RawCacheWriteResult

MetadataValue: TypeAlias = bool | int | float | str
"""Primitive metadata value accepted by the Rust cache writer."""

Metadata: TypeAlias = Mapping[str, MetadataValue]
"""Mapping from metadata column name to primitive metadata value."""

BytesLike: TypeAlias = bytes | bytearray | memoryview
"""Payload object accepted by `CacheInput`.

Project-specific serialization must happen before values reach DatasetRT.
Rust stores and returns these payloads as bytes without interpreting them.
"""

CompressionAlgo: TypeAlias = Literal["none", "lz4"]
"""Shard compression algorithms supported by the stable v0.1 writer."""


class ShardCompression(BaseModel):
    """Compression policy requested for payload shards.

    `ratio` is part of the explicit policy object so callers and manifests use
    the same structured shape. `ratio` is advisory metadata for compressed
    algorithms; Rust validates `algo="none"` with `ratio == 1.0`.
    """

    model_config = ConfigDict(frozen=True)

    algo: CompressionAlgo = Field(
        default="none",
        description="Compression algorithm to apply independently to each payload record.",
    )
    """Compression algorithm to apply to each shard."""

    ratio: float = Field(
        default=1.0,
        gt=0.0,
        description="Expected compression ratio; `algo='none'` requires exactly `1.0`.",
    )
    """Expected compression ratio for this policy; `none` requires `1.0`."""


DEFAULT_SHARD_COMPRESSION = ShardCompression()


class WriterProfilerConfig(BaseModel):
    """Optional writer profiler output.

    Profiling is disabled by default. When enabled, Rust writes a structured
    JSON summary at `path` after successful writes and handled failures such as
    Ctrl-C.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="Whether Rust collects and writes writer-stage timing statistics.",
    )
    """Whether Rust should collect and write writer-stage timing stats."""

    path: Path = Field(
        default=Path("dataset_rt_profile.json"),
        description="JSON summary path used when profiling is enabled.",
    )
    """JSON summary path used when profiling is enabled."""


DEFAULT_WRITER_PROFILER_CONFIG = WriterProfilerConfig()


class WriterConfig(BaseModel):
    """Configuration for Rust-owned cache writing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefetch_size: int = Field(
        default=64,
        gt=0,
        description="Maximum number of writer tasks/results buffered by Rust.",
    )
    """Maximum buffered writer task/result capacity."""

    max_shard_bytes: int = Field(
        default=64 * 1024 * 1024,
        gt=0,
        description="Target shard byte size before Rust rotates to a new shard.",
    )
    """Target maximum shard size before rotating to a new shard."""

    shard_compression: ShardCompression = Field(
        default=DEFAULT_SHARD_COMPRESSION,
        description="Per-record payload compression policy for new shards.",
    )
    """Compression policy for payload shards."""

    show_progress: bool = Field(
        default=True,
        description="Whether Rust renders cache write progress with samples/s and MB/s.",
    )
    """Show Rust-owned cache write progress with samples/s and MB/s."""

    validate_cache: bool = Field(
        default=False,
        description="Whether reused existing caches are checksum-validated before loading.",
    )
    """Validate existing caches during writer reuse before returning them."""

    profiler: WriterProfilerConfig = Field(
        default=DEFAULT_WRITER_PROFILER_CONFIG,
        description="Optional JSON writer-stage profiler configuration.",
    )
    """Optional writer-stage profiler output."""


class ReaderConfig(BaseModel):
    """Configuration for Rust-owned dataset reading and sampling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(description="Seed used for deterministic shuffled epoch planning.")
    """Seed used for deterministic epoch sampling."""

    prefetch_size: int = Field(
        default=64,
        gt=0,
        description="Capacity of Rust's bounded reader result queue.",
    )
    """Capacity of Rust's bounded reader result queue."""

    shuffle: bool = Field(
        default=True,
        description="Whether future iterators use deterministic weighted sampling.",
    )
    """Whether each epoch uses deterministic weighted shuffling."""

    validate_cache: bool = Field(
        default=False,
        description="Whether cache checksums are verified while loading metadata and indexes.",
    )
    """Verify cache checksums while loading dataset metadata and indexes."""


DEFAULT_WRITER_CONFIG = WriterConfig()


class CacheInput(NamedTuple):
    """One sample yielded by a `CacheSource`.

    `data` is an already-serialized payload. `metadata` is stored separately in
    Arrow-compatible columns and is available for weighting/filtering.
    """

    data: BytesLike
    """Bytes-like payload to store in DatasetRT shards."""

    metadata: Metadata
    """Primitive metadata columns for this physical sample."""


class CachedSample(NamedTuple):
    """One sample emitted by `CachedDataset` iteration."""

    data: bytes
    """Payload bytes loaded from the immutable cache."""

    metadata: dict[str, MetadataValue]
    """Metadata row associated with this physical sample."""

    cache_id: int
    """Position of the source cache passed to `DatasetRuntime.cached_dataset`."""

    sample_id: int
    """Physical sample row within the source cache."""


class CacheWriteSuccess(NamedTuple):
    """Successful per-source cache write result."""

    source_name: str
    """Source name used to derive the cache path."""

    path: Path
    """Published cache directory."""


class CacheWriteError(NamedTuple):
    """Failed per-source cache write result."""

    source_name: str
    """Source name, or a source index label if the name could not be read."""

    message: str
    """Human-readable reason the source was not written."""


CacheWriteResult: TypeAlias = CacheWriteSuccess | CacheWriteError
"""Per-source cache write outcome returned by `DatasetRuntime.write_cache`."""


class CacheSourcesDatasetSuccess(NamedTuple):
    """Dataset creation result when at least one source produced a cache."""

    dataset: CachedDataset
    """Dataset loaded from all successful cache writes."""

    results: list[CacheWriteResult]
    """Per-source write outcomes, including failures."""


class CacheSourcesDatasetError(NamedTuple):
    """Dataset creation result when no source produced a cache."""

    results: list[CacheWriteResult]
    """Per-source write outcomes explaining why no dataset was loaded."""

    message: str
    """Human-readable summary of the failed dataset creation."""


CacheSourcesDatasetResult: TypeAlias = CacheSourcesDatasetSuccess | CacheSourcesDatasetError
"""Best-effort result returned by `DatasetRuntime.from_cache_sources`."""


class SizedTorchIterableDataset(Protocol):
    """Sized PyTorch iterable view returned by `to_torch_iterable_dataset`."""

    def __iter__(self) -> Iterator[CachedSample]:
        """Yield cached samples in DatasetRT iterator order."""
        ...

    def __len__(self) -> int:
        """Return the physical sample count visible to PyTorch."""
        ...


class CacheSource(Protocol):
    """Synchronous source protocol consumed by the Rust cache writer."""

    name: str
    """Plain source name used by Rust when generating `base_cache_dir/name`."""

    def __iter__(self) -> Iterator[CacheInput]:
        """Yield cache inputs synchronously.

        DatasetRT does not use Python threads or queues. Rust pulls from this
        iterator and owns bounded prefetching, worker threads, and commits.
        """
        ...


def _cache_write_result(result: _RawCacheWriteResult) -> CacheWriteResult:
    status, source_name, detail = result
    match status:
        case "success":
            return CacheWriteSuccess(source_name, Path(detail))
        case "error":
            return CacheWriteError(source_name, detail)
        case _:
            raise ValueError(f"unknown cache write result status: {status}")


def _successful_cache_paths(results: Sequence[CacheWriteResult]) -> list[Path]:
    paths = []
    for result in results:
        match result:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case CacheWriteError():
                continue
    return paths


def _cache_sources_dataset_error(results: list[CacheWriteResult]) -> CacheSourcesDatasetError:
    return CacheSourcesDatasetError(results, _format_cache_sources_dataset_error(results))


def _format_cache_sources_dataset_error(results: Sequence[CacheWriteResult]) -> str:
    messages = []
    for result in results:
        match result:
            case CacheWriteError(source_name=source_name, message=message):
                messages.append(f"{source_name}: {message}")
            case CacheWriteSuccess():
                continue
    if not messages:
        return "no cache sources were provided"
    details = "; ".join(messages)
    return f"no caches were written: {details}"


class DatasetRuntime:
    """Owner of the fixed Rust worker pool used by DatasetRT operations.

    Create one runtime per process or training job, then call its methods to
    write caches, load datasets, and iterate samples. The worker count is chosen
    once at construction and reused; per-operation APIs do not resize the pool.
    """

    __slots__ = ("_inner", "_num_workers")

    def __init__(self, *, num_workers: int) -> None:
        """Create exactly `num_workers` reusable Rust worker threads.

        `num_workers` must be positive. Rust validates the value before creating
        the native pool. The pool is owned by this runtime and kept alive by any
        datasets loaded through it.
        """
        self._num_workers = num_workers
        self._inner = _RustDatasetRuntime(num_workers)

    @property
    def num_workers(self) -> int:
        """Return the fixed worker count selected when this runtime was created."""
        return self._num_workers

    def write_cache(
        self,
        sources: CacheSource | list[CacheSource],
        path: str | Path,
        *,
        writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
    ) -> list[CacheWriteResult]:
        """Write one immutable cache per source under `path`.

        `sources` may be one `CacheSource` or a list of sources. Each source
        must expose a plain `name` and yield `CacheInput` values. Rust writes
        source `name` under `path / name`, validates a stable metadata schema,
        stores payload bytes in shards, writes `metadata.arrow` and `index.bin`,
        then publishes a manifest only after the cache is complete.

        Returns one `CacheWriteSuccess` or `CacheWriteError` per source in input
        order. Per-source failures are reported as values instead of exceptions
        when Rust can handle them cleanly.
        """
        results = _write_cache(
            self._inner,
            sources,
            str(path),
            writer_config,
            False,
        )
        return [_cache_write_result(result) for result in results]

    def cached_dataset(
        self,
        paths: Sequence[str | Path],
        *,
        reader_config: ReaderConfig,
    ) -> CachedDataset:
        """Load immutable cache directories into a `CachedDataset`.

        `paths` order defines stable `cache_id` values for the dataset.
        DatasetRT validates manifests, schemas, metadata/index shape, and shard
        lengths while loading. Expensive checksum hashing is controlled by
        `reader_config.validate_cache`.

        The returned dataset keeps this runtime's Rust worker pool alive and
        uses it for every future iterator.
        """
        return CachedDataset._load(self._inner, paths, reader_config)

    def from_cache_sources(
        self,
        sources: CacheSource | list[CacheSource],
        path: str | Path,
        *,
        reader_config: ReaderConfig,
        writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
    ) -> CacheSourcesDatasetResult:
        """Create or reuse source caches, then load all successful cache paths.

        Existing complete caches under `path / source.name` are reused. Missing
        caches are written with `writer_config`. The method then loads every
        successful cache with `reader_config`.

        Returns `CacheSourcesDatasetSuccess` when at least one cache is loaded;
        returns `CacheSourcesDatasetError` when every source failed or no source
        was provided. The result always includes per-source write outcomes so
        callers can audit partial success.
        """
        results = [
            _cache_write_result(result)
            for result in _write_cache(
                self._inner,
                sources,
                str(path),
                writer_config,
                True,
            )
        ]
        cache_paths = _successful_cache_paths(results)
        if not cache_paths:
            return _cache_sources_dataset_error(results)
        dataset = self.cached_dataset(cache_paths, reader_config=reader_config)
        return CacheSourcesDatasetSuccess(dataset, results)


class CachedDataset:
    """Synchronous iterable view over Rust-owned dataset state.

    Users do not construct this class directly; use
    `DatasetRuntime.cached_dataset` or `DatasetRuntime.from_cache_sources`.
    Rust owns cache validation, active metadata state, epoch planning, sampling,
    bounded prefetching, and iterator cancellation.
    """

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
        """Create an iterator from the current active metadata table.

        Iterator construction snapshots Rust runtime state. Later
        `update_metadata` calls affect future iterators, not this iterator.

        With `ReaderConfig.shuffle=True`, Rust creates a deterministic weighted
        multinomial sampler over active metadata rows. With `shuffle=False`,
        iteration follows active metadata table order, including duplicate rows.
        """
        for data, metadata, cache_id, sample_id in self._inner:
            yield CachedSample(data, metadata, cache_id, sample_id)

    def __len__(self) -> int:
        """Return the active metadata row count used by future iterators.

        This value changes after `update_metadata`. Filtered-out rows reduce the
        length, and duplicate rows increase it. Existing iterators keep their
        own snapshot even if this value changes mid-epoch.
        """
        return len(self._inner)

    def to_torch_iterable_dataset(self) -> SizedTorchIterableDataset:
        """Return a sized `torch.utils.data.IterableDataset` view.

        The adapter yields the same `CachedSample` objects as DatasetRT's
        normal iterator and implements `__len__`, so PyTorch consumers can use
        it with `DataLoader` while keeping domain decoding in Python.

        Raises:
            ImportError: If PyTorch is not installed in the active environment.
        """
        try:
            torch_data = import_module("torch.utils.data")
        except ImportError as error:
            raise ImportError(
                "CachedDataset.to_torch_iterable_dataset requires PyTorch to be installed"
            ) from error

        iterable_dataset = cast(type[object], torch_data.IterableDataset)
        dataset = self

        class DatasetRTTorchIterableDataset(iterable_dataset):
            """Sized PyTorch iterable view over a `CachedDataset`."""

            def __iter__(self) -> Iterator[CachedSample]:
                if torch_data.get_worker_info() is not None:
                    raise RuntimeError(
                        "Use Dataloader with num workers = 0. "
                        "Use DatasetRuntime(num_workers=N) for parallel cache reads."
                    )
                return iter(dataset)

            def __len__(self) -> int:
                return len(dataset)

        return DatasetRTTorchIterableDataset()

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
        - `len(dataset)` equals the number of active rows returned here,
          including duplicate rows.
        - Cache files are not read or rewritten by this method beyond exporting
          the current Rust-owned in-memory table.
        """
        return pl.read_ipc(self._inner.metadata_ipc())

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
        - `len(dataset)` becomes `metadata.height`, including duplicate rows.
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
        buffer = metadata.write_ipc(None)
        if buffer is None:
            raise RuntimeError("Polars did not return an in-memory IPC buffer")
        self._inner.update_metadata_ipc(buffer.getvalue())
