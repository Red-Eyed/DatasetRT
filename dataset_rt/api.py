"""Public Python API for DatasetRT.

The Python layer is intentionally thin. It defines ergonomic types, forwards
cache construction and dataset state to Rust, and converts returned outcomes
or rows into familiar Python objects.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
)

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from dataset_rt._dataset_rt import CachedDataset as _RustCachedDataset
from dataset_rt._dataset_rt import write_cache as _write_cache

if TYPE_CHECKING:
    from dataset_rt._dataset_rt import CacheWriteRecord as _RawCacheWriteResult

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _deprecated(message: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate compatibility APIs so callers get a runtime migration warning."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return function(*args, **kwargs)

        return wrapper

    return decorate


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

    algo: CompressionAlgo = "none"
    """Compression algorithm to apply to each shard."""

    ratio: float = Field(default=1.0, gt=0.0)
    """Expected compression ratio for this policy; `none` requires `1.0`."""


DEFAULT_SHARD_COMPRESSION = ShardCompression()


class WriterProfilerConfig(BaseModel):
    """Optional writer profiler output.

    Profiling is disabled by default. When enabled, Rust writes a structured
    JSON summary at `path` after successful writes and handled failures such as
    Ctrl-C.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    """Whether Rust should collect and write writer-stage timing stats."""

    path: Path = Path("dataset_rt_profile.json")
    """JSON summary path used when profiling is enabled."""


DEFAULT_WRITER_PROFILER_CONFIG = WriterProfilerConfig()


class WriterConfig(BaseModel):
    """Configuration for Rust-owned cache writing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefetch_size: int = Field(default=64, gt=0)
    """Maximum buffered writer task/result capacity."""

    max_shard_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    """Target maximum shard size before rotating to a new shard."""

    shard_compression: ShardCompression = DEFAULT_SHARD_COMPRESSION
    """Compression policy for payload shards."""

    show_progress: bool = True
    """Show Rust-owned cache write progress with samples/s and MB/s."""

    validate_cache: bool = False
    """Validate existing caches during writer reuse before returning them."""

    profiler: WriterProfilerConfig = DEFAULT_WRITER_PROFILER_CONFIG
    """Optional writer-stage profiler output."""


class ReaderConfig(BaseModel):
    """Configuration for Rust-owned dataset reading and sampling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    """Seed used for deterministic epoch sampling."""

    prefetch_size: int = Field(default=64, gt=0)
    """Capacity of Rust's bounded reader result queue."""

    shuffle: bool = True
    """Whether each epoch uses deterministic weighted shuffling."""

    validate_cache: bool = False
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
    """Position of the source cache in the `CachedDataset` constructor."""

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
"""Per-source cache write outcome returned by `write_cache`."""


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
"""Best-effort result returned by `CachedDataset.from_cache_sources`."""


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


def write_cache(
    sources: CacheSource | list[CacheSource],
    path: str | Path,
    *,
    num_workers: int,
    writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
) -> list[CacheWriteResult]:
    """Write one or more immutable caches below `path`.

    Args:
        sources: A single `CacheSource` or a list of sources.
        path: Base cache directory. Rust creates one `name` cache
            directory per source below this directory.
        num_workers: Fixed process-wide Rust worker count.
        writer_config: Writer behavior owned and validated by Rust.

    Returns:
        Per-source success or error results in source order.

    Raises:
        ValueError: If Rust rejects batch-level configuration or duplicate
            generated cache paths before writing starts.
    """
    results = _write_cache(
        sources,
        str(path),
        num_workers,
        writer_config,
        False,
    )
    return [_cache_write_result(result) for result in results]


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


class CachedDataset:
    """Synchronous dataset wrapper over Rust-owned cache state."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        num_workers: int,
        reader_config: ReaderConfig,
    ) -> None:
        """Load immutable caches and initialize deterministic sampling state."""
        self.cache_paths = [Path(path) for path in paths]
        self.reader_config = reader_config
        self._inner = _RustCachedDataset(
            [str(path) for path in self.cache_paths],
            reader_config.seed,
            reader_config.prefetch_size,
            num_workers,
            reader_config.shuffle,
            reader_config.validate_cache,
        )

    @classmethod
    def from_cache_sources(
        cls,
        sources: CacheSource | list[CacheSource],
        path: str | Path,
        *,
        num_workers: int,
        reader_config: ReaderConfig,
        writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
    ) -> CacheSourcesDatasetResult:
        """Create missing caches from sources, reuse valid existing caches, and load them.

        Rust owns cache path generation, existence checks, validation of existing
        caches, and writes for missing caches. Invalid existing caches raise
        instead of being silently overwritten. Per-source write failures are
        returned in the result instead of being raised.
        """
        results = [
            _cache_write_result(result)
            for result in _write_cache(
                sources,
                str(path),
                num_workers,
                writer_config,
                True,
            )
        ]
        cache_paths = _successful_cache_paths(results)
        if not cache_paths:
            return _cache_sources_dataset_error(results)
        dataset = cls(cache_paths, num_workers=num_workers, reader_config=reader_config)
        return CacheSourcesDatasetSuccess(dataset, results)

    def __iter__(self) -> Iterator[CachedSample]:
        """Create an iterator from the current Rust-side sampling state.

        With `ReaderConfig.shuffle=True`, Rust snapshots the current weight
        table and creates a deterministic weighted epoch plan. With
        `shuffle=False`, iteration follows physical cache order.
        """
        for data, metadata, cache_id, sample_id in self._inner:
            yield CachedSample(data, metadata, cache_id, sample_id)

    def __len__(self) -> int:
        """Return the physical sample count across all loaded caches."""
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
                return iter(dataset)

            def __len__(self) -> int:
                return len(dataset)

        return DatasetRTTorchIterableDataset()

    def samples_metadata(self) -> pl.DataFrame:
        """Return editable dataset-level sample metadata as a Polars table.

        The table contains `cache_id`, `sample_id`, all metadata columns, and
        `weight`. Mutating the returned frame has no effect until it is passed to
        `set_samples_metadata`.
        """
        return pl.read_ipc(self._inner.samples_metadata_ipc())

    def set_samples_metadata(self, metadata: pl.DataFrame) -> None:
        """Replace Rust-owned sampling weights from a Polars table.

        Rust validates that every physical `(cache_id, sample_id)` appears
        exactly once and that each weight is positive and finite.
        """
        weight_columns = metadata.select(["cache_id", "sample_id", "weight"])
        buffer = weight_columns.write_ipc(None)
        if buffer is None:
            raise RuntimeError("Polars did not return an in-memory IPC buffer")
        self._inner.set_samples_metadata_ipc(buffer.getvalue())

    @_deprecated("CachedDataset.weight_table() is deprecated; use samples_metadata().")
    def weight_table(self) -> pl.DataFrame:
        """Return the editable samples metadata table for compatibility callers."""
        return self.samples_metadata()

    @_deprecated("CachedDataset.set_weight_table() is deprecated; use set_samples_metadata().")
    def set_weight_table(self, weights: pl.DataFrame) -> None:
        """Replace sampling weights from a samples metadata table for compatibility callers."""
        self.set_samples_metadata(weights)
