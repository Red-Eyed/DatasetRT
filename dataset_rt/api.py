"""Public Python API for DatasetRT.

The Python layer is intentionally thin. It defines ergonomic types, forwards
cache construction and dataset state to Rust, and converts returned paths or
rows into familiar Python objects.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, TypeAlias, cast

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from dataset_rt._dataset_rt import CachedDataset as _RustCachedDataset
from dataset_rt._dataset_rt import write_cache as _write_cache

MetadataValue: TypeAlias = bool | int | float | str
"""Primitive metadata value accepted by the Rust cache writer."""

Metadata: TypeAlias = Mapping[str, MetadataValue]
"""Mapping from metadata column name to primitive metadata value."""

BytesLike: TypeAlias = bytes | bytearray | memoryview
"""Payload object accepted by `CacheInput`.

Project-specific serialization must happen before values reach DatasetRT.
Rust stores and returns these payloads as bytes without interpreting them.
"""

CompressionAlgo: TypeAlias = Literal["none"]
"""Shard compression algorithms supported by the stable v0.1 writer."""


class ShardCompression(BaseModel):
    """Compression policy requested for payload shards.

    `ratio` is part of the explicit policy object so callers and manifests use
    the same structured shape. DatasetRT v0.1 supports only `algo="none"`,
    which Rust validates with `ratio == 1.0`.
    """

    model_config = ConfigDict(frozen=True)

    algo: CompressionAlgo = "none"
    """Compression algorithm to apply to each shard."""

    ratio: float = Field(default=1.0, gt=0.0)
    """Expected compression ratio for this policy; v0.1 requires `1.0`."""


DEFAULT_SHARD_COMPRESSION = ShardCompression()


class WriterConfig(BaseModel):
    """Configuration for Rust-owned cache writing."""

    model_config = ConfigDict(frozen=True)

    prefetch_size: int = Field(default=64, gt=0)
    """Capacity of Rust's bounded ingestion queue."""

    num_threads: int = Field(default=4, gt=0)
    """Number of Rust serialization worker threads."""

    max_shard_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    """Target maximum shard size before rotating to a new shard."""

    shard_compression: ShardCompression = DEFAULT_SHARD_COMPRESSION
    """Compression policy for payload shards."""

    show_progress: bool = True
    """Show Rust-owned cache write progress with samples/s and MB/s."""


class ReaderConfig(BaseModel):
    """Configuration for Rust-owned dataset reading and sampling."""

    model_config = ConfigDict(frozen=True)

    seed: int
    """Seed used for deterministic epoch sampling."""

    prefetch_size: int = Field(default=64, gt=0)
    """Capacity of Rust's bounded reader result queue."""

    num_workers: int = Field(default=4, gt=0)
    """Number of Rust worker threads used to load samples."""

    shuffle: bool = True
    """Whether each epoch uses deterministic weighted shuffling."""


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
    """Plain source name used by Rust when generating `base_cache_dir/name_hash`."""

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
    writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
) -> list[Path]:
    """Write one or more immutable caches below `path`.

    Args:
        sources: A single `CacheSource` or a list of sources.
        path: Base cache directory. Rust creates one `name_hash` cache
            directory per source below this directory.
        writer_config: Writer behavior owned and validated by Rust.

    Returns:
        Exact cache directories published by Rust, in source order.

    Raises:
        ValueError: If Rust rejects configuration, source names, payloads,
            metadata, or cache paths.
        RuntimeError: If an I/O, manifest, worker, or Python bridge error
            occurs while constructing the cache.
    """
    paths = _write_cache(
        sources,
        str(path),
        writer_config,
        False,
    )
    return [Path(cache_path) for cache_path in paths]


class CachedDataset:
    """Synchronous dataset wrapper over Rust-owned cache state."""

    def __init__(self, paths: Sequence[str | Path], *, reader_config: ReaderConfig) -> None:
        """Load immutable caches and initialize deterministic sampling state."""
        self.cache_paths = [Path(path) for path in paths]
        self.reader_config = reader_config
        self._inner = _RustCachedDataset(
            [str(path) for path in self.cache_paths],
            reader_config.seed,
            reader_config.prefetch_size,
            reader_config.num_workers,
            reader_config.shuffle,
        )

    @classmethod
    def from_cache_sources(
        cls,
        sources: CacheSource | list[CacheSource],
        path: str | Path,
        *,
        reader_config: ReaderConfig,
        writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
    ) -> CachedDataset:
        """Create missing caches from sources, reuse valid existing caches, and load them.

        Rust owns cache path generation, existence checks, validation of existing
        caches, and writes for missing caches. Invalid existing caches raise
        instead of being silently overwritten.
        """
        cache_paths = _write_cache(
            sources,
            str(path),
            writer_config,
            True,
        )
        return cls([Path(cache_path) for cache_path in cache_paths], reader_config=reader_config)

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

    def weight_table(self) -> pl.DataFrame:
        """Return an editable Polars weight table.

        The table contains `cache_id`, `sample_id`, all metadata columns, and
        `weight`. Mutating the returned frame has no effect until it is passed
        to `set_weight_table`.
        """
        weights = pl.DataFrame(self._inner.get_weight_rows())
        # Column presentation is Python ergonomics; Rust remains the source of truth.
        metadata_columns = [
            column
            for column in weights.columns
            if column not in {"cache_id", "sample_id", "weight"}
        ]
        return weights.select(["cache_id", "sample_id", *metadata_columns, "weight"])

    def set_weight_table(self, weights: pl.DataFrame) -> None:
        """Replace Rust-owned sampling weights from a Polars table.

        Rust validates that every physical `(cache_id, sample_id)` appears
        exactly once and that each weight is positive and finite.
        """
        self._inner.set_weight_table(weights)
