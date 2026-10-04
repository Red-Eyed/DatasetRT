"""Public source protocols, payload records, and typed cache outcomes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeAlias

if TYPE_CHECKING:
    from dataset_rt.dataset import CachedDataset

MetadataValue: TypeAlias = bool | int | float | str
"""Primitive metadata value accepted by the Rust cache writer."""

Metadata: TypeAlias = Mapping[str, MetadataValue]
"""Mapping from metadata column name to primitive metadata value."""

BytesLike: TypeAlias = bytes | bytearray | memoryview
"""Payload object accepted by `CacheInput`.

Project-specific serialization must happen before values reach DatasetRT.
Rust stores and returns these payloads as bytes without interpreting them.
"""


class CacheInput(NamedTuple):
    """One sample yielded by a `CacheSource`.

    `data` is an already-serialized payload. `metadata` is stored separately in
    Arrow-compatible columns and is available for weighting/filtering.
    """

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    data: BytesLike
    """Bytes-like payload to store in DatasetRT shards."""

    metadata: Metadata
    """Primitive metadata columns for this physical sample."""


class CachedSample(NamedTuple):
    """One sample emitted by `CachedDataset` iteration."""

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

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

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    source_name: str
    """Source name used to derive the cache path."""

    path: Path
    """Published cache directory."""


class CacheWriteError(NamedTuple):
    """Failed per-source cache write result."""

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    source_name: str
    """Source name, or a source index label if the name could not be read."""

    message: str
    """Human-readable reason the source was not written."""


CacheWriteResult: TypeAlias = CacheWriteSuccess | CacheWriteError
"""Per-source cache write outcome returned by `DatasetRuntime.write_cache`."""


class CacheSourcesDatasetSuccess(NamedTuple):
    """Dataset creation result when at least one source produced a cache."""

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    dataset: CachedDataset
    """Dataset loaded from all successful cache writes."""

    results: list[CacheWriteResult]
    """Per-source write outcomes, including failures."""


class CacheSourcesDatasetError(NamedTuple):
    """Dataset creation result when no source produced a cache."""

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    results: list[CacheWriteResult]
    """Per-source write outcomes explaining why no dataset was loaded."""

    message: str
    """Human-readable summary of the failed dataset creation."""


CacheSourcesDatasetResult: TypeAlias = CacheSourcesDatasetSuccess | CacheSourcesDatasetError
"""Best-effort result returned by `DatasetRuntime.from_cache_sources`."""


class SizedTorchIterableDataset(Protocol):
    """Sized PyTorch iterable view returned by `to_torch_iterable_dataset`."""

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    def __iter__(self) -> Iterator[CachedSample]:
        """Yield cached samples in DatasetRT iterator order."""
        ...

    def __len__(self) -> int:
        """Return the physical sample count visible to PyTorch."""
        ...


class CacheSource(Protocol):
    """Synchronous source protocol consumed by the Rust cache writer."""

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    name: str
    """Plain source name used by Rust when generating `base_cache_dir/name`."""

    def __iter__(self) -> Iterator[CacheInput]:
        """Yield cache inputs synchronously.

        DatasetRT does not use Python threads or queues. Rust pulls from this
        iterator and owns bounded prefetching, worker threads, and commits.
        """
        ...
