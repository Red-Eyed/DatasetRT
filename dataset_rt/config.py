"""Frozen configuration passed to Rust-owned cache operations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

CompressionAlgo: TypeAlias = Literal["none", "lz4"]
"""Shard compression algorithms supported by the stable v0.1 writer."""


class ShardCompression(BaseModel):
    """Compression policy requested for payload shards.

    `ratio` is part of the explicit policy object so callers and manifests use
    the same structured shape. `ratio` is advisory metadata for compressed
    algorithms; Rust validates `algo="none"` with `ratio == 1.0`.
    """

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

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

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

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

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

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

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

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
