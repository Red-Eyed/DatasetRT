"""Frozen configuration passed to Rust-owned cache operations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ModelWrapValidatorHandler,
    model_validator,
)

if TYPE_CHECKING:
    from typing import Self


CompressionAlgo: TypeAlias = Literal["none", "lz4"]
"""Codec names retained by the legacy compression facade."""


class ShardCompression(BaseModel):
    """Deprecated payload-only policy retained for existing writers and saved configs."""

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
        description="Deprecated legacy metadata; this value never controlled compression.",
    )
    """Legacy manifest value, retained without changing compression behavior."""

    @model_validator(mode="wrap")
    @classmethod
    def _validate_legacy(
        cls, value: Self | JsonValue, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Warn on construction while keeping implicit legacy defaults quiet."""
        match value:
            case ShardCompression():
                return handler(value)
            case _:
                if isinstance(value, Mapping) and {"mode", "level", "acceleration"}.intersection(
                    value
                ):
                    raise ValueError(
                        "compression options belong to NoCompression, Lz4Fast, or Lz4HC"
                    )
        result = handler(value)
        warnings.warn(
            "ShardCompression is deprecated; use NoCompression(), Lz4Fast(acceleration=1), "
            "or Lz4HC(level=9). Legacy ratio never controlled compression and remains ignored. "
            "Legacy calls write payload-only v3 caches; new variants write whole-record v4 caches.",
            DeprecationWarning,
            stacklevel=3,
        )
        return result


class NoCompression(BaseModel):
    """Write independent uncompressed records using the v4 record layout."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    algo: Literal["none"] = "none"
    """Persist an explicit uncompressed codec discriminator."""
    mode: Literal["none"] = "none"
    """Select the uncompressed whole-record variant."""


class Lz4Fast(BaseModel):
    """Compress complete records with LZ4; higher acceleration favors write speed."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    algo: Literal["lz4"] = "lz4"
    """Persist the standard LZ4 block codec discriminator."""
    mode: Literal["fast"] = "fast"
    """Select the fast encoder with acceleration control."""
    acceleration: int = Field(default=1, ge=1, le=65_537, strict=True)
    """Accept 1–65537; higher values favor faster compression."""


class Lz4HC(BaseModel):
    """Compress complete records with LZ4 HC; higher levels increase compression effort."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    algo: Literal["lz4"] = "lz4"
    """Persist the standard LZ4 block codec discriminator."""
    mode: Literal["hc"] = "hc"
    """Select the high-compression encoder with effort-level control."""
    level: int = Field(default=9, ge=1, le=12, strict=True)
    """Accept 1–12; higher values increase HC compression effort."""


# Resolve variant annotations before assigning their stable public pickle paths.
NoCompression.__module__ = "dataset_rt.api"
Lz4Fast.__module__ = "dataset_rt.api"
Lz4HC.__module__ = "dataset_rt.api"


RecordCompression: TypeAlias = Annotated[
    NoCompression | Lz4Fast | Lz4HC, Field(discriminator="mode")
]
"""Closed whole-record policies with codec-specific settings."""

# Keep implicit defaults quiet and retain their historical v3 storage behavior.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
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
    """Native cache writing with optional spawned source-process supervision."""

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_processes: int = Field(
        default=0,
        ge=0,
        description="Number of spawned source writers; zero writes in the calling process.",
    )
    """Spawn this many source writers; zero keeps writing in the calling process."""

    process_timeout_seconds: float = Field(
        default=3600.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Deadline in seconds for child startup or a dispatched source, including "
            "deserialization. Timeout stops the pool without retries; ignored in serial mode."
        ),
    )
    """Deadline for child startup or one dispatched source, including deserialization.

    A timeout stops the pool and reports unfinished sources as errors without
    retrying source side effects. Ignored for serial execution.
    """

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

    shard_compression: RecordCompression | ShardCompression = Field(
        default=DEFAULT_SHARD_COMPRESSION,
        description="Whole-record compression variants, or the deprecated payload-only policy.",
    )
    """Typed whole-record policy, or a legacy payload-only policy for compatibility."""

    show_progress: bool = Field(
        default=True,
        description="Whether Rust renders cache write progress with samples/s and MB/s.",
    )
    """Show native sample progress in serial mode or parent source counts in parallel mode."""

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
