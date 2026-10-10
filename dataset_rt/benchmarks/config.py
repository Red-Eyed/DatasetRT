"""Validated benchmark workload and launcher inputs, separate from dataset configuration."""

from __future__ import annotations

from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Literal

from pydantic import DirectoryPath, Field, model_validator
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict


class BenchmarkConfig(BaseSettings):
    """Control finite work; ranks count training processes and workers are per rank.

    Empty cache_paths generates identical synthetic caches privately in each rank.
    Existing caches are read without modification. Output belongs to global rank
    zero. CUDA is explicit and never silently falls back to CPU.
    """

    model_config = SettingsConfigDict(cli_kebab_case=True, env_prefix="DATASETRT_BENCH_")

    ranks: int = Field(default=1, gt=0)
    num_workers: int = Field(default=0, ge=0)
    device: Literal["cpu", "cuda"] = "cpu"
    worker_context: Literal["spawn", "forkserver", "fork"] = "spawn"
    native_num_workers: int = Field(default=1, gt=0)
    batch_size: int = Field(default=32, gt=0)
    samples_per_epoch: int | None = Field(default=None, gt=0, le=1 << 53)
    shuffle: CliImplicitFlag[bool] = True
    worker_partition: Literal["split", "replicate"] = "split"
    drop_last: CliImplicitFlag[bool] = False
    persistent_workers: CliImplicitFlag[bool] = True
    pin_memory: CliImplicitFlag[bool] = False
    seed: int = Field(default=7, ge=0, lt=1 << 64)
    repeats: int = Field(default=3, gt=0)
    samples_per_cache: int = Field(default=512, gt=0)
    caches: int = Field(default=2, gt=0)
    payload_bytes: int = Field(default=4096, ge=16)
    metadata_bytes: int = Field(default=0, ge=0)
    cache_paths: tuple[DirectoryPath, ...] = ()
    input_features: int = Field(default=256, gt=0)
    model_width: int = Field(default=512, gt=0)
    transform_rounds: int = Field(default=0, ge=0)
    prefetch_size: int = Field(default=16, gt=0)
    prefetch_factor: int = Field(default=2, gt=0)
    timeout_seconds: int = Field(default=600, gt=0)
    worker_timeout: float = Field(default=30, gt=0, allow_inf_nan=False)
    output: Path = Path("benchmark-results.json")
    quiet: CliImplicitFlag[bool] = False

    @model_validator(mode="after")
    def _validate_workload(self) -> BenchmarkConfig:
        """Reject unsupported contexts and unusable tensor shapes before launch."""
        if self.worker_context not in get_all_start_methods():
            raise ValueError("worker_context is unavailable on this platform")
        if self.device == "cuda" and self.num_workers and self.worker_context == "fork":
            raise ValueError("CUDA loading requires spawn or forkserver workers")
        if self.input_features % 32:
            raise ValueError("input_features must be divisible by 32")
        return self


class Topology(BaseSettings):
    """Parse torchrun identity once; loading workers never read launcher identity."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    local_world_size: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_membership(self) -> Topology:
        """Reject malformed launcher environments before device or group setup."""
        if self.rank >= self.world_size or self.local_rank >= self.local_world_size:
            raise ValueError("rank does not belong to its launcher topology")
        return self
