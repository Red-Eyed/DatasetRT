"""Explicit construction artifacts; no native reader state crosses processes."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    PositiveInt,
    TypeAdapter,
)

from dataset_rt.config import ReaderConfig  # noqa: TC001 - Pydantic resolves this model at runtime.

if TYPE_CHECKING:
    from dataset_rt.dataset import CachedDataset

CONFIG_NAME = "dataset.json"
METADATA_NAME = "metadata.arrow"
ORDER_NAME = "partition-order.npy"
DIRECTORY_PATH: TypeAdapter[Path] = TypeAdapter(DirectoryPath)
FILE_PATH: TypeAdapter[Path] = TypeAdapter(FilePath)


class DatasetConfiguration(BaseModel):
    """Validated reconstruction inputs; relative artifact names stay inside the bundle."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal[2] = 2
    cache_paths: tuple[DirectoryPath, ...] = Field(min_length=1)
    reader_config: ReaderConfig
    num_workers: PositiveInt
    epoch_len: PositiveInt
    row_count: PositiveInt
    metadata_file: Literal["metadata.arrow"] = METADATA_NAME


@dataclass
class ConstructionArtifacts:
    """Parent-owned export lifetime; forked children cannot delete parent artifacts."""

    directory: Path
    pid: int
    temporary: bool

    @classmethod
    def create(cls, work_dir: DirectoryPath | None) -> ConstructionArtifacts:
        """Isolate each loader export without overwriting another loader's snapshot."""
        root = None if work_dir is None else DIRECTORY_PATH.validate_python(work_dir)
        return cls(Path(mkdtemp(prefix="dataset-rt-", dir=root)), os.getpid(), work_dir is None)

    def discard(self) -> None:
        """Remove only this parent's newly allocated artifact directory."""
        if self.pid == os.getpid():
            shutil.rmtree(self.directory, ignore_errors=True)

    def __del__(self) -> None:
        """Keep caller-selected output; release managed output after its last parent owner."""
        if self.temporary:
            self.discard()


def metadata_ipc(table: pa.Table) -> bytes:
    """Serialize an explicitly requested legacy in-memory recipe, outside worker setup."""
    output = pa.BufferOutputStream()
    with pa.ipc.new_file(output, table.schema) as writer:
        writer.write_table(table)
    return output.getvalue().to_pybytes()


def load_configuration(path: FilePath) -> DatasetConfiguration:
    """Validate configuration before creating native state or loading payloads."""
    config_path = FILE_PATH.validate_python(path)
    return DatasetConfiguration.model_validate_json(config_path.read_bytes())


def load_metadata(path: FilePath, config: DatasetConfiguration) -> pa.Table:
    """Map immutable Arrow buffers shared across processes without decoding a private table."""
    config_path = FILE_PATH.validate_python(path)
    metadata_path = FILE_PATH.validate_python(config_path.parent / config.metadata_file)
    with pa.memory_map(str(metadata_path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    if table.num_rows != config.row_count:
        raise ValueError("metadata row count does not match dataset configuration")
    return table


def prepare_row_order(path: FilePath, count: int, seed: int) -> None:
    """Build one shared deterministic permutation with bounded initialization buffers."""
    order_path = path.parent / ORDER_NAME
    order = np.lib.format.open_memmap(order_path, mode="w+", dtype=np.uint64, shape=(count,))
    for start in range(0, count, 65_536):
        end = min(count, start + 65_536)
        order[start:end] = np.arange(start, end, dtype=np.uint64)
    np.random.default_rng(seed).shuffle(order)
    order.flush()


def population_selection(
    path: FilePath, count: int, *, offset: int, length: int, partition_seed: int | None
) -> pa.Table | None:
    """Select shared row positions; native readers retain full metadata buffers without a take."""
    if offset < 0 or length <= 0:
        raise ValueError("population selection requires a nonnegative offset and positive length")
    if partition_seed is None and offset == 0 and length == count:
        return None
    if partition_seed is None:
        indices = np.arange(length, dtype=np.uint64)
        indices += offset % count
        indices %= count
    else:
        order = np.load(path.parent / ORDER_NAME, mmap_mode="r", allow_pickle=False)
        if order.dtype != np.uint64 or order.shape != (count,):
            raise ValueError("partition order does not match metadata row count")
        start = offset % count
        if start + length <= count:
            indices = order[start : start + length]
        else:
            positions = np.arange(length, dtype=np.uint64)
            positions += start
            positions %= count
            indices = order[positions]
    return pa.table({"row_index": pa.array(indices, type=pa.uint64())})


def select_population(
    table: pa.Table, *, offset: int, length: int, partition_seed: int | None
) -> pa.Table:
    """Reproduce common row ordering, then select a cyclic span without Python row objects."""
    if length < 0 or offset < 0:
        raise ValueError("population selection requires nonnegative offset and length")
    if length == 0:
        return table.slice(0, 0)
    count = table.num_rows
    if partition_seed is None and offset + length <= count:
        return table.slice(offset, length)
    indices = np.arange(length, dtype=np.int64)
    indices += offset % count
    indices %= count
    if partition_seed is not None:
        order = np.random.default_rng(partition_seed).permutation(count)
        indices = order[indices]
    return table.take(pa.array(indices))


def _sync_file(path: Path) -> None:
    """Finish artifact writes before the referencing configuration is published."""
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def dump_configuration(dataset: CachedDataset, path: DirectoryPath) -> FilePath:
    """Publish config last in an exclusively owned directory; never overwrite an export."""
    directory = DIRECTORY_PATH.validate_python(path).resolve()
    config_path = directory / CONFIG_NAME
    metadata_path = directory / METADATA_NAME
    pending_config = directory / f"{CONFIG_NAME}.partial"
    pending_metadata = directory / f"{METADATA_NAME}.partial"
    targets = (config_path, metadata_path, pending_config, pending_metadata)
    if any(target.exists() for target in targets):
        raise FileExistsError("dataset export already exists; choose an unused directory")
    try:
        row_count = dataset._inner.export_metadata_arrow(str(pending_metadata))
        config = DatasetConfiguration(
            cache_paths=tuple(cache.resolve() for cache in dataset._recipe.cache_paths),
            reader_config=dataset._recipe.reader_config,
            num_workers=dataset._runtime_num_workers,
            epoch_len=len(dataset),
            row_count=row_count,
        )
        _sync_file(pending_metadata)
        pending_config.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        _sync_file(pending_config)
        pending_metadata.replace(metadata_path)
        pending_config.replace(config_path)
    except BaseException:
        for target in targets:
            target.unlink(missing_ok=True)
        raise
    return FILE_PATH.validate_python(config_path)
