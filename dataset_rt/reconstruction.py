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
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, PositiveInt, TypeAdapter

from dataset_rt.config import ReaderConfig  # noqa: TC001 - Pydantic resolves this model at runtime.

if TYPE_CHECKING:
    from dataset_rt.dataset import CachedDataset

CONFIG_NAME = "dataset.json"
METADATA_NAME = "metadata.parquet"
DIRECTORY_PATH: TypeAdapter[Path] = TypeAdapter(DirectoryPath)
FILE_PATH: TypeAdapter[Path] = TypeAdapter(FilePath)


class DatasetConfiguration(BaseModel):
    """Validated reconstruction inputs; relative artifact names stay inside the bundle."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal[1] = 1
    cache_paths: tuple[DirectoryPath, ...] = Field(min_length=1)
    reader_config: ReaderConfig
    num_workers: PositiveInt
    epoch_len: PositiveInt
    row_count: PositiveInt
    metadata_file: Literal["metadata.parquet"] = METADATA_NAME


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


def arrow_metadata(ipc: bytes) -> pa.Table:
    """Read accepted columnar IPC without using inherited Polars thread pools."""
    table = pa.ipc.open_file(pa.BufferReader(ipc)).read_all()
    fields = [field.with_type(_portable_type(field.type)) for field in table.schema]
    # Parquet persists view types, but Arrow's row-selection kernels need offset arrays.
    return table.cast(pa.schema(fields, metadata=table.schema.metadata))


def _portable_type(dtype: pa.DataType) -> pa.DataType:
    """Normalize nested view arrays for Parquet round trips and Arrow row selection."""
    if pa.types.is_string_view(dtype):
        return pa.large_string()
    if pa.types.is_binary_view(dtype):
        return pa.large_binary()
    if pa.types.is_large_list(dtype):
        return pa.large_list(dtype.value_field.with_type(_portable_type(dtype.value_type)))
    if pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype):
        size = dtype.list_size if pa.types.is_fixed_size_list(dtype) else -1
        return pa.list_(dtype.value_field.with_type(_portable_type(dtype.value_type)), size)
    if pa.types.is_struct(dtype):
        return pa.struct([field.with_type(_portable_type(field.type)) for field in dtype])
    if pa.types.is_dictionary(dtype):
        return pa.dictionary(dtype.index_type, _portable_type(dtype.value_type), dtype.ordered)
    if pa.types.is_map(dtype):
        return pa.map_(
            _portable_type(dtype.key_type), _portable_type(dtype.item_type), dtype.keys_sorted
        )
    return dtype


def metadata_ipc(table: pa.Table) -> bytes:
    """Encode selected columns for the existing authoritative native validation boundary."""
    output = pa.BufferOutputStream()
    with pa.ipc.new_file(output, table.schema) as writer:
        writer.write_table(table)
    return output.getvalue().to_pybytes()


def load_configuration(path: FilePath) -> DatasetConfiguration:
    """Validate configuration before creating native state or loading payloads."""
    config_path = FILE_PATH.validate_python(path)
    return DatasetConfiguration.model_validate_json(config_path.read_bytes())


def load_metadata(path: FilePath, config: DatasetConfiguration) -> pa.Table:
    """Read Parquet synchronously so warmed-parent fork does not inherit a used pool."""
    config_path = FILE_PATH.validate_python(path)
    metadata_path = FILE_PATH.validate_python(config_path.parent / config.metadata_file)
    with pq.ParquetFile(metadata_path) as reader:
        table = reader.read(use_threads=False)
    if table.num_rows != config.row_count:
        raise ValueError("metadata row count does not match dataset configuration")
    return table


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
    table = arrow_metadata(dataset._inner.metadata_ipc())
    config = DatasetConfiguration(
        cache_paths=tuple(cache.resolve() for cache in dataset._recipe.cache_paths),
        reader_config=dataset._recipe.reader_config,
        num_workers=dataset._runtime_num_workers,
        epoch_len=len(dataset),
        row_count=table.num_rows,
    )
    try:
        pq.write_table(table, pending_metadata, compression="zstd", store_schema=True)
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
