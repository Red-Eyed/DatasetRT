"""Shared Arrow ownership boundary; Rust validates authoritative sampling state."""

from typing import cast

import polars as pl
import pyarrow as pa

from dataset_rt._dataset_rt import CachedDataset as NativeDataset
from dataset_rt.records import MetadataSnapshot, RowSpan


def arrow_metadata(dataset: NativeDataset) -> pa.Table:
    """Import reference-counted native arrays without a whole-table serialized buffer."""
    return pa.RecordBatchReader._import_from_c_capsule(dataset.metadata_arrow()).read_all()


def decode_arrow_metadata(dataset: NativeDataset) -> pl.DataFrame:
    """Share numeric Arrow buffers and preserve chunks instead of eagerly rechunking."""
    return cast("pl.DataFrame", pl.from_arrow(arrow_metadata(dataset), rechunk=False))


def decode_metadata(data: bytes) -> pl.DataFrame:
    """Decode a native snapshot without expanding its rows into Python objects."""
    return pl.read_ipc(data)


def slice_metadata(snapshot: MetadataSnapshot, span: RowSpan) -> MetadataSnapshot:
    """Slice active row positions columnarly, preserving schema and physical IDs.

    Empty spans are valid planning results; consumers must yield without native
    construction rather than applying an empty table to a native dataset.
    Arrow slices share owners and avoid both Polars pools and IPC serialization.
    """
    count = snapshot.table.num_rows if snapshot.selection is None else snapshot.selection.num_rows
    if span.offset < 0 or span.length < 0 or span.offset + span.length > count:
        raise ValueError("metadata span is outside the active table")
    if snapshot.selection is not None:
        return MetadataSnapshot(snapshot.table, snapshot.selection.slice(span.offset, span.length))
    return MetadataSnapshot(snapshot.table.slice(span.offset, span.length))
