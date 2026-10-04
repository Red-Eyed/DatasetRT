"""In-memory columnar IPC boundary; Rust validates authoritative metadata."""

import polars as pl

from dataset_rt.records import MetadataSnapshot, RowSpan


def decode_metadata(data: bytes) -> pl.DataFrame:
    """Decode a native snapshot without expanding its rows into Python objects."""
    return pl.read_ipc(data)


def encode_metadata(metadata: pl.DataFrame) -> bytes:
    """Serialize the whole frame in memory for atomic native validation."""
    buffer = metadata.write_ipc(None)
    if buffer is None:
        raise RuntimeError("Polars did not return an in-memory IPC buffer")
    return buffer.getvalue()


def slice_metadata(snapshot: MetadataSnapshot, span: RowSpan) -> MetadataSnapshot:
    """Slice active row positions columnarly, preserving schema and physical IDs.

    Empty spans are valid planning results; consumers must yield without native
    construction rather than applying an empty table to a native dataset.
    Prepare slices in the training process before fork; children restore the
    resulting IPC directly in Rust rather than accessing inherited Polars pools.
    """
    frame = decode_metadata(snapshot.ipc)
    if span.offset < 0 or span.length < 0 or span.offset + span.length > frame.height:
        raise ValueError("metadata span is outside the active table")
    return MetadataSnapshot(encode_metadata(frame.slice(span.offset, span.length)))
