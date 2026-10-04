"""In-memory columnar IPC boundary; Rust validates authoritative metadata."""

import polars as pl


def decode_metadata(data: bytes) -> pl.DataFrame:
    """Decode a native snapshot without expanding its rows into Python objects."""
    return pl.read_ipc(data)


def encode_metadata(metadata: pl.DataFrame) -> bytes:
    """Serialize the whole frame in memory for atomic native validation."""
    buffer = metadata.write_ipc(None)
    if buffer is None:
        raise RuntimeError("Polars did not return an in-memory IPC buffer")
    return buffer.getvalue()
