# Serialization Boundary

DatasetRT has two serialization layers:

- User/project serialization into a bytes payload.
- DatasetRT cache serialization into immutable cache files.

Python adapters may perform the first layer. For example, a project-specific adapter can encode an image, tensor, token sequence, or structured record into bytes before yielding `CacheInput`.

Rust always owns the second layer:

- Payload bytes are placed into shards by Rust.
- Metadata is typed and written by Rust.
- `metadata.arrow`, `index.bin`, shard files, and `manifest.json` are written by Rust.
- Checksums are calculated and verified by Rust.
- Deserialization from cache shards back into Python `bytes` is performed by Rust.

This boundary keeps DatasetRT framework-independent while still allowing project-specific payload formats. The cache runtime does not need to know whether the bytes represent JPEG, NumPy `.npy`, Arrow IPC, MessagePack, protobuf, or a custom format.

## v0.1 Payload Contract

`CacheInput.data` accepts:

- `bytes`
- `bytearray`
- `memoryview`

DatasetRT v0.1 does not pickle arbitrary Python objects and does not call a Python serializer internally. If a project needs a richer payload format, the adapter must produce bytes before handing the sample to DatasetRT.

## Metadata Contract

Metadata is not an opaque payload. It is part of the cache’s queryable, sample-level identity and sampling surface, so Rust stores it separately as Arrow-compatible primitive columns.
