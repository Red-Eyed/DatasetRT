# Storage Format

DatasetRT writes immutable cache directories.

```text
cache/
    manifest.json
    metadata.arrow
    index.bin
    shards/
        000000.bin
        000001.bin
```

## Manifest

`manifest.json` is the publication marker for a completed cache. Readers validate it before opening any payload data.

The manifest records:

- Format version.
- Source name.
- Sample count.
- Metadata schema.
- Index checksum.
- Metadata checksum.
- Shard names, byte lengths, compression metadata, and SHA-256 checksums.

If `manifest.json` is missing, the cache is incomplete and must not be read.

## Metadata

`metadata.arrow` stores one row per physical sample. Metadata is separate from payload bytes so sampling and filtering can inspect metadata without opening payload shards.

Metadata values are primitive Arrow-compatible values:

- Boolean.
- Signed 64-bit integer.
- 64-bit float.
- UTF-8 string.

The metadata row order is physical sample order.

## Index

`index.bin` is a binary table with one fixed-size row per sample.

Each row stores:

- Shard id.
- Byte offset within that shard.
- Payload byte length.

All integer fields are little-endian unsigned 64-bit values.

## Shards

Shard files contain concatenated payload byte slices. Payloads are not self-delimiting; the index is the authority for locating them.

Shard rotation is controlled by a validated `max_shard_bytes` configuration value. A single sample may exceed the target shard size; it is still written atomically.

Each shard manifest entry records:

- `name`
- `uncompressed_byte_len`
- `byte_len`
- `compression`: `{ "algo": "none", "ratio": 1.0 }` or `{ "algo": "lz4", "ratio": ... }`
- `sha256`

Compression is applied per indexed payload record. `index.bin` offsets and byte lengths point to stored bytes in the shard, which may be compressed. Readers decompress the single addressed payload before returning sample bytes.

## Integrity

Writers calculate SHA-256 checksums while committing files. Readers verify:

- Manifest format version.
- Manifest sample count matches metadata and index rows.
- Every indexed shard exists.
- Shard lengths match the manifest.
- SHA-256 checksums match the manifest.
- Index and metadata checksums match the manifest.

Reader validation happens before iteration starts.
