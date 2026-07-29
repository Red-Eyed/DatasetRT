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

The same metadata is also embedded redundantly in each shard record. That copy is for debugging, visualization, and sample-level inspection when iterating through raw shard records. Readers compare the embedded metadata with `metadata.arrow` when materializing a sample and reject mismatches.

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

Shard files contain concatenated sample records. The index is the authority for locating each record.

Each indexed record stores:

- Metadata JSON byte length as a little-endian unsigned 64-bit value.
- Metadata JSON object.
- Stored payload bytes.

Shard rotation is controlled by a validated `max_shard_bytes` configuration value. A single sample may exceed the target shard size; it is still written atomically.

Each shard manifest entry records:

- `name`
- `uncompressed_byte_len`
- `byte_len`
- `compression`: `{ "algo": "none", "ratio": 1.0 }` or `{ "algo": "lz4", "ratio": ... }`
- `sha256`

Compression is applied per indexed payload, after the redundant metadata envelope. `index.bin` offsets and byte lengths point to full sample records in the shard. Readers parse the metadata envelope, validate it against `metadata.arrow`, and then decompress the single addressed payload before returning sample bytes.

## Integrity

Writers calculate SHA-256 checksums while committing files. Readers verify:

- Manifest format version.
- Manifest sample count matches metadata and index rows.
- Every indexed shard exists.
- Shard lengths match the manifest.
- SHA-256 checksums match the manifest.
- Index and metadata checksums match the manifest.
- Embedded shard metadata matches `metadata.arrow` when a sample is read.

Reader validation happens before iteration starts.
