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
- Persistent integer `cache_id` in format v3.
- Source name.
- Sample count.
- Metadata schema.
- Index checksum.
- Metadata checksum.
- Shard names, byte lengths, compression metadata, and SHA-256 checksums.

If `manifest.json` is missing, the cache is incomplete and must not be read.

### Versioned cache identity

Readers support manifest versions 2 and 3 through separate data structures.
Version 2 has no persistent cache identity: `cache_id` is the cache path's
zero-based position in the supplied dataset list. Reading or reusing v2 caches
does not change their manifests or their IDs. Version 3 requires an unsigned
64-bit `cache_id`, which readers use regardless of path order or directory name.
Unsupported versions and duplicate resolved IDs are rejected. Mixed v2/v3
datasets use each version's identity rule, without automatic renumbering.

New caches use SHA-256 of the source name's exact UTF-8 bytes. Interpret the
first eight digest bytes as a big-endian unsigned integer and clear the high bit.
Store that nonnegative 63-bit result in the v3 manifest. Names are not normalized
before hashing, and Python's randomized `hash()` is not used. Equal names give
equal IDs; readers reject collisions instead of assigning replacement IDs.

### Explicit migration

Load the original ordered cache list used by existing sample tables, then call
`dataset.update_manifests(version=3)`. It writes each v2 cache's current positional
ID into a v3 manifest, preserving `(cache_id, sample_id)` previously generated tables without remapping.
Migrated IDs intentionally need not equal the source-name hash. V3 manifests keep
their stored IDs. A subset of the original paths cannot infer omitted positions.

Rust checks every target before replacement. Each manifest is written beside
its resolved target, flushed, synced, atomically renamed and its directory synced.
Symlinks remain intact. Unrelated manifest fields and all sample files are
preserved. The operation changes no dataset metadata, weights or iterator state.
The caller owns manifest mutation exclusively; no lock files are used.

The method returns `Ok(ManifestUpdateReport)` or `Err(ManifestUpdateError)`.
Entries identify `updated`, `unchanged` or `durability_unknown` manifests. A
directory-sync failure after rename reports that replacement, with unconfirmed
durability. Several caches are not one atomic transaction: inspect partial
progress and retry in the original order. A retry also syncs unchanged targets.
Interrupts propagate after owned temporary-file cleanup. Hard termination can
leave temporary manifest files; these are never treated as publication markers
or removed as if they belonged to a later operation.

Older DatasetRT releases that only accept v2 cannot read upgraded v3 caches.

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
