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
- Persistent integer `cache_id` in formats v3–v5.
- Source name.
- Sample count.
- Metadata schema.
- Index checksum.
- Metadata checksum.
- Shard names, byte lengths, compression metadata, and SHA-256 checksums.

If `manifest.json` is missing, the cache is incomplete and must not be read.

### Versioned cache identity

Readers distinguish legacy v2 identity from persisted identity in v3–v5.
Version 2 has no persistent cache identity: `cache_id` is the cache path's
zero-based position in the supplied dataset list. Reading or reusing v2 caches
does not change their manifests or their IDs. Version 3 requires an unsigned
64-bit `cache_id`, which readers use regardless of path order or directory name.
Unsupported versions and duplicate resolved IDs are rejected. Mixed-version
datasets use each version's identity rule, without automatic renumbering.

New caches use SHA-256 of the source name's exact UTF-8 bytes. Interpret the
first eight digest bytes as a big-endian unsigned integer and clear the high bit.
Store that nonnegative 63-bit result in the manifest. Names are not normalized
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

## Record layout dispatch

Rust resolves each manifest once into a closed `RecordLayout`: `LegacyEnvelope`
(v2/v3), `RecordEnvelope` (v4), or `PayloadOnly` (v5). The record module owns
exhaustive framing and codec dispatch. Decoded values carry their metadata source
as a typed variant, so storage orchestration does not inspect versions or infer
missing metadata from an empty mapping. Metadata resolution and validation remain
separate from binary record framing. Compression policy/layout conflicts are
rejected when parsing the manifest.

Compatibility is tested against immutable caches and complete reader ground truth
captured from each format's own producer. The matrix covers v2–v5, all supported
codec wire variants, weighted epochs and mixed-format datasets. See the
`tests/fixtures/storage-compatibility/README.md` workflow
and `tests/test_storage_compatibility.py`. A new default writer format requires
its own saved oracle before the compatibility coverage guard can pass.

## Metadata

`metadata.arrow` stores one row per physical sample. Writers stream rows into Arrow
record batches bounded by row count and owned metadata bytes, keeping each UTF-8
column within its signed 32-bit offset limit. A single row may exceed the batch
byte target, but individual strings exceeding the Arrow limit are rejected with
a typed input error before array construction. Batch boundaries do not change
sample order or the cache format. Empty metadata mappings produce zero-column
batches with explicit row counts; the first sample fixes the schema even when it
has no columns. Metadata is separate from payload bytes so sampling and filtering
can inspect metadata without opening payload shards.

Writers retain at most one metadata batch plus the current sample. Index entries
are appended directly to `index.bin`; neither all metadata rows nor all index
entries accumulate in memory. Arrow retains one small footer descriptor per
record batch, and the manifest retains one descriptor per shard. Completed
samples awaiting an earlier sample's ordered commit still count against the
writer's scheduling window.

V5 stores metadata exclusively in `metadata.arrow`. Sample reads lazily map that
file and index batch row ranges from IPC message headers, without decoding all
columns or expanding rows at dataset construction. After first validation of a
batch, lookup is O(log batches + metadata columns + selected metadata bytes). First access validates that batch's Arrow buffers in
O(batch bytes). A shared per-batch marker allows later decoders to reuse that
validation, avoiding repeated whole-column UTF-8 scans during shuffled reads.
Only successfully validated bytes from the same immutable mapping are trusted;
failed validation never enables reuse. Each worker retains one decoded batch
total across all caches;
file mappings and O(batches) descriptors are shared by that dataset's readers.
Mapped pages are managed by the OS. Unaligned or externally compressed Arrow
buffers may require copies of the addressed batch. Native writers bound batches
by rows and bytes, except for a single oversized metadata row.

Legacy v2–v4 records also embed metadata JSON. With `validate_cache=True`, readers
compare that copy against Arrow when reading a sample. V5 has no redundant copy
to compare; full validation checks checksums and streams Arrow schema/null
validation one batch at a time. Default v5 reads validate row count from batch
headers and validate each addressed batch's schema and nulls before reading cells.
A raw v5 shard cannot provide sample metadata without `metadata.arrow`.

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

Readers map the immutable `index.bin` read-only instead of allocating a decoded
table per process. Each lookup checks the row bounds and decodes its three
integers in O(1) time. Startup still checks row count and scans shard references
in O(samples) time, without retaining decoded entries. Checksum hashing remains
opt-in through `validate_cache=True`. The binary format is unchanged.

Independent readers discover the index from each supplied cache directory and
map the same file. The OS shares file-backed pages across readers on one host;
the mapping remains owned by the loaded cache after its file handle closes.
Keep published cache files immutable while readers are alive.

## Shards

Shard files contain concatenated sample records. The index is the authority for locating each record.

In formats v2/v3, each indexed record stores:

- Metadata JSON byte length as a little-endian unsigned 64-bit value.
- Metadata JSON object.
- Stored payload bytes.

Shard rotation is controlled by a validated `max_shard_bytes` configuration value. A single sample may exceed the target shard size; it is still written atomically.

Each shard manifest entry records:

- `name`
- `uncompressed_byte_len`
- `byte_len`
- `compression`: legacy `{ "algo": "none", "ratio": 1.0 }` / `{ "algo": "lz4", "ratio": ... }`, or a v4 mode described below.
- `sha256`

For v2/v3, compression is applied per indexed payload, after the redundant metadata envelope. `index.bin` offsets and byte lengths point to full sample records in the shard. Readers parse the metadata envelope, validate it against `metadata.arrow`, and then decompress the single addressed payload before returning sample bytes.

### Format v4: complete-record compression

DatasetRT 0.5.0–0.5.9 wrote format v4 with typed compression policies.
The logical record remains an eight-byte metadata length, metadata JSON, then
payload bytes. LZ4 compresses that entire record as one independent block, with
a four-byte little-endian decoded-length prefix. `NoCompression` writes the
logical record directly. Index offsets and lengths address the stored record;
reading one sample never requires decoding adjacent samples or a whole shard.
The metadata copy in `metadata.arrow` remains available for columnar queries.

V4 compression settings are typed variants:

- `{ "algo": "none", "mode": "none" }`
- `{ "algo": "lz4", "mode": "fast", "acceleration": 1 }`
- `{ "algo": "lz4", "mode": "hc", "level": 9 }`
- `{ "algo": "brotli", "mode": "brotli", "level": 5 }`
- `{ "algo": "zstd", "mode": "zstd", "level": 3 }`

Fast acceleration accepts integers 1–65537; HC levels accept integers 1–12.
Brotli accepts levels 0–11. Brotli records contain a four-byte little-endian
decoded-length prefix followed by a standard Brotli stream of the complete
logical record. The decoder writes into an exactly bounded output buffer and
requires the actual decoded length to match the prefix.
Zstd accepts positive standard levels 1–22. Each record is a single standard
Zstandard frame, including frame magic, content size, and a content checksum.
There is no additional DatasetRT size prefix. The content size must be present
and no larger than the shard's declared `uncompressed_byte_len`; output is
allocated fallibly to exactly that size. Decoding must produce that exact length.
Skippable frames, unknown content sizes, concatenated frames, trailing bytes,
truncation, and checksum failures are rejected.
Higher acceleration favors write speed; higher HC levels increase compression
effort. No setting promises a particular size ratio. The reader uses the same
standard LZ4 block decoder for both compressed modes. Compression mode, record
scope, and manifest version must agree, or the cache is rejected.

For v4, `uncompressed_byte_len` sums complete logical record bytes, including
metadata envelopes. In v2/v3 it sums payload bytes only. `byte_len` always counts
actual stored shard bytes. LZ4 input records must fit the backend's validated
block-size limit; unsupported sizes return errors before compression.

Existing `ShardCompression(...)` configurations retain their v3 payload-only
behavior and emit a Python deprecation warning when constructed explicitly.
Their historical `ratio` remains manifest metadata and is never used as a level.
The default for new writes is `Zstd(level=3)`, now using v5. Old caches stay readable without
rewriting, and a dataset can combine v2, v3, v4, and v5 caches. Releases before
0.5.0 reject v4; releases before 0.5.3 reject the Zstd codec discriminator even
when they support other v4 codecs. Previously written LZ4 v4 caches remain compatible with 0.5.0–0.5.2 readers. Manifest-only identity migration still targets v3 and leaves
v4/v5 caches unchanged; it cannot convert record layouts.

### Format v5: payload-only shards

Starting with 0.5.10, typed `NoCompression`, `Lz4Fast`, `Lz4HC`, `Brotli`, and
`Zstd` policies write v5. Index rows address only stored payload bytes. Metadata
JSON and its eight-byte length prefix are absent. Codec framing and settings
are unchanged from v4, but codec input and `uncompressed_byte_len` count only
payload bytes. Empty uncompressed payloads have zero-length index entries and
may share an offset; an all-empty shard is a valid zero-byte file.

Readers before 0.5.10 reject v5. Existing caches stay unchanged, including with
`reuse_existing=True`. Removing duplication from an old cache requires rewriting
it into a new destination; changing a manifest version cannot convert the layout.
Deprecated `ShardCompression` policies continue writing the historical v3 layout.
See [storage benchmarks](storage-benchmarks.md) for paired text/image comparisons.

## Integrity

Writers flush and sync every shard, `metadata.arrow`, and `index.bin` before
writing the manifest. The manifest is explicitly flushed and synced, then its
cache directory is synced. Publication renames the completed cache directory and
syncs the source parent, destination parent, and destination ancestors to persist
the rename and newly created paths.
Write, flush, and sync errors are returned to the caller. If a sync fails after
rename, the error states that the cache exists but publication durability is
unconfirmed; the published directory is retained for inspection.

Writers calculate SHA-256 checksums while committing files. Readers verify:

- Manifest format version.
- Manifest sample count matches metadata and index rows.
- Every indexed shard exists.
- Shard lengths match the manifest.
- SHA-256 checksums match the manifest.
- Index and metadata checksums match the manifest.
- Legacy embedded metadata matches `metadata.arrow` when full validation is enabled.

Full checksum and shape validation happens at construction when explicitly enabled.
Record bounds, codecs and addressed v5 metadata batches are checked during reads.

## Native error handling

Readers reject indexed record ranges outside their declared shard before payload
allocation. LZ4 size prefixes must fit within the shard's declared uncompressed
length, and decoded lengths must match their prefix. Full metadata validation
checks column count, names, types, and null values before reading cells.

Input and cache validation errors reach Python as `ValueError`. Unexpected native
unwinds contained at worker and dependency boundaries become `RuntimeError` with
the operation and original diagnostic. Payload and channel reservations are
fallible; allocator aborts, OS termination, and invalid memory access remain
outside this recovery contract.
