# Manifest v2 compatibility fixtures

These caches and saved metadata were produced by the actual v2 writer at
revision `94f2c3e` before the v3 implementation. Do not regenerate them with
the current writer. `provenance.json` records their file checksums, producer
revision, ordered paths, seed and expected weighted sample sequence.

Each cache contains four recognizable payloads and all primitive metadata
types. `legacy-plain` is uncompressed; `legacy-é` uses per-record LZ4. A 256-byte
shard limit exercises shard rotation. The saved Arrow table contains reordered,
filtered and duplicate identities with non-uniform weights, exported by the
legacy reader. It remains the compatibility oracle before and after migration.
