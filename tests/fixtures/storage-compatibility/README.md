# Storage-format compatibility oracles

These are immutable caches plus ground truth captured by each producer's own
reader. Tests never regenerate them with the candidate writer or derive expected
values from the candidate reader.

| Format | Producer source | Codec cases |
| --- | --- | --- |
| v2 | `94f2c3e1d22930f9dfd5402b098bc45e702e3b1e` | legacy none, LZ4 |
| v3 | `38864c828fb639ea58b6966c9c96a94d701320a3` | legacy none, LZ4 |
| v4 | `bfe84fef1da1aa75ddbdcfd3e82000681cc79b3f` | none, LZ4 Fast, LZ4 HC, Brotli, Zstd |
| v5 | Explicit working-tree snapshot based on `bfe84fef…` | none, LZ4 Fast, LZ4 HC, Brotli, Zstd |

V5 was captured before its implementation was committed. Its oracle explicitly
records `working_tree: true` and a SHA-256 fingerprint of the exported build
inputs; the base revision alone does **not** identify that producer. V2–v4 were
exported directly from the listed commits without source edits. Every oracle
records the producer package/Python versions and source fingerprint.

All producers use release builds with `CARGO_PROFILE_RELEASE_STRIP=none`, matching
the project's macOS LINKEDIT workaround without modifying historical source.
The native algorithms and stored layouts remain those of the selected producer.

Each codec cache contains six recognizable payloads: empty bytes, a sentence,
UTF-8 text, binary bytes, zeroes and repeated text. A 512-byte shard target
exercises rotation and oversized records. Metadata covers booleans, large negative
integers, exact floats, empty strings, Unicode and source identity.

Each `vN/oracle.json` contains:

- Complete sequential outputs: payload bytes encoded as hex, all metadata, and
  producer-resolved `(cache_id, sample_id)` identities.
- Two consecutive seeded weighted epochs, represented by physical identities.
- Producer provenance, codec settings, cache path order and file checksums.

`weights.arrow` is exported by that producer after assigning non-uniform weights.
The current reader loads those exact saved weights when checking epoch order.
Fixture generation also checks producer output against independently specified
input values before recording it as ground truth.

## Adding a format

After implementing the new writer format, capture it from its own code. Prefer
a committed producer revision so the build is directly reproducible:

```bash
uv run --python 3.11 --extra dev scripts/generate_storage_fixtures.py \
    --versions '[6]' --revision <producer-commit>
```

The script exports the selected commit into a temporary checkout, builds a locked
release wheel, creates an isolated environment, runs that writer and reader, then
publishes the complete `v6` fixture directory. It automates all setup. Nothing is
installed into the historical checkout or the system Python environment.

For an intentionally uncommitted producer, explicitly use
`--revision working-tree`. This copies only current build inputs and records
their fingerprint and the snapshot marker. It does not reset, stash or check out
over the user's working tree.

Add new codec cases to the generator when the format introduces another codec
wire variant. Existing formats and their oracles must remain unchanged. The
generator refuses existing output directories and publishes only successful
captures, so failed builds cannot leave a partial version directory.

For the initial v2–v5 corpus, the generator defaults to the pinned historical
revisions and a current v5 snapshot. To reproduce it for investigation, use a
new `--output` directory; never overwrite the checked-in originals.

Run:

```bash
uv run --python 3.11 --extra dev pytest tests/test_storage_compatibility.py
```

The test discovers every saved `vN/oracle.json` automatically. A separate guard
probes the current default writer and requires uninterrupted oracle coverage from
v2 through that format. Bumping the writer to v6 without adding its oracle fails
CI. V1 is intentionally outside the compatibility promise.
