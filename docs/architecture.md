# DatasetRT Architecture

DatasetRT is a framework-independent dataset cache with a Rust correctness core and a thin Python API.

The founding rule is simple: if behavior affects correctness, reproducibility, integrity, or concurrency, it belongs in Rust. Python expresses user intent and adapts Python objects into the narrow data shapes the Rust core accepts.

## Inversion Analysis

Charlie Munger often pointed to Jacobi's maxim, "Invert, always invert." DatasetRT applies that idea by starting with the failures an ML dataset runtime must make hard:

- Silent sample duplication.
- Nondeterministic epoch order.
- Weight updates applied to the wrong physical sample.
- Readers accepting incomplete or corrupt caches.
- Startup paths that hash or materialize more data than requested.
- Queues that grow without bound under slow consumers.
- Python adapters that accidentally own correctness-sensitive iteration state.

The architecture is the inverse of those failures. Rust owns identity, sampling, validation, storage layout, bounded queues, and ordering. Python remains the ergonomic edge that describes sources and decodes payload bytes, but unsupported adapter modes should fail clearly rather than produce plausible-looking bad training data.

## Boundaries

Rust owns:

- Cache lifecycle.
- Storage layout.
- Payload byte storage and materialization.
- Metadata serialization and validation.
- Manifest and index validation.
- Checksum verification.
- Dataset and iterator state.
- Sampling, scheduling, weights, and epoch ordering.
- Thread pools, queues, backpressure, and output ordering.

Python owns:

- `CacheSource` implementations.
- Public API ergonomics.
- Project-specific conversion of domain objects into bytes payloads.
- Metadata authoring before it crosses into Rust.
- Optional framework adapters.

Python never owns cache state, iterator state, queues, worker pools, sampler state, scheduler state, or weight vectors.

## Foundational Data Contract

For v0.1, payloads are bytes-like values. A cache source may yield `bytes`, `bytearray`, or `memoryview` payloads.

This is deliberately stricter than arbitrary Python objects. Allowing arbitrary objects would require Python-side serialization rules and would create a second correctness implementation. Framework adapters can encode tensors, images, or records to bytes before calling DatasetRT, but the runtime cache format remains Rust-owned.

Metadata is a mapping from string keys to primitive Arrow-compatible values:

- `bool`
- `int`
- `float`
- `str`

Each metadata field must have a stable type across all rows. Missing values are not part of the v0.1 cache contract; sources should provide explicit values or split data into separate caches.

Weights are exposed to Python as a Polars `DataFrame` with `cache_id`, `sample_id`, metadata columns, and `weight`. Rust owns the authoritative weight vector and validates any updated table before accepting it.

## Module Shape

The Rust core is organized by responsibility:

- `types`: validated IDs, configuration values, metadata values, and errors.
- `storage`: manifest, metadata, binary index, shard layout, and validation.
- `writer`: ingestion, serialization scheduling, ordered commit, rolling checksums, and shard rotation.
- `sampling`: deterministic weighted multinomial epoch planning.
- `dataset`: immutable dataset state, weight snapshots, and iterator construction.
- `runtime`: synchronous iterator pipeline with bounded queues and native Rust threads.

The Python package exposes only:

- `CacheInput`
- `CachedSample`
- `CacheSource`
- `DatasetRuntime`
- `ReaderConfig`
- `WriterConfig`
- `ShardCompression`
- `CachedDataset`

All other implementation details are private.

## Immutability

A completed cache is immutable. Writers create the cache directory, write all data files, validate the completed layout, and only then publish `manifest.json`. Readers refuse incomplete or malformed caches.

## Framework Independence

The Rust crate has no PyTorch, TensorFlow, JAX, or ML framework dependency. Framework integration lives in optional Python adapters that convert framework-native values into bytes payloads and primitive metadata.
