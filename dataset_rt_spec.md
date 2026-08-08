# DatasetRT --- Project Specification

> Version: 0.1 (Foundational Architecture)

## Vision

DatasetRT is a framework-independent dataset cache whose primary goal is
**correctness**. Performance is a consequence of a sound architecture,
not the primary motivation.

The system consists of a Rust core with a thin Python API.

------------------------------------------------------------------------

# Core Principles

## Rust owns correctness

Anything affecting correctness, reproducibility, integrity, or
concurrency must be implemented in Rust.

Examples:

-   cache lifecycle
-   serialization/deserialization
-   manifest validation
-   checksum verification
-   iterator state
-   sampling
-   scheduling
-   concurrency
-   weight validation
-   cache integrity

Python must never become a second implementation of these rules.

------------------------------------------------------------------------

## Invalid states are unrepresentable

Prefer Rust's type system over runtime validation.

Use:

-   enums instead of booleans
-   newtypes instead of primitive IDs
-   validated constructors
-   state-specific types
-   private fields

Avoid mutable "mode" flags.

------------------------------------------------------------------------

## Rust owns mutable state

Python objects are wrappers around opaque Rust objects.

Python never owns:

-   dataset state
-   iterator state
-   queues
-   worker pools
-   sampler
-   scheduler
-   weight vectors

------------------------------------------------------------------------

## Python expresses intent

Python is responsible only for:

-   CacheSource implementations
-   metadata manipulation
-   configuration
-   framework adapters
-   user ergonomics

------------------------------------------------------------------------

## Framework independence

The Rust core has no dependency on PyTorch or any ML framework.

Framework support is implemented through optional adapters.

------------------------------------------------------------------------

# Public Python API

``` python
class CacheSource(Protocol[DataT, MetadataT]):
    name: str

    def __iter__(self) -> Iterator[CacheInput[DataT, MetadataT]]:
        ...
```

``` python
class CacheInput(NamedTuple):
    data: DataT
    metadata: MetadataT
```

``` python
class CachedSample(NamedTuple):
    data: DataT
    metadata: MetadataT
    cache_id: int
    sample_id: int
```

``` python
runtime = DatasetRuntime(num_workers=4)
runtime.write_cache(source, path, ...)
```

``` python
dataset = runtime.cached_dataset([...], reader_config=ReaderConfig(seed=42))
```

``` python
metadata = dataset.samples_metadata()
dataset.set_samples_metadata(metadata)
```

------------------------------------------------------------------------

# Storage Layout

    cache/
        manifest.json
        metadata.arrow
        index.bin
        shards/
            000000.bin
            000001.bin

-   metadata stored separately from payload
-   immutable cache
-   Arrow for metadata
-   binary index
-   binary payload shards

------------------------------------------------------------------------

# Sampling

-   metadata-driven
-   weighted multinomial
-   epoch length defaults to active sample count and can be changed at runtime
-   deterministic for cache + weights + seed + stream position

------------------------------------------------------------------------

# Execution Model

No async runtime.

Concurrency uses native Rust threads and bounded queues.

    sampler
        ↓
    bounded read queue
        ↓
    I/O thread pool
        ↓
    bounded decode queue
        ↓
    decode thread pool
        ↓
    reorder buffer
        ↓
    bounded output queue
        ↓
    Python iterator

Characteristics:

-   synchronous public API
-   fixed thread pools
-   bounded MPMC queues
-   explicit backpressure
-   deterministic output ordering
-   no Tokio
-   no async/await
-   no Python threads
-   no Python queues

------------------------------------------------------------------------

# Writer Pipeline

    Python CacheSource
        ↓
    Rust ingestion
        ↓
    bounded serialization queue
        ↓
    serialization thread pool
        ↓
    ordered commit stage
        ↓
    shard writer

Commit stage owns:

-   sample IDs
-   shard offsets
-   metadata ordering
-   index generation
-   rolling SHA-256
-   shard rotation

------------------------------------------------------------------------

# Reader Pipeline

    metadata load
        ↓
    Rust runtime initialization
        ↓
    multinomial sampler
        ↓
    read workers
        ↓
    decode workers
        ↓
    reorder buffer
        ↓
    Python iterator

------------------------------------------------------------------------

# Determinism

Deterministic given:

-   cache contents
-   weights
-   seed
-   epoch

------------------------------------------------------------------------

# Thread Safety

-   immutable cache files
-   multiple readers allowed
-   independent runtime per process
-   iterator snapshots weight state

------------------------------------------------------------------------

# Future Extensions

-   GPU decoding
-   remote storage
-   DLPack
-   adaptive worker pools
-   distributed sampling
-   memory cache
-   additional language bindings

------------------------------------------------------------------------

# Non-Goals (MVP)

-   mutable cache
-   augmentation
-   batching
-   DataLoader replacement
-   framework dependency
-   Python concurrency

------------------------------------------------------------------------

# Recommended Technology Stack

## Rust

-   stable Rust
-   PyO3
-   maturin
-   crossbeam-channel (or flume)
-   std::thread
-   rayon (CPU parallelism where appropriate)
-   Arrow
-   Polars interoperability

## Python

-   typing
-   Protocol
-   NamedTuple
-   Polars
-   NumPy

No Python concurrency.

------------------------------------------------------------------------

# Guiding Rule

Whenever implementing a feature, ask:

> Does this affect correctness?

If yes, implement it in Rust.

If not, it may belong in Python.
