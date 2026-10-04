# Runtime Model

DatasetRT exposes a synchronous Python API backed by native Rust execution.

No async runtime is used. There is no Tokio, no `async`/`await`, no Python threads, and no Python queues. `DatasetRuntime(num_workers=...)` creates one fixed Rust worker pool, and every operation called through that object reuses those threads for cache loading, reading, and writer jobs. Hardware parallelism is never selected implicitly. Creating another runtime creates another independent, explicitly sized pool.

## Reader Pipeline

```text
metadata load
    -> DatasetRuntime
    -> physical-order planner or deterministic weighted sampler
    -> bounded in-flight read window
    -> runtime-owned worker pool
    -> reorder buffer
    -> Python iterator
```

Rust owns every queue, worker, and iterator cursor.

Payload materialization means assembling cache records from shard bytes, metadata, `cache_id`, and `sample_id`. It does not mean JPEG, PNG, tensor, or domain-object decoding; that belongs in Python or optional framework adapters.

Reader operation settings:

- `prefetch_size`: bounded Rust read/result queue capacity.
- runtime `num_workers`: fixed pool size and maximum active read jobs.
- `shuffle`: choose deterministic weighted sampling or physical cache order.

Each iterator keeps at most `min(num_workers, prefetch_size)` reads active. Completed reads fit in a result queue of the same size, so workers do not wait on a full per-iterator result queue. Slow Python consumption stops new submissions instead of growing memory without bound.

## Writer Pipeline

```text
Python CacheSource
    -> Rust ingestion
    -> bounded in-flight writer window
    -> runtime-owned worker pool
    -> caller-owned ordered commit stage
    -> shard writer
```

The commit stage owns:

- Physical sample IDs.
- Shard offsets.
- Metadata ordering.
- Index generation.
- Rolling SHA-256.
- Shard rotation.

Writer operation settings:

- `prefetch_size`: maximum buffered writer task/result capacity.
- runtime `num_workers`: fixed pool size and maximum active writer jobs.
- `show_progress`: optional Rust-owned progress rendering with committed samples/s and MB/s for the active source, plus source-count progress and ETA for multi-source writes.
- `validate_cache`: optional checksum validation for existing caches before writer reuse.
- `profiler`: optional JSON timing summary for diagnosing whether time is spent in Python iteration, Python-to-Rust extraction, queue backpressure, compression, disk writes, finish steps, or cache publish.

If Python iteration is faster than writing, Rust keeps at most `min(num_workers, prefetch_size)` writer jobs active and then commits a completed job before pulling more input. For multi-source writes, the bounded window can span source boundaries while ordered commit keeps manifests and result ordering deterministic.

Each operation's result queue has the same capacity as its active-job limit. Cache commit and publish remain sequential for deterministic manifests and result ordering.

## Backpressure

The runtime task queue and every operation result queue are bounded. Submission applies backpressure when the runtime pool is saturated, and each operation reserves result capacity before submitting work. This keeps memory controlled without allowing pool workers to deadlock on full operation queues.

Queues use their own mutexes and condition variables. They do not block through Rust's thread-local parker, whose used macOS semaphore is invalid after fork. A runtime constructed in the consuming process creates fresh queue synchronization, including when the parent has already used DatasetRT. Existing native runtimes, datasets, and iterators must still not be reused in a forked child or serialized to spawned workers; pass construction settings and cache paths instead.

The [standalone C reproducer](../scripts/repro_macos_fork.c) demonstrates the macOS failure using only system libraries. The [investigation notes](fork-reproducer.md) include build commands, passing controls, and further Rust and direct Mach reductions.

Dropping the last receiver disconnects producers, wakes blocked submissions, and releases queued payloads outside the queue lock. Dropping the last sender wakes receivers; already queued values remain available before disconnection is reported. Timed writer waits retain Python signal checks.

## Cache Validation

Readers and writer reuse skip checksum validation by default. Dataset construction still reads manifests, metadata, indexes, and shard file lengths, but it does not hash metadata, index, or payload shard contents unless `validate_cache=True` is set on the relevant config. This keeps restart time tied to cache metadata size instead of payload size.

## Internal Reader Reconstruction Contracts

Python keeps an immutable internal recipe containing original ordered cache paths, reader settings, the configured sample count, and either original-cache metadata or accepted metadata IPC. Recipe export does not create a native reader, read payloads, or export a default full metadata table. It does not copy an active iterator's cursor. Public DataLoader construction is a later feature; these are its internal prerequisites.

Metadata updates retain the exact IPC only after Rust accepts it. Failed updates preserve the previous recipe; prior exported recipes remain immutable snapshots. Retaining an override costs its IPC byte size in addition to native metadata state. Default datasets retain only O(cache paths) construction inputs, with no eager metadata export or per-row Python objects.

Reconstruction restores accepted IPC directly into Rust. It must not decode/re-encode snapshots through Polars in a forked child: inherited Polars thread-pool state can block even when the child creates a fresh DatasetRT runtime. Prepare sequential columnar row slices in the training process before worker launch and select them using the actual worker identity; empty slices yield nothing without native construction. Original cache IDs and intentional duplicate metadata rows remain intact.

The pure contiguous-span helper divides rows across ranks and then local workers without batch-size input, padding, or added duplicates. Zero DataLoader workers means one local consumer. For a fixed base seed, reader-seed derivation version 1 packs u32 rank/worker IDs and uses a seed-keyed SplitMix64 permutation; distinct rank/worker pairs cannot collide. The seed is unused for sequential reading. Omitted shuffled seeds use fresh OS randomness at the consuming iterator boundary. Rank/group-size capture runs in the training process, with single-rank fallback when no group is initialized; loading workers do not query process groups.

## Ordering

Output order is deterministic. Workers may complete out of order, but the reorder stage publishes samples in the sampler's planned order.

With `shuffle=False`, the plan is active metadata table order. With `shuffle=True`, the plan comes from deterministic weighted sampling over active metadata rows.

## Iterator Snapshots

Each iterator snapshots:

- Cache manifests.
- Current active metadata table.
- Seed and epoch number when `shuffle=True`.

Metadata changes made after iterator construction do not affect that iterator.
