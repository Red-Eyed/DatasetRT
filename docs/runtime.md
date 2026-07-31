# Runtime Model

DatasetRT exposes a synchronous Python API backed by native Rust execution.

No async runtime is used. There is no Tokio, no `async`/`await`, no Python threads, and no Python queues. The first reader or writer operation initializes one fixed process-wide Rust worker pool from its required top-level `num_workers` argument, and DatasetRT reuses those threads for cache loading, reading, and writer jobs. Every later operation must pass the same value; hardware parallelism is never selected implicitly.

## Reader Pipeline

```text
metadata load
    -> Rust runtime initialization
    -> physical-order planner or deterministic weighted sampler
    -> bounded in-flight read window
    -> process-wide worker pool
    -> reorder buffer
    -> Python iterator
```

Rust owns every queue, worker, and iterator cursor.

Payload materialization means assembling cache records from shard bytes, metadata, `cache_id`, and `sample_id`. It does not mean JPEG, PNG, tensor, or domain-object decoding; that belongs in Python or optional framework adapters.

Reader operation settings:

- `prefetch_size`: bounded Rust read/result queue capacity.
- top-level `num_workers`: fixed process pool size and maximum active read jobs.
- `shuffle`: choose deterministic weighted sampling or physical cache order.

Each iterator keeps at most `min(num_workers, prefetch_size)` reads active. Completed reads fit in a result queue of the same size, so workers do not wait on a full per-iterator result queue. Slow Python consumption stops new submissions instead of growing memory without bound.

## Writer Pipeline

```text
Python CacheSource
    -> Rust ingestion
    -> bounded in-flight writer window
    -> process-wide worker pool
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
- top-level `num_workers`: fixed process pool size and maximum active writer jobs.
- `show_progress`: optional Rust-owned progress rendering with committed samples/s and MB/s for the active source, plus source-count progress and ETA for multi-source writes.
- `validate_cache`: optional checksum validation for existing caches before writer reuse.
- `profiler`: optional JSON timing summary for diagnosing whether time is spent in Python iteration, Python-to-Rust extraction, queue backpressure, compression, disk writes, finish steps, or cache publish.

If Python iteration is faster than writing, Rust keeps at most `min(num_workers, prefetch_size)` writer jobs active and then commits a completed job before pulling more input. For multi-source writes, the bounded window can span source boundaries while ordered commit keeps manifests and result ordering deterministic.

Each operation's result queue has the same capacity as its active-job limit. Cache commit and publish remain sequential for deterministic manifests and result ordering.

## Backpressure

The global task queue and every operation result queue are bounded. Submission applies backpressure when the global pool is saturated, and each operation reserves result capacity before submitting work. This keeps memory controlled without allowing pool workers to deadlock on full operation queues.

## Cache Validation

Readers and writer reuse skip checksum validation by default. Dataset construction still reads manifests, metadata, indexes, and shard file lengths, but it does not hash metadata, index, or payload shard contents unless `validate_cache=True` is set on the relevant config. This keeps restart time tied to cache metadata size instead of payload size.

## Ordering

Output order is deterministic. Workers may complete out of order, but the reorder stage publishes samples in the sampler's planned order.

With `shuffle=False`, the plan is physical cache order. With `shuffle=True`, the plan comes from deterministic weighted sampling.

## Iterator Snapshots

When `shuffle=True`, each iterator snapshots:

- Cache manifests.
- Current weight vector.
- Seed.
- Epoch number.

Weight changes made after iterator construction do not affect that iterator.
