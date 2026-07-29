# Runtime Model

DatasetRT exposes a synchronous Python API backed by native Rust execution.

No async runtime is used. There is no Tokio, no `async`/`await`, no Python threads, and no Python queues.

## Reader Pipeline

```text
metadata load
    -> Rust runtime initialization
    -> physical-order planner or deterministic weighted sampler
    -> bounded read task queue
    -> sample-loading worker pool
    -> reorder buffer
    -> Python iterator
```

Rust owns every queue, worker, and iterator cursor.

Payload materialization means assembling cache records from shard bytes, metadata, `cache_id`, and `sample_id`. It does not mean JPEG, PNG, tensor, or domain-object decoding; that belongs in Python or optional framework adapters.

Reader configuration:

- `prefetch_size`: bounded Rust read/result queue capacity.
- `num_workers`: fixed Rust sample-loading worker count.
- `shuffle`: choose deterministic weighted sampling or physical cache order.

If storage reads are slower than Python consumption, Rust workers fill up to `prefetch_size` results. If Python consumption is slower than storage reads, workers block instead of growing memory without bound.

## Writer Pipeline

```text
Python CacheSource
    -> Rust ingestion
    -> bounded serialization queue
    -> serialization thread pool
    -> ordered commit stage
    -> shard writer
```

The commit stage owns:

- Physical sample IDs.
- Shard offsets.
- Metadata ordering.
- Index generation.
- Rolling SHA-256.
- Shard rotation.

Writer configuration:

- `prefetch_size`: bounded Rust ingestion queue capacity.
- `num_threads`: fixed Rust serialization worker count.
- `show_progress`: optional Rust-owned progress rendering with committed samples/s and MB/s for the active source, plus source-count progress and ETA for multi-source writes.

If Python iteration is faster than writing, Rust prefetches up to `prefetch_size` samples and then blocks the ingestion edge. This gives burst smoothing without unbounded memory growth.

## Backpressure

Queues are bounded. If downstream work cannot keep up, upstream producers block. This keeps memory usage controlled and makes execution behavior explicit.

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
