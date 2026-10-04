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

Python keeps an immutable internal recipe containing original ordered cache paths, reader settings, the configured sample count, and either original-cache metadata or accepted metadata IPC. Recipe export does not create a native reader, read payloads, or export a default full metadata table. It does not copy an active iterator's cursor.

Metadata updates retain the exact IPC only after Rust accepts it. Failed updates preserve the previous recipe; prior exported recipes remain immutable snapshots. Retaining an override costs its IPC byte size in addition to native metadata state. Default datasets retain only O(cache paths) construction inputs, with no eager metadata export or per-row Python objects.

Reconstruction restores accepted IPC directly into Rust. It must not decode/re-encode snapshots through Polars in a forked child: inherited Polars thread-pool state can block even when the child creates a fresh DatasetRT runtime. Prepare sequential columnar row slices in the training process before worker launch and select them using the actual worker identity; empty slices yield nothing without native construction. Original cache IDs and intentional duplicate metadata rows remain intact.

The pure contiguous-span helper divides rows across ranks and then local workers without batch-size input, padding, or added duplicates. Zero DataLoader workers means one local consumer. For a fixed base seed, reader-seed derivation version 1 packs u32 rank/worker IDs and uses a seed-keyed SplitMix64 permutation; distinct rank/worker pairs cannot collide. The seed is unused for sequential reading. Omitted shuffled seeds use fresh OS randomness once during process-local setup. Rank/group-size capture runs in the training process, with single-rank fallback when no group is initialized; loading workers do not query process groups.

## PyTorch DataLoader helper

`CachedDataset.to_torch_dataloader()` returns a standard PyTorch DataLoader supporting zero-worker loading and caller-selected fork/spawn/forkserver contexts. Real multi-rank DDP acceptance is subsequent work. The legacy `to_torch_iterable_dataset()` retains its original behavior.

The internal adapter's idempotent `setup()` creates one consuming runtime/dataset, records its PID, and reuses it. With workers, internal `worker_init_fn` runs setup before the caller's initialization callback; with zero workers, iteration supplies the fallback. Construction and length queries do not create a consuming reader. Empty validation partitions complete setup without a native reader. Native state never enters serialized adapter state; a serialized initialized adapter reconstructs an independent stream. An initialized adapter inherited into a different PID cannot reuse its native objects.

With `shuffle=True` (the helper default), reading is infinite weighted sampling with replacement over the full active population. Explicit seeds reproduce newly initialized streams; omitted seeds draw OS randomness once at setup. Repeated iterator calls continue the retained draw iterator, including unfinished native windows, without resetting the seed. Source `set_epoch_len()` controls native window size, not a training sample quota; an infinite loader has no finite length. Retaining a paused loader retains its bounded native prefetch state.

With `shuffle=False`, validation traverses each rank's contiguous active-row partition, split among actual local workers, once per iterator. It uses active row count, ignoring source epoch-length overrides and seed, and never pads partitions. Subsequent passes reuse the same native dataset. Preparation decodes once and slices columnarly in the training process in O((active rows + workers) × metadata columns) work, including per-slice schema encoding; original duplicates, weights, extras, and physical IDs remain intact. Workers select prepared slices using their actual IDs and restore IPC directly in Rust. They never enter inherited Polars pools or query distributed groups. Empty partitions create no native reader. Shuffled construction instead retains only O(cache paths) inputs plus any existing metadata snapshot. Native startup and queues retain their existing costs and bounds.

Sequential adapters retain all prepared rank-local IPC slices for worker selection. Their total parent size is O(rank metadata bytes + workers × schema bytes); spawn/forkserver copies that set to each worker, so aggregate IPC storage can reach O(workers × rank metadata bytes + workers² × schema bytes), in addition to worker-local native metadata and replicated cache indexes. The original full-table override is excluded once slices are prepared. Fork may share immutable bytes through copy-on-write. This tradeoff avoids worker-side Polars and a custom metadata transport; account for it when measuring process memory.

`multiprocessing_context`, `prefetch_factor`, `timeout`, and `persistent_workers` follow PyTorch's constructor and lifecycle rules. The default context is PyTorch's platform default; no automatic context switch is made. Persistent workers keep native readers, seed, and sampling state. Non-persistent workers reconstruct once in each new process. Recipe edits in the training process do not propagate to workers; construct a new loader to change snapshots or seeds. Callbacks must be importable/picklable under spawn/forkserver, and application process-launch code must use the usual main guard.

PyTorch batches each worker's iterable independently, so `len(loader)` can differ from actual batch count for incomplete worker tails. Dropping a partial iterator or resetting persistent workers can discard already-prefetched outputs according to ordinary PyTorch behavior; native streams continue without seed reset, but concatenated consumer prefixes need not be gap-free. No custom cleanup, transport, sampler, or equal-step policy is introduced. Worker initialization, transform, native-read failures, worker death, and timeouts propagate through PyTorch.

PyTorch owns `batch_size`, collation, `drop_last`, and pinning. `sample_transform_fn` receives a `CachedSample` after native delivery and returns a domain value; exceptions propagate. Without a transform or collator, Torch's ordinary conversion/collation rules apply to the sample fields. `native_num_workers` controls Rust read threads, separately from DataLoader workers.

```python
loader = dataset.to_torch_dataloader(
    shuffle=True, seed=123, batch_size=32,
    num_workers=4, multiprocessing_context="forkserver", persistent_workers=True,
    sample_transform_fn=decode_sample,
)
for step, batch in enumerate(loader):
    train_step(batch)
    if step + 1 == training_steps:
        break

validation_loader = dataset.to_torch_dataloader(
    shuffle=False, batch_size=32, sample_transform_fn=decode_sample,
)
for batch in validation_loader:
    validate_batch(batch)
```

## Ordering

Output order is deterministic. Workers may complete out of order, but the reorder stage publishes samples in the sampler's planned order.

With `shuffle=False`, the plan is active metadata table order. With `shuffle=True`, the plan comes from deterministic weighted sampling over active metadata rows.

## Iterator Snapshots

Each iterator snapshots:

- Cache manifests.
- Current active metadata table.
- Seed and epoch number when `shuffle=True`.

Metadata changes made after iterator construction do not affect that iterator.
