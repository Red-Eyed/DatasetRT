# Runtime Model

DatasetRT exposes a synchronous Python API backed by native Rust execution.

No async runtime is used. There is no Tokio or `async`/`await`. `DatasetRuntime(num_workers=...)` creates one fixed Rust worker pool, and native operations reuse those threads for cache loading, reading, and writer jobs. Optional spawned source writing uses bounded Python control mailboxes; sample payloads and native prefetch stay inside each writing process. Hardware parallelism is never selected implicitly. Creating another runtime creates another independent, explicitly sized pool.

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

### Optional source processes

`WriterConfig(num_processes=0)` preserves native writing in the caller. Positive
values spawn up to that many children, limited by source count. Each child creates
one native runtime using the caller's `num_workers` and reuses it across jobs.
Independent sources are assigned dynamically, so a slow first source does not
block completion collection or later jobs. The returned list remains input-ordered.

```python
if __name__ == "__main__":
    runtime = DatasetRuntime(num_workers=1)
    results = runtime.write_cache(
        sources,
        cache_root,
        writer_config=WriterConfig(num_processes=4, process_timeout_seconds=3600),
    )
```

Source descriptors must be picklable and importable under spawn. Keep descriptors
small and open files/connections in `__iter__`; serial sources need not be
picklable. Each child receives at most one active source through a capacity-one
mailbox. Python retains O(processes) serialized descriptions and O(sources)
ordered outcomes, with no per-sample transport or metadata expansion. Per-child
native memory remains bounded by the existing writer window; aggregate native
threads and buffers scale with the number of children, in addition to the caller's
runtime. Payload-size bounds still belong to the application.

Each source has one owning worker and one destination. Preflight rejects duplicate,
case-folded, and Unicode-normalized names before any worker starts, including on
case-sensitive filesystems. No destination locks or lock files are used. Callers
must exclude concurrent write invocations targeting the same destinations.
Parallel writing reserves the `tmp` source name, refuses existing `tmp/<source>`
paths, and removes only temporary paths owned by the operation after the associated
child stops. Published caches remain untouched. Hard parent termination can leave
temporary caches requiring inspection before another parallel attempt.

Cache roots and their `tmp` directories may be symlinks. Cleanup tracks the
resolved per-source temporary path and leaves the symlink and shared target
directory intact. Keep these paths stable during an active write. Publication
still uses native rename, so temporary and published caches must be on the same
filesystem.

Serialization and ordinary source failures produce `CacheWriteError` values and
allow healthy jobs to continue. Worker death, transport failure, or the configured
`process_timeout_seconds` stops the pool and reports unfinished jobs as errors;
completed outcomes are retained. No source is retried automatically. A cache
may have been published before its worker died delivering the outcome; inspect
or reuse that complete cache rather than assuming an error implies no side effects.
The timeout is one hour by default, applies separately to startup and each
dispatched source (including child deserialization), and excludes source-name
access and parent-side pickle hooks. Those caller-executed hooks must not block.
Shutdown uses bounded joins with termination and kill escalation; source-created
subprocesses are outside the writer's ownership. KeyboardInterrupt/SystemExit
propagate after cleanup. Child sample bars are disabled; `show_progress` renders
completed source counts in the parent. Enabled profiling uses serial execution.

Measure useful preparation work and cheap controls with:

```bash
uv run --python 3.11 --extra dev scripts/bench_writer.py --output plan/evidence/writer-spawn.json
```

The benchmark rotates serial/one/two/four-process cases over five paired runs,
warms each case, verifies physical payload headers and checksums after timing,
and records first source/sample latency and sampled process-tree RSS/threads.
Write time includes pool startup and shutdown. RSS may double-count shared pages;
cheap workloads expose process overhead. T10 measurements are descriptive; final
combined writer/reader acceptance belongs to T12.

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

The pure contiguous-span helper divides rows across ranks and then local workers without batch-size input, padding, or added duplicates. Zero DataLoader workers means one local consumer. For a fixed base seed, reader-seed derivation version 1 packs u32 rank/worker IDs and uses a seed-keyed SplitMix64 permutation; distinct rank/worker pairs cannot collide. The seed is unused for sequential reading. The loader resolves omitted shuffled seeds once during construction and rejects an explicit seed for sequential reading. Rank/group-size capture runs in the training process, with single-rank fallback when no group is initialized; loading workers do not query process groups.

## PyTorch DataLoader helper

`CachedDataset.to_torch_dataloader()` returns a finite PyTorch DataLoader for
training or evaluation. The legacy `to_torch_iterable_dataset()` is unchanged.
Neither shuffling nor worker partitioning implies a validation mode.

`samples_per_epoch` is a **sample count per loader, before batching**. `None`
inherits `len(dataset)` at construction, including any `set_epoch_len()` override.
A positive integer overrides that inherited count for this loader. Sequential
reads wrap when their population is exhausted. Later changes to the source
dataset do not change the loader's snapshotted sample budget.

For sample budget N and batch size B, `len(loader)` is `ceil(N / B)`, or
`floor(N / B)` with `drop_last=True`. `batch_size` must be a positive integer;
`None` is rejected. For example,
10 samples with batch size 3 yields four batches, or three if dropping the last
batch. Workers receive whole-batch quotas and at most one tail across the loader,
so the reported length matches actual iteration regardless of worker count.
Workers with no quota yield nothing. The total budget is shared, never multiplied
by the number of workers.

| Worker partition | `shuffle=False` | `shuffle=True` |
| --- | --- | --- |
| `"split"` (default) | Preserve metadata order, split into disjoint worker populations, read each sequentially | Shuffle metadata in the parent, split it, then sample each worker population by weight with replacement |
| `"replicate"` | Give every worker the full population; workers start at its beginning, so outputs can overlap | Give every worker the full population and independently sample by weight with replacement |

Weights do not affect sequential reads. Randomizing split populations improves
the weight mix but only approximates the global weighted distribution: quotas
follow batch counts, not partition weight sums. Use replicate mode when every
worker must sample the full distribution. With one consumer, both modes use the
full population. Torch interleaves worker batches, so multiple
workers need not emit samples in global metadata order.

`seed` is accepted only with `shuffle=True`; otherwise construction raises
`ValueError`. An explicit seed reproduces fresh loaders with the same topology.
An omitted seed selects randomness once during construction. Split populations
are fixed for the loader's lifetime. Repeated passes continue reader streams and
cursors with zero workers or persistent workers. Nonpersistent workers rebuild
fresh readers each pass and replay explicit seeds. Reconstruct the loader to
change metadata, populations, or seeds.

Python prepares metadata columnarly in the parent in O((rows + workers) × columns)
work, without reading payloads or expanding rows into Python objects. Rust
validates each restored snapshot and owns weighted sampling and bounded reads.
Preparation does not create a consuming native reader. Each consumer creates its
reader once during setup; workers run setup before the caller's `worker_init_fn`.
Empty quotas create no native reader. Initialized readers are PID-bound and
excluded from serialization.

Split adapters retain all prepared IPC slices. Their total parent size is
O(metadata bytes + workers × schema bytes). Spawn/forkserver copies these
inputs into each worker; fork can share immutable bytes through copy-on-write.
Replicate recipes share one IPC value in each process, though native metadata
and cache indexes remain process-local. Account for these costs when sizing
workers. Native payload queues remain bounded by `ReaderConfig.prefetch_size`.

`num_workers` controls Torch processes; `native_num_workers` controls Rust reader
threads per consuming process. Context, pinning, prefetch, timeout, worker
lifetime, and partial-iterator cancellation follow PyTorch semantics. Dropping
or resetting a partial multiprocess iterator can discard prefetched outputs;
consumer prefixes need not be gap-free. Callbacks must be picklable for spawn
and forkserver. Initialization, transform, native-read, worker-death, and timeout
failures propagate.

`sample_transform_fn` receives a native `CachedSample`. Collation always receives
a list, even with `batch_size=1`: transformed samples if a transform is supplied,
otherwise `CachedSample` values. `multiprocessing_context` selects how workers
start: `"fork"`, `"spawn"`, `"forkserver"`, or a `multiprocessing.BaseContext`
object. `None` uses PyTorch's platform default.

```python
# Finite weighted training: 32,000 samples / 32 = 1,000 batches per epoch.
loader = dataset.to_torch_dataloader(
    shuffle=True, seed=123, samples_per_epoch=32_000,
    worker_partition="replicate", batch_size=32,
    num_workers=4, multiprocessing_context="forkserver", persistent_workers=True,
    sample_transform_fn=decode_sample,
)
for batch in loader:
    train_step(batch)

# Sequential evaluation using the dataset's current epoch sample count.
validation_loader = dataset.to_torch_dataloader(
    shuffle=False, worker_partition="split", batch_size=32,
    sample_transform_fn=decode_sample,
)
for batch in validation_loader:
    validate_batch(batch)
```

Review failure conditions with the separate
[inversion checklists](inversion-checklists.md), covering the Python adapters,
native reader/writer pipelines, storage, sampling, and shared runtime primitives.

## CPU DDP composition on macOS

Initialize the application's default distributed group **before** calling
`to_torch_dataloader()` in each rank. The helper captures that group's rank and
size once. Loading workers receive the captured identity, never query the group,
and create their own native reader during worker setup. Rank launch and loader
worker launch are separate choices: the acceptance tests spawn ranks, then use
the caller-selected fork, spawn, or forkserver context for loading workers.

For shuffled training, ranks retain the full active weighted population.
Pass the same explicit base seed to each rank; native seeds are derived from the
base seed, captured rank, and actual worker ID. The finite sample budget is set
with `samples_per_epoch`. Fresh loaders replay explicit seeds; persistent workers
continue their reader state. Metadata changes require a new loader.

For sequential reads, contiguous active-row positions are split across ranks
without padding. Intentional duplicate identities remain separate positions.
Local lengths and batch counts can differ, and a rank can have no samples.
Applications must account for this when scheduling collectives and reducing
metrics. DDP's `join()` context does not cover arbitrary application collectives
or metric reduction.

The self-contained example writes a small temporary fixture cache, initializes
real CPU Gloo ranks, and performs forward/backward/optimizer steps. No manual
cache preparation is required:

```bash
uv run --python 3.11 --extra dev examples/ddp_loading.py --ranks 2 --workers 2 --worker-context fork
uv run --python 3.11 --extra dev examples/ddp_loading.py --ranks 4 --workers 2 --worker-context forkserver --no-shuffle
```

The example's `--gloo-interface lo0` default explicitly selects this Mac's
loopback interface for single-host communication. Automatic interface selection
on the acceptance Mac timed out during DDP construction, before loader use;
explicit loopback selection succeeds. This is an application networking setting,
not a loader context override. See PyTorch's
[Gloo interface configuration](https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.md).
The example also accepts `--quiet` and `--json-output` (one JSON record per rank).

Reader resources replicate across ranks. With R ranks and W loading workers per
rank, there are R × W consuming native datasets (R when W=0), in addition to any
source datasets retained by rank code. Each consuming dataset loads cache
indexes, owns native metadata and `native_num_workers` read threads, and retains
bounded native prefetch state. Torch also prefetches worker batches according to
`prefetch_factor` and `batch_size`; these bounds multiply across ranks/workers.
Sequential IPC-copy costs described above also multiply across ranks. The tests
bound their fixture observations and serialize multi-rank cases across pytest
workers to avoid turning correctness tests into unbounded process fan-out.

Transform failures remain terminal. A failing rank may leave peers waiting for
collectives; application launchers own peer termination. The integration tests
supervise rank sessions with deadlines and terminate their loading workers too.
This evidence covers CPU Gloo on this Mac; it does not establish GPU, FSDP,
DeepSpeed, multi-host, or custom-subgroup behavior.

## Reader performance measurements

`scripts/bench_reader.py` measures `sample_transform_fn` in actual DataLoader
worker processes. It creates immutable fixture caches automatically and compares
direct native iteration, a zero-worker loader, and one/two/four-worker loaders.
The benchmark supports Linux and macOS and defaults to fork; the helper's
own default remains PyTorch's platform default.

```bash
uv run --python 3.11 --extra dev scripts/bench_reader.py --output plan/evidence/reader-fork.json
uv run --python 3.11 --extra dev scripts/bench_reader.py --contexts '["spawn","forkserver"]' --workloads '["cheap"]' --output plan/evidence/reader-startup.json
```

Both shuffle modes run five paired rounds by default, with case order rotated
each round. Each trial runs in a fresh process, constructs its own native source
dataset, warms a native read before worker launch, and consumes a complete first
pass followed by a repeated pass. Persistent workers are enabled in this
benchmark by default; `--no-persistent-workers` measures reconstruction instead.
The first pass supplies warmup and cold-start evidence. Timing gates use the
repeated pass and normalize by actually consumed samples after its first batch.
Construction, iterator creation, first-batch latency, and final cleanup remain
separate fields. Outer trial-process import/startup is outside these timings.

The heavy workload hashes delivered payload bytes in an explicit Python loop;
the cheap control performs only the initial checksum. Every mode uses the same
transform and compact checksum outputs, batch size, useful sample count, and
payload byte count. Sequential passes must end exactly at their assigned total;
shuffled passes stop at the caller's fixed count. Native prefetch and Torch
prefetch may compute additional outputs beyond a stopped shuffled prefix; those
outputs are not counted as accepted work. Weighted physical repeats are normal
sampling behavior. This synthetic workload does not establish image-decoding,
large-tensor IPC, cold-disk, or GPU training throughput.

Acceptance targets remain at least 2× four-worker throughput relative to the
zero-worker loader for the heavy workload, and at most 10% zero-worker serial
regression relative to the direct-native control. The report retains each
paired ratio and its median, reports failed targets without changing thresholds,
and requires at least five pairs before a gate can pass. Small smoke runs check
correctness only. Historical T01 JSON files remain separate; the current gates
use fresh paired controls rather than comparing unrelated historical timings.

Resource observation samples the trial's entire process tree, including context
service processes, every 50 ms and after timed consumption. These are sampled
maxima, not exact peaks. RSS is summed and can count shared fork pages more than
once. Thread counts include the benchmark observer. Native queues, Torch result
prefetch, cache indexes, metadata, and worker memory still multiply according to
the reader budgets described above. The benchmark retains only bounded output
batches, a bounded set of consumer PIDs, and per-trial measurement records; it
does not preload the payload population.

JSON records identify framework versions, Git revision, dirty-worktree state,
and source fingerprints. Reader or benchmark source changes during a matrix
invalidate that run. Trial timeouts and interruption terminate the benchmark's
owned process session, including its loading workers. `--quiet` suppresses
progress while retaining JSON output. Timing targets are not pytest assertions;
default validation runs bounded benchmark correctness smoke tests.

## Ordering

Output order is deterministic. Workers may complete out of order, but the reorder stage publishes samples in the sampler's planned order.

With `shuffle=False`, the plan is active metadata table order. With `shuffle=True`, the plan comes from deterministic weighted sampling over active metadata rows.

## Iterator Snapshots

Each iterator snapshots:

- Cache manifests.
- Current active metadata table.
- Seed and epoch number when `shuffle=True`.

Metadata changes made after iterator construction do not affect that iterator.
