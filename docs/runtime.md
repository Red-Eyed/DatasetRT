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

Each iterator keeps at most `min(num_workers, prefetch_size)` scheduled but
unconsumed samples across active reads, queued results, and the reorder buffer.
Out-of-order completion does not free a scheduling slot: the next consumer call
replenishes slots released by emitted samples. A delayed early read or slow Python
consumption therefore cannot grow the buffered sample count beyond this window.

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

Python keeps an immutable internal recipe containing original ordered cache paths, reader settings, the configured sample count, and either original-cache metadata or accepted Arrow metadata. Recipe export does not create a native reader, read payloads, or export a default full metadata table. It does not copy an active iterator's cursor.

Metadata updates retain shared immutable Arrow buffers only after Rust accepts them. Failed updates preserve the previous recipe; prior exported recipes remain immutable snapshots. Python and Rust share the column buffers instead of retaining separate IPC copies. Default datasets retain only O(cache paths) construction inputs, with no eager metadata export or per-row Python objects.

Public reconstruction uses `dataset.dump_config(existing_directory)` to write
`dataset.json` and `metadata.arrow`, returning the config FilePath.
`CachedDataset.from_config(config_path)` creates a fresh runtime itself. Exports
preserve active order, duplicates, weights, extra columns, reader settings,
native thread count, and epoch length. Cache paths are absolute references;
payloads and cursor progress are not copied. Keep exports immutable while used.
Existing files are never overwritten. Configuration is published last.

Workers receive a config path and compact plans, map the same immutable Arrow
buffers, and pass row-position views to Rust without gathering metadata columns.
A shuffled split creates one memory-mapped permutation per loader, shared by its
workers. Cache indexes map their original immutable files; selected physical
indices, weights, and sampling state remain process-local. This avoids inherited Polars thread
pools in forked children. Rust validates selected metadata and owns sampling and
reads. One common seed determines partition ordering; distinct derived seeds
identify native sampling streams. Empty quotas create no native state.

### How metadata sharing uses mmap

Metadata sharing uses read-only, file-backed `mmap`. Each loader writes one
immutable `metadata.arrow` snapshot; it does not allocate a named shared-memory
segment. Workers receive the path to `dataset.json`, read its `metadata_file`
field, and resolve that filename relative to the config directory. Each worker
then maps the same file independently. This works with `spawn`, `fork`, and
`forkserver` without relying on inherited mappings or matching virtual addresses.

The OS can share physical file-cache pages between processes on the same host
mapping the same snapshot. Arrow buffers reference those mapped pages, and the
Arrow C stream bridge transfers buffer ownership to Rust without copying column
contents. Retained owners keep mappings alive while their arrays are in use.
Pages can be loaded on demand and reclaimed by the OS; mapping a file does not
require loading its full contents into a private heap allocation.

For shuffled splits, the parent also creates `partition-order.npy` beside the
config. Workers map that file read-only and use their assigned row positions.
Cache indexes also use read-only mmap of each cache's original `index.bin`.
Independent loaders on one host share those pages without exporting a second
index file. Selected physical indices, weights, and sampling arrays remain
process-local. Separate metadata exports have separate backing files, and processes
on different hosts do not share physical memory. Explicit DataFrame operations
or selected-table exports may still allocate new buffers.

### Index memory comparison

A local macOS arm64 check on 2026-10-12 compared decoded indexes with mapped
indexes using the same two caches, each containing 500,000 one-byte samples.
Fresh spawned processes each constructed a non-shuffled reader with one native
thread, read one sample, and stayed alive until all readers were ready. The
measurement sums `psutil.memory_full_info().uss` (private resident memory) across
consumers; it excludes the parent and is a steady-state observation, not peak RAM.
Both builds already included the Arrow metadata sharing change.

| Consumers | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 84.1 MiB | 72.5 MiB | 13.9% |
| 4 | 340.2 MiB | 197.8 MiB | 41.9% |

The one-reader measurement counts resident mapped index pages as unique because
no other process maps them. With multiple readers, those pages are shared.
The change also removes transient file reads and decoded-table allocations;
allocator retention affects observed savings. These figures include Python and
sampling state and do not predict total training RAM. Cache index storage is
24 bytes per sample regardless of consumer count, while private weights and
selected physical indices still scale with consumers.

## PyTorch DataLoader helper

Loader defaults retain persistent workers and automatically budget native reader
threads. Both worker counts default to the explicit `"auto"` mode; `None` is
rejected. `num_workers="auto"` selects `min(4, max(0, CPUs per local rank - 1))`
loading processes, while explicit zero retains main-process loading and supports
local callbacks. Auto loading may launch processes, so transforms must be picklable
when the resolved count is positive. Explicit topology is preferable for
cross-machine reproducibility. Both automatic counts share one budget reading.
`native_num_workers="auto"` then selects 1–4 threads per consumer from the
minimum of host CPUs, process affinity, and visible Linux cgroup v1/v2 CPU quotas
and CPU-set constraints. Detection follows `/proc/self/cgroup` and mountinfo,
checks visible ancestors, and supports remounted controller roots. Fractional
quotas are rounded down with a one-thread minimum; malformed or unreadable limits
warn and fall back to one. Hidden ancestor limits cannot be inferred.
The reusable `dataset_rt.resources.available_cpu_count()` helper reports that
whole-CPU budget without importing Torch.

The budget is divided by `max(1, num_workers)` and the local training-rank count
from `LOCAL_WORLD_SIZE`. Without launcher information, distributed world size
is a conservative fallback. This may underuse separately pinned rank allocations.
It cannot reserve CPUs from unrelated workloads, account for Python transform or
training threads, or keep total native threads below the budget when consumers
already outnumber CPUs. An explicit integer overrides automatic selection.
For example, eight consumers sharing 32 CPUs receive four native threads each;
a Docker quota of two CPUs reduces them to one thread each.

`seed="random"` chooses randomness once per loader construction. An integer seed
reproduces fresh loaders with the same topology. Persistent readers continue
their stream across passes; reconstructed nonpersistent readers still reset it.
Sequential reads accept the default random mode but reject explicit integer seeds.

`worker_partition=None` selects replicate for shuffled weighted training and
split for sequential reading. Replicate preserves full-population weighted
sampling but retains private weights and sampling arrays in every consumer.
Ten million f64 weights alone use 80 MB per consumer. Explicit split remains
available as a memory tradeoff and can change global weighted frequencies.

`CachedDataset.to_torch_dataloader()` returns a finite PyTorch DataLoader for
training or evaluation. The legacy `to_torch_iterable_dataset()` is unchanged.
Neither shuffling nor worker partitioning implies a validation mode.

`samples_per_epoch` is a **global sample count before batching**. `None`
inherits `len(dataset)` at construction, including any `set_epoch_len()` override.
A positive integer overrides that inherited count for this loader. Sequential
reads wrap when their population is exhausted. Later changes to the source
dataset do not change the loader's snapshotted sample budget.

With one training process, for sample budget N and batch size B, `len(loader)` is `ceil(N / B)`, or
`floor(N / B)` with `drop_last=True`. `batch_size` must be a positive integer;
`None` is rejected. For example,
10 samples with batch size 3 yields four batches, or three if dropping the last
batch. Workers receive whole-batch quotas and at most one tail across the loader,
so the reported length matches actual iteration regardless of worker count.
Workers with no quota yield nothing. The total budget is shared, never multiplied
by the number of workers.

| Worker partition | `shuffle=False` | `shuffle=True` |
| --- | --- | --- |
| `"split"` (default) | Preserve metadata order, split into disjoint worker populations, read each sequentially | Reproduce a common seeded metadata order, split it, then sample each worker population by weight with replacement |
| `"replicate"` | Give every worker the full population; workers start at its beginning, so outputs can overlap | Give every worker the full population and independently sample by weight with replacement |

Weights do not affect sequential reads. Randomizing split populations improves
the weight mix but only approximates the global weighted distribution: quotas
follow batch counts, not partition weight sums. Use replicate mode when every
worker must sample the full distribution. With one training process and one consumer, both modes use the
full population. Torch interleaves worker batches, so multiple
workers need not emit samples in global metadata order.

`seed` is accepted only with `shuffle=True`; otherwise construction raises
`ValueError`. An explicit seed reproduces fresh loaders with the same topology.
An omitted seed selects randomness once during construction. Split populations
are fixed for the loader's lifetime. Repeated passes continue reader streams and
cursors with zero workers or persistent workers. Nonpersistent workers rebuild
fresh readers each pass and replay explicit seeds. Reconstruct the loader to
change metadata, populations, or seeds.

Construction exports active metadata once per loader, in O(rows × columns)
work, without payload loading or Python row expansion. Plan construction is
O(workers); each serialized adapter is O(workers + config path), independent
of metadata width. Sending all plans to every worker is O(workers²) small
descriptors in aggregate. Native setup happens on consumption or before `worker_init_fn`
and is reused in its process. Native handles are excluded from serialization.

`work_dir=None` creates a managed temporary artifact directory. The parent loader
and active iterators retain its owner; it is removed when their dataset is
released. An existing DirectoryPath creates an isolated subdirectory retained for
inspection or reconstruction. Concurrent loaders never overwrite each other.
Keep snapshots immutable and accessible to all consumers. Hard parent termination
can leave temporary artifacts behind.

Metadata column buffers are mapped from one immutable Arrow snapshot and shared
by consumers on the same host. Rust retains their owners and validates selected
identities and weights without copying metadata columns or serializing them into
IPC bytes. Original cache export uses mapped input and chunks of at most 65,536
rows, bounding transient serialization memory by chunk metadata size. Shuffled
splits share one mapped O(rows) permutation instead of rebuilding it per worker.
Metadata view descriptors scale with record batches, and selected validation is
O(selected rows × log(record batches)); it does not scan all physical samples
for each row. Cache indexes share mapped pages from their original files.
Selected physical indices, weights, and sampling state remain process-local.
Replicate mode retains full sampling arrays in each
consumer. Explicitly requesting a selected metadata DataFrame gathers its columns.
Native payload queues remain bounded by `ReaderConfig.prefetch_size`; Torch
batch prefetch multiplies by worker count. New config bundles use version 2 and
Arrow only; old Parquet exports must be recreated.

`num_workers` controls Torch processes; `native_num_workers` controls Rust reader
threads per consuming process. Context, pinning, prefetch, timeout, worker
lifetime, and partial-iterator cancellation follow PyTorch semantics. Dropping
or resetting a partial multiprocess iterator can discard prefetched outputs;
consumer prefixes need not be gap-free. With workers, `sample_transform_fn` is checked for picklability before export,
including fork. Other callbacks must be picklable for spawn/forkserver. Successful
serialization does not guarantee child imports succeed. Initialization, transform, native-read, worker-death, and timeout
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

The budget N is global, with the same local batch size B on every rank.
Without `drop_last`, each rank gets `ceil(N/R)` samples; with `drop_last`, each gets
`B*floor(N/(R*B))`. All ranks report identical lengths and emit identical batch
counts. N=1000, R=3, B=32 gives 334 samples/11 batches per rank, or 320 samples/10
batches with `drop_last=True`. Rank zero warns when padding or discarding changes
the effective global count. Zero effective budgets create no native readers.
Padded split reads repeat active positions; account for them when reducing metrics.

Both partition policies apply across ranks and workers. Split divides active
positions; replicate gives every consumer the full population. Weighted draws
use replacement. Explicit seeds reproduce the same topology. Without a seed,
training ranks broadcast a common random partition seed during construction.
Construct loaders in the same order with consistent budgets, batch sizes,
metadata, and shuffle settings on every rank. Persistent readers continue
streams; changed metadata requires a new loader. Applications own group setup,
metric reduction, and peer termination after failures.

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
source datasets retained by rank code. Each consuming dataset maps cache
indexes, retains metadata owners and `native_num_workers` read threads, and retains
bounded native prefetch state. Torch also prefetches worker batches according to
`prefetch_factor` and `batch_size`; these bounds multiply across ranks/workers.
Metadata pages are shared for readers mapping the same snapshot on one host;
cache index pages are also shared on one host, while private sampling arrays
still multiply across ranks. Separate loader
exports create separate mapped snapshots. The tests
bound their fixture observations and serialize multi-rank cases across pytest
workers to avoid turning correctness tests into unbounded process fan-out.

Transform failures remain terminal. A failing rank may leave peers waiting for
collectives; application launchers own peer termination. The integration tests
supervise rank sessions with deadlines and terminate their loading workers too.
This evidence covers CPU Gloo on this Mac; it does not establish GPU, FSDP,
DeepSpeed, multi-host, or custom-subgroup behavior.

## Reader performance measurements

For portable CPU/GPU and multi-host commands, see
[remote validation](remote-validation.md). The package entry point
`python -m dataset_rt.run_benchmark --ranks 4 --num-workers 8 --device cuda`
launches four local training ranks and eight loading workers per rank, validates
finite counts and replay, and saves JSON timing/resource results.

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
prefetch and private worker state multiply according to the reader budgets
described above; summed RSS also counts shared index and metadata pages.
The benchmark retains only bounded output
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
