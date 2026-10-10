# Inversion review checklists

Start with “How could this abstraction fail?” rather than “Does the happy path
work?” For each relevant item, construct a counterexample and record the source,
test, measurement, or documented limitation that answers it. Leave unsupported
claims unchecked. These are review templates, not a record of completed checks.

The [inversion analysis](inversion-analysis.md) explains the architectural
principle. This document turns it into checks for the Python API, Rust core, and
their boundaries. Sections group related types by the contract they share;
private implementation helpers belong to their owning abstraction. When adding
an abstraction, add its failure conditions here.

For example, ten epoch samples with batch size three imply four emitted batches,
or three with `drop_last=True`. An inversion asks what could break that promise:
each worker receiving ten samples, each worker dropping its own tail, or an
inherited epoch override being mistaken for the population size.

## Checks across all abstractions

- [ ] State the observable contract, its owner, and how failure reaches callers.
  Distinguish expected errors, interrupts, worker failures, and process death.
- [ ] Try empty inputs, smallest valid inputs, invalid values, numeric limits,
  malformed external data, and failure after partial progress.
- [ ] Account for memory and work in samples, caches, shards, metadata columns,
  threads, and processes. Separate fixed bounds from input-sized allocations.
- [ ] Check cancellation and lifetime boundaries: who owns resources, who wakes
  blocked work, and what survives failure or an interrupted caller?
- [ ] Verify deterministic ordering and stable identity where promised. Separate
  intentional repeated samples from accidental duplication or lost work.
- [ ] Reject Python row-object expansion, internal temporary-file transport,
  accidental quadratic lookup, and unbounded queues on scalable paths.
- [ ] Make examples and types agree with behavior. Describe ignored or advisory
  settings explicitly; avoid making an unrelated flag select a hidden mode.

## Python configuration and native validated types

Sources: [config.py](../dataset_rt/config.py),
[types.rs](../src/dataset_rt/types.rs).

Failure to prevent: an apparently valid configuration reaches native work with
an impossible value or silently changes meaning at the language boundary.

- [ ] Exercise zero, negatives, booleans used as integers, nonintegral numbers,
  NaN, infinity, and values beyond the target integer representation.
- [ ] Verify Rust validates correctness-sensitive values independently of Python
  validation, including calls made directly through the extension.
- [ ] Check combinations: compression algorithm and ratio, zero-worker options,
  profiling and source processes, and context-dependent timeout semantics.
- [ ] Keep units explicit: bytes, samples, queue capacities, threads, processes,
  and seconds must not be interchangeable.
- [ ] Frozen Python settings and native newtypes must preserve their invariants
  through reconstruction and must not conceal unsupported settings.

## Source and payload records

Sources: [records.py](../dataset_rt/records.py),
[writer.rs](../src/dataset_rt/writer.rs).

Failure to prevent: source objects or domain serialization corrupt the cache
contract or escape the owning process.

- [ ] Reject unsupported payload objects and metadata values before publication;
  exercise bytes, bytearray, memoryview, and changing source schemas.
- [ ] Establish when mutable input buffers become owned bytes; later source
  mutation must not alter queued or committed payloads.
- [ ] Treat source iteration, name access, serialization, and deserialization as
  possible failure points, including KeyboardInterrupt and SystemExit.
- [ ] Keep domain decoding outside Rust. Returned bytes and physical IDs must
  describe the stored sample, independent of active-table filtering.
- [ ] Spawned source descriptors are importable and picklable; files and live
  connections are opened in the consuming process rather than inherited.

## Results and boundary errors

Sources: [records.py](../dataset_rt/records.py),
[runtime.py](../dataset_rt/runtime.py), [types.rs](../src/dataset_rt/types.rs).

Failure to prevent: a caller interprets an error as proof that nothing happened,
or a boundary loses the identity and extent of partial success.

- [ ] Preserve source/path identity, diagnostics, and input ordering when
  converting native outcomes into Python result records.
- [ ] Distinguish no target from a known failing target, and retain completed
  outcomes when later work fails.
- [ ] Do not report successful publication as rolled back when outcome delivery
  fails, or a replaced manifest as unchanged when directory sync fails.
- [ ] Expected malformed inputs return typed native errors rather than panics;
  Python interrupts keep their interruption semantics after cleanup.

## DatasetRuntime and runtime construction

Sources: [runtime.py](../dataset_rt/runtime.py),
[dataset_runtime.rs](../src/dataset_rt/dataset_runtime.rs).

Failure to prevent: a thin factory creates hidden pools, imports optional
frameworks, or reconstructs a different dataset than the caller requested.

- [ ] Pool sizes are explicit; operations reuse the runtime's pool instead of
  creating threads proportional to samples or cache count.
- [ ] Test empty path/source inputs, failures among successful sources, and
  existing-cache reuse without hiding unsuccessful outcomes.
- [ ] Preserve supplied cache order and accepted reader settings through factory
  composition; do not silently discard a source or substitute a cache.
- [ ] Importing the package does not require Torch. Native handles stay in their
  creating process; reconstruction uses immutable inputs and fresh runtimes.

## CachedDataset and iterator snapshots

Sources: [dataset.py](../dataset_rt/dataset.py),
[dataset.rs](../src/dataset_rt/dataset.rs).

Failure to prevent: changing future dataset state alters an existing iterator,
or epoch length is confused with the sampling population.

- [ ] Capture an iterator, then change metadata and epoch length: the captured
  iterator retains its snapshot and future iterators adopt accepted changes.
- [ ] Exercise sequential wraparound, short windows, long windows, and multiple
  live iterators. Confirm when cursor/draw ranges are reserved, including when
  an iterator is abandoned before consuming its whole window.
- [ ] Metadata updates reset the stream and epoch length as documented;
  `set_epoch_len` changes the finite window without changing the population.
- [ ] Failed updates leave native state and the Python reconstruction recipe
  consistent. Public mutable path lists must not change frozen reconstruction.
- [ ] Concurrent operations cannot observe partially replaced metadata, weights,
  epoch length, or cursor state.

## Physical identity, cache lookup, and direct access

Sources: [dataset.rs](../src/dataset_rt/dataset.rs),
[runtime.rs](../src/dataset_rt/runtime.rs), [types.rs](../src/dataset_rt/types.rs).

Failure to prevent: a public cache ID is treated as an array position, or an
active-row position is mistaken for a physical sample ID.

- [ ] Use noncontiguous v3 IDs and reordered cache paths. Resolve IDs through the
  lookup rather than indexing a cache array with the public ID.
- [ ] Reject duplicate physical cache IDs, unknown cache IDs, and out-of-range
  sample IDs; preserve positional legacy v2 identity until explicit migration.
- [ ] Direct reads return immutable physical samples even when active metadata
  filters, reorders, or duplicates their identities, without advancing cursors.
- [ ] Check identity lookup and cache-offset arithmetic for overflow and missing
  entries; many-cache workloads must not scan all samples per lookup.

## Active metadata and authoritative weights

Sources: [samples_metadata.rs](../src/dataset_rt/samples_metadata.rs),
[dataset.rs](../src/dataset_rt/dataset.rs), [dataset.py](../dataset_rt/dataset.py).

Failure to prevent: valid-looking columnar edits select the wrong samples or
partially mutate authoritative state.

- [ ] Reject missing required columns, incompatible Arrow types, null identity
  or weight cells, unknown identities, and nonpositive or nonfinite weights.
- [ ] Validate the whole update before committing it. A bad final row must leave
  the previous table, weights, epoch budget, and recipe intact.
- [ ] Preserve filtering, table order, duplicate active rows, and extra columns.
  Duplicate active rows are allowed; they are not duplicate cache identities.
- [ ] Verify each active row's weight follows that row after reorder or
  duplication. Repeated identities contribute separate sampling entries.
- [ ] Editing an exported table alone does not mutate the dataset. Runtime
  updates do not rewrite immutable cache metadata or payload files.
- [ ] Measure export, validation, and identity resolution on many caches and
  rows; keep columns columnar and avoid per-row Python callbacks.

## In-memory metadata IPC bridge

Sources: [metadata.py](../dataset_rt/metadata.py),
[samples_metadata.rs](../src/dataset_rt/samples_metadata.rs).

Failure to prevent: crossing the Python/Rust boundary loses schema or expands
dataset-scale metadata into row objects.

- [ ] Round-trip integer widths, floating weights, extra columns, column order,
  duplicated rows, and sliced frames without changing accepted semantics.
- [ ] Truncated IPC, malformed schemas, and invalid required cells fail at the
  native validation boundary before authoritative state changes.
- [ ] Account for IPC buffers, decoded frames, and retained snapshots together;
  an in-memory bridge can still make multiple full-table copies.
- [ ] Forked consumers restore accepted IPC directly in Rust rather than using
  inherited Python columnar thread-pool state.

## ReaderRecipe, MetadataSnapshot, and RowSpan

Source: [records.py](../dataset_rt/records.py).

Failure to prevent: reconstruction transports live state or recreates a reader
with different cache identity, population, or length.

- [ ] Recipes contain construction inputs, not native handles, open iterators,
  synchronization objects, or a hidden mutable cursor.
- [ ] Preserve accepted cache order, metadata snapshot, reader settings, and
  finite sample count through serialization.
- [ ] Distinguish original metadata from an accepted override; an unaccepted
  candidate table must never enter the recipe.
- [ ] Row spans partition active positions with valid bounds, including empty
  slices; slicing must never renumber physical sample identities.

## EpochSampler and EpochPlan

Sources: [sampling.rs](../src/dataset_rt/sampling.rs),
[runtime.rs](../src/dataset_rt/runtime.rs).

Failure to prevent: sampling changes distribution or sequence when a finite
window is split, or planning allocates an epoch-sized task list unnecessarily.

- [ ] Same population, weights, seed, and draw range reproduce the same sequence;
  split finite windows reproduce the corresponding uninterrupted draw stream.
- [ ] Weighted draws use replacement. Test extreme valid weights and reject
  invalid distributions, including failures while constructing cumulative sums.
- [ ] Uniform defaults avoid a full all-ones weight vector; sequential plans
  avoid materializing every physical position in an epoch.
- [ ] Exercise draw-index and cursor arithmetic near numeric limits; overflow
  must not silently repeat draws or address the wrong row.
- [ ] Active indices map planned positions to physical samples correctly for
  filtered, reordered, and duplicated metadata.

## RuntimeIterator and ordered read scheduling

Source: [runtime.rs](../src/dataset_rt/runtime.rs).

Failure to prevent: slow or failed reads reorder output, deadlock shared workers,
or let completed results grow without bound.

- [ ] Deliberately complete reads out of order and verify planned output order.
  A slow first read must not permit an unbounded reorder buffer.
- [ ] Reserve result capacity before submitting jobs; check active reads and
  buffered results against the operation window under a slow consumer.
- [ ] Dropping an iterator stops future submissions, disconnects result delivery,
  and lets outstanding jobs release resources without trapping pool threads.
- [ ] Corrupt records and worker failures reach the caller at the documented
  boundary without skipped samples or indefinite waits.
- [ ] Account for maximum loaded sample size and transient read/decompression
  buffers; a count-bounded queue alone does not bound bytes for arbitrary inputs.

## WorkerPool

Source: [worker_pool.rs](../src/dataset_rt/worker_pool.rs).

Failure to prevent: one operation monopolizes the shared pool or leaves a caller
waiting forever after a job fails.

- [ ] Test thread-creation failure, submission after disconnection, and job
  unwinding; accepted jobs deliver a result or encounter a cancelled receiver.
- [ ] Multiple reader/writer operations cannot deadlock workers on full result
  queues while callers block submitting more jobs.
- [ ] Queued jobs remain bounded, worker count stays fixed, and shutdown releases
  queued closures and captured resources.
- [ ] Runtime ownership and process boundaries are explicit; a freshly created
  child pool must not reuse inherited native synchronization state.

## Bounded channels

Source: [channel.rs](../src/dataset_rt/channel.rs).

Failure to prevent: lost wakeups, premature disconnection, or resource destructors
running under a lock prevent progress.

- [ ] Zero capacity is rejected; full queues block producers without exceeding
  capacity, and receive order preserves the queue's FIFO contract.
- [ ] Last-sender drop wakes receivers and allows queued values to drain;
  last-receiver drop wakes blocked senders and disconnects delivery.
- [ ] Cloned endpoints share one queue and keep it connected until the last owner
  drops. Concurrent transport neither loses nor duplicates values.
- [ ] Timeouts do not close the channel or restart the deadline after every
  wakeup; waits tolerate spurious wakeups.
- [ ] Queued payload destruction happens outside the queue lock. Check teardown
  with reentrant or slow destructors and previously active parent runtimes.

## CacheBuilder and native writer pipelines

Sources: [storage.rs](../src/dataset_rt/storage.rs),
[writer.rs](../src/dataset_rt/writer.rs),
[writer/pipeline.rs](../src/dataset_rt/writer/pipeline.rs).

Failure to prevent: parallel serialization changes physical sample order or
publishes an incomplete cache after an ingestion or disk error.

- [ ] Out-of-order worker completion still commits samples, metadata, indexes,
  checksums, and source outcomes in their required order.
- [ ] Slow commit applies backpressure before ingestion accumulates an unbounded
  number of Python inputs or serialized records.
- [ ] Source iteration errors, compression failures, disk-full conditions, and
  interruption leave no newly published incomplete cache.
- [ ] Shard rotation handles empty and oversized records with documented target
  size semantics; offsets and byte lengths use checked arithmetic.
- [ ] Finish flushes and durably completes required files before publication;
  cleanup removes only temporary files owned by this operation.
- [ ] Existing-cache reuse honors requested validation without rewriting a
  completed cache or bypassing required shape checks.

## Spawned source-writer supervision

Source: [writer.py](../dataset_rt/writer.py).

Failure to prevent: multiple children own the same destination, a failed child
strands resources, or a retry repeats external source side effects.

- [ ] Preflight rejects duplicate, case-folded, Unicode-normalized, and reserved
  destination names before starting children; existing foreign temporary paths
  are not adopted or deleted.
- [ ] Each source has one owner and each child at most one active job; transport
  contains descriptors and outcomes rather than per-sample payloads.
- [ ] Test serialization failure, startup failure, death before outcome delivery,
  timeout, malformed messages, and parent interruption. Retain completed results
  in input order and do not automatically retry source side effects.
- [ ] Stop a child before deleting its temporary output. Shutdown joins are
  bounded and escalate termination without deleting published caches.
- [ ] Symlinks, resolved temporary paths, and same-filesystem publication retain
  explicit ownership. Hard process termination may leave inspectable artifacts.
- [ ] Account for aggregate threads, buffers, descriptors, and outcomes across
  all children; document timeout exclusions and source-created subprocesses.

## Manifest, index, shard, and LoadedCache validation

Sources: [storage.rs](../src/dataset_rt/storage.rs),
[storage/manifest.rs](../src/dataset_rt/storage/manifest.rs).

Failure to prevent: plausible metadata makes a missing, incompatible, or corrupt
cache appear complete, or ordinary startup reads all payloads.

- [ ] Missing or malformed manifests, unsupported versions, inconsistent schemas,
  duplicate reserved fields, and mismatched counts fail clearly.
- [ ] Reject malformed index lengths, invalid shard references, overflowing or
  out-of-file record ranges, and missing shard files before unsafe access.
- [ ] Check metadata schema and row-count agreement with the manifest and index;
  redundant embedded metadata validation must follow its documented policy.
- [ ] Required shape validation remains active when full checksum validation is
  disabled. Checksum failures are detected when `validate_cache=True` requests it.
- [ ] Default construction avoids hashing all shards or eagerly expanding all
  metadata cells. Account explicitly for necessary manifest and index storage.
- [ ] Validate on-disk names and paths so records cannot redirect reads or writes
  outside their intended cache directories.

## ShardReaderCache and record access

Source: [storage.rs](../src/dataset_rt/storage.rs).

Failure to prevent: random access exhausts file descriptors, reuses a stale
handle, or trusts corrupt lengths before allocating memory.

- [ ] Many-cache/shard workloads keep open handles within the configured cache
  bound and release evicted handles.
- [ ] Repeated random reads seek correctly and return the addressed record;
  worker-local readers must not share a mutable file cursor accidentally.
- [ ] Truncated metadata envelopes and payloads fail without unchecked slicing;
  validate lengths and range arithmetic before allocating record buffers.
- [ ] Thread-local handle retention has an explicit lifetime; cache immutability
  and replacement assumptions must not conceal stale reads.

## Per-record compression

Sources: [compression.rs](../src/dataset_rt/compression.rs),
[types.rs](../src/dataset_rt/types.rs).

Failure to prevent: compression changes payload meaning or corrupt size headers
cause excessive allocation before an error.

- [ ] Round-trip empty, incompressible, repetitive, and large payloads for every
  supported algorithm; compression is per record and preserves random access.
- [ ] Unknown algorithms, truncated envelopes, malformed LZ4 data, and impossible
  decoded lengths fail clearly.
- [ ] Compression policy and manifest agree; advisory ratio metadata must not
  become a correctness assumption about actual compressed sizes.
- [ ] Include stored record buffers and decoded buffers in peak memory estimates;
  verify how untrusted size headers are bounded before allocation.

## Manifest migration and update reports

Sources: [storage/migration.rs](../src/dataset_rt/storage/migration.rs),
[storage/manifest.rs](../src/dataset_rt/storage/manifest.rs),
[records.py](../dataset_rt/records.py).

Failure to prevent: an explicit identity-preserving upgrade renumbers samples,
partially rewrites payload files, or misreports persistence after a failure.

- [ ] Preflight all targets before replacement, reject conflicting identities,
  and retain resolved legacy IDs rather than recomputing name-derived IDs.
- [ ] Update only manifests; indexes, shards, active metadata, and live iterator
  state remain unchanged. Already-upgraded targets are idempotent.
- [ ] Inject failures before write, before rename, and after rename during
  directory sync. Report completed and durability-uncertain entries accurately.
- [ ] Partial updates are retryable in the original legacy cache order; do not
  imply an atomic transaction across multiple cache directories.
- [ ] Require exclusive caller ownership for mutation and remove only owned
  temporary manifests; use no speculative lock files.

## Writer profiling and progress

Sources: [writer/profiler.rs](../src/dataset_rt/writer/profiler.rs),
[writer/progress.rs](../src/dataset_rt/writer/progress.rs).

Failure to prevent: optional observability changes cache correctness, blocks
cancellation, or reports ingestion as durable completion.

- [ ] Disabled profiling avoids collecting detailed timing state or producing
  artifacts. Progress settings do not change sample order or results.
- [ ] Rates and totals distinguish queued, committed, and published work;
  empty sources and zero elapsed time do not produce invalid statistics.
- [ ] Profiling output failure and interrupted writes follow the documented error
  policy without concealing the cache's actual publication state.
- [ ] Source-level statistics stay bounded by source/stage count, and child
  progress output does not compete with the parent's display.

## PyO3 surface, stubs, and public Python facade

Sources: [lib.rs](../src/dataset_rt/lib.rs),
[_dataset_rt.pyi](../dataset_rt/_dataset_rt.pyi),
[api.py](../dataset_rt/api.py), [__init__.py](../dataset_rt/__init__.py).

Failure to prevent: runtime behavior, static types, pickle paths, and public
imports describe different interfaces.

- [ ] Changed native signatures, return shapes, and errors are reflected in
  stubs and wrapper conversions; direct extension calls preserve validation.
- [ ] Public classes keep supported import and pickle identities. Implementation
  modules do not depend on the public facade and create import cycles.
- [ ] GIL release, Python callbacks, and signal checks preserve object ownership
  and responsiveness without introducing hidden Python sample queues.
- [ ] Generated API documentation and examples match accepted values, units,
  collation output, and mutation/lifetime behavior.

## Legacy Torch iterable view

Source: [integrations/torch.py](../dataset_rt/integrations/torch.py).

Failure to prevent: a wrapper around a live dataset is mistakenly used as a
multiprocess reconstruction adapter.

- [ ] The view delegates current dataset length and iteration without changing
  sample types or sampling semantics.
- [ ] Worker use fails clearly rather than consuming inherited native state or
  silently duplicating a parent reader.
- [ ] Missing Torch raises a focused optional-dependency error only when the
  integration is requested.

## DataLoader orchestration and worker budgets

Sources: [integrations/loader.py](../dataset_rt/integrations/loader.py),
[integrations/loading.py](../dataset_rt/integrations/loading.py),
[dataset.py](../dataset_rt/dataset.py).

Failure to prevent: independent workers multiply the epoch budget, alter the
promised population, or retain native state from another process. These checks
cover ordinary DataLoader workers; distributed execution is deferred.

- [ ] Invalid options fail before reading/serializing metadata: worker counts,
  batch size, sample count, context, persistence, timeout, prefetch, and seed.
  Finite counts must fit Python's length protocol and native integer bounds.
- [ ] Inherit `len(dataset)`, including `set_epoch_len`, once at construction;
  explicit budgets override it locally. Later source edits require rebuilding.
- [ ] Compare `len(loader)` with emitted batches for short/exact/long windows,
  uneven tails, excess workers, both partition modes, and both `drop_last`
  settings. Allocate whole batches and at most one tail across one total budget.
- [ ] Sequential reads randomize neither metadata nor native output. Explain
  wraparound, replicated overlap, and worker interleaving rather than promising
  global order or unique physical coverage.
- [ ] Reading every active row once requires a complete sequential split pass,
  a budget equal to the active row count, and no dropped tail. Active duplicate
  rows still intentionally repeat their physical sample.
- [ ] Split shuffling happens in the parent before slicing. Unequal-weight tests
  expose the difference between per-partition and global weighted distributions;
  shuffling does not make fixed worker quotas globally exact.
- [ ] Replicated workers receive the full population but share the epoch budget;
  weighted sampling uses replacement and is not a shuffled permutation.
- [ ] State reproducibility conditions: fresh loaders, same seed and topology,
  persistent versus nonpersistent workers, and partial passes with discarded
  prefetch. Repeated passes need not replay persistent reader streams.
- [ ] Setup creates native readers in the consumer before the user's initializer;
  zero-quota workers create none. PID mismatches and unexpected worker topology
  fail explicitly; serialization excludes initialized readers.
- [ ] Include every prepared IPC slice, native metadata/index copy, Torch batch
  prefetch, and native read buffer in process-tree memory. Spawn currently sends
  all split recipes to every worker, so aggregate metadata can grow with workers.
- [ ] Collation receives lists even for batch size one; transformed samples and
  custom batch return types agree with overloads and examples.
- [ ] Initialization, transform/read errors, timeout, worker death, interruption,
  and partial teardown propagate or clean up under supported start contexts.

### Completed single-GPU loader review

The evidence below covers one training process with zero or multiple local
DataLoader workers. It exercises the adapter using real native caches and Torch
workers; it does not establish GPU-kernel, distributed, or multi-host behavior.
The checklists for other abstractions remain independent review templates.

Evidence comes from [serial loader tests](../tests/integrations/test_serial_loader.py),
[parallel loader tests](../tests/integrations/test_parallel_loader.py), the native
queue/runtime checks, and [runtime contracts](runtime.md#pytorch-dataloader-helper).

| Checklist condition | Evidence and outcome |
| --- | --- |
| Reject invalid options before metadata work | `test_invalid_workers_fail_before_metadata`, `test_invalid_sample_budget`, and `test_boolean_options_reject_truthy_integers` make preparation fail if reached. Invalid persistence, contexts, prefetch, timeout, flags, and numeric bounds are rejected first. |
| Inherit the finite budget once | `test_inherited_and_explicit_epoch_budgets`, `test_sequential_inherits_source_epoch_length`, and `test_inherited_budget_limit` cover inheritance, local overrides, source mutation, and limits. |
| Match reported and emitted batches | Serial short/long-window tests and real-worker batch-tail, batch-drop, quota-short, quota-long, empty, and tiny-replicate cases cover whole-batch quotas and one shared tail. Numeric-limit tests check reported length without consuming a huge epoch. |
| Preserve sequential semantics | Serial order/wrap tests and parallel validation/replicate-sequential cases verify actual identities. Parent population tests verify unchanged metadata order when shuffling is disabled. |
| Distinguish a complete active-row pass from physical uniqueness | Duplicate snapshot and parallel validation cases preserve intentional repeated identities. The runtime example now describes inherited sample count rather than promising every row once. |
| Expose split distribution limits | `test_partition_policy_exposes_weight_distribution` uses weights 1 and 10^12 with two workers: split emits 50 draws from each one-row population; replicate emits all 100 draws from the heavy sample for the fixed seed. It runs under fork, spawn, and forkserver. This difference is documented behavior. |
| Share one budget in replicate mode | Real-worker weighted, replicate-sequential, and tiny-replicate cases check finite counts, full populations, and intentional overlap. Parent recipe tests verify a shared IPC snapshot. |
| State replay and partial-pass behavior | Serial partial-window tests and parallel replay, persistent-validation, persistent-shuffle, random-persistent, and partial cases exercise fresh versus retained readers. Discarded multiprocess prefetch can leave gaps; the contract says so. |
| Enforce process ownership and topology | Lazy-setup, pickle, actual inherited-fork rejection, initializer-order, and `test_prepared_topology_must_match_consumer` checks cover fresh PID-bound readers and empty quotas. `test_explicit_base_context` also consumes through a real BaseContext. |
| Account for scalable memory and work | Source inspection keeps preparation columnar and linear. Serial and parallel many-cache smoke cases process 10,000 active rows without collecting outputs. The memory costs below remain explicit input-sized costs. |
| Match collation and type contracts | Custom collators receive lists for batch size one and larger batches; transform and batching tests check actual outputs. Torch's `DataLoader` generic describes input samples, and its iterator does not statically guarantee the custom batch type. The adapter retains that standard Torch typing boundary. |
| Propagate failures and release workers | Supervised real-worker callback-error, transform-error, native-error, death, timeout, partial teardown, and Ctrl-C cases exercise supported process contexts with finite deadlines and cleanup. |

Both original findings are resolved: oversized explicit or inherited budgets fail
before metadata preparation, and zero-worker persistence fails before source
metadata is accessed. The budget review also exposed floating-point rounding in
Torch's batch-length calculation. Accepted sample budgets are now bounded by
`min(sys.maxsize, 2**53)` so the ordinary Torch length calculation remains exact.
Batch size must fit `sys.maxsize`; invalid boolean flags are rejected rather than
being treated as truthy values.

Memory accounting remains part of the contract, rather than a claim of constant
memory. For W loading workers and M bytes of prepared metadata IPC, spawn and
forkserver retain approximately `(W + 1) * M` IPC bytes across parent and workers,
before native tables, indexes, and transient serialization copies. All recipes
currently travel to each worker, including zero-quota workers. Replicate mode
shares one IPC value within each process; fork may share immutable bytes through
copy-on-write. Each consuming process also owns native indexes and metadata.
Torch prefetch costs scale with workers, prefetch factor, batch size, and decoded
sample size; native payload buffers follow their bounded read window. Arbitrary
payload sizes and user decoding are application-sized memory costs.

Supported contexts require picklable callbacks where Python multiprocessing
requires them. Exact global weighted draws require replicate mode; shuffled split
mode deliberately approximates the distribution. These documented limits are
accepted contract boundaries, not untested promises of coverage or replay.

Verification: `just check` passed with 499 Python tests and 15 Rust tests,
including formatting, Ruff, precise-type checks, Pyrefly, generated API docs,
Clippy, and a rebuilt extension. Distributed tests marked slow were outside this
single-GPU review. Rust implementation and binding signatures were unchanged.
