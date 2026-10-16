---
title: Dataset Runtime, Sampling, and PyTorch Loading
description: Understand cache reuse, native worker counts, finite epochs, metadata snapshots, deterministic sampling, and PyTorch loader resource tradeoffs in DatasetRT.
---

# Dataset runtime, sampling, and PyTorch loading

`DatasetRuntime(num_workers=N)` owns a reusable native worker pool. It creates
caches and loads existing cache directories into datasets. A loaded dataset keeps
its runtime alive, so it can continue reading after the original runtime variable
goes out of scope.

## Reuse complete caches

`from_cache_sources()` reuses complete caches under `root / source.name`, creates
missing caches, and loads successful results. Inspect the per-source outcomes:
the combined call can succeed even when some sources fail. Use a new name or
cache root after changing preprocessing; reuse does not fingerprint source code.

`cached_dataset()` loads known cache directories without discovering sources.
Default loading validates structural requirements without hashing all shard
contents. Enable `validate_cache=True` explicitly for checksum validation.

See [cache training data](cache-training-data.md) for a runnable example.

## Set the active population and epoch length

`get_metadata()` returns a Polars table containing physical identity, stored
metadata, and weights. `update_metadata()` validates an edited table in Rust,
resets the sampling stream, and sets epoch length to its row count. It does not
rewrite the immutable cache. Call `set_epoch_len(n)` afterward to override the
finite number of samples emitted by future dataset iterators.

For a population of 100 rows and epoch length 20, each iterator emits 20 samples.
Sequential iterators continue through the table cyclically; shuffled iterators
continue a weighted draw stream with replacement. Shuffled epochs can repeat
records and omit others. Existing iterators retain their snapshot when metadata
or epoch length changes. See [determinism](determinism.md).

## Choose loading processes and native threads

`to_torch_dataloader()` creates a finite PyTorch loader with Python transforms
and native cache reads. `num_workers` controls Python loading processes;
`native_num_workers` controls reader threads per consumer. These are independent
resource choices. Start with zero loading workers as a baseline, then measure
with your actual transforms and payload sizes.

Spawned workers need a guarded application entry point and importable, picklable
callbacks. They reconstruct readers from settings and cache paths; do not pass
live native runtimes or iterators into child processes. Keep transforms on the
CPU, and transfer batches to the training device in the training process.

Persistent loading workers are automatic with loading processes. They retain
sampling progress between passes. Explicitly disabling persistence reconstructs
readers and repeats their initial sequence, even if the loader selected a random
seed at construction. See the [complete PyTorch example](pytorch-dataloader.md).

## Share metadata and indexes with mmap

Multiprocessing loaders reduce repeated metadata and index allocations through
read-only, file-backed memory mapping (`mmap`). Each loader exports one immutable
Arrow metadata snapshot. Its workers map that same file, so the operating system
can share the physical file-cache pages between them on the same host. Four
workers can reference the same metadata buffers instead of each decoding a
complete private copy of the table.

Workers map the cache's original `index.bin` files too. Those index pages can be
shared even across independently constructed loaders reading the same caches.
For shuffled split loading, workers also map the loader's shared partition-order
file. Workers open their mappings independently, so this design works with
`spawn`, `fork`, and `forkserver` without inheriting live native readers.

This is file-backed sharing rather than a named shared-memory segment. Separate
metadata exports have separate backing files, and processes on different hosts
do not share RAM. Weights, selected sample indices, sampling state, decoded
records, and prefetched batches still allocate memory per consumer. Explicit
DataFrame operations can also allocate new buffers. Summed process RSS can count
shared mapped pages more than once; use private/proportional memory measures
when evaluating the savings.

## Account for weights, ordering, and memory

Shuffled loaders default to replicated populations to preserve global weighted
frequencies. Each consumer retains private sampling state: ten million f64
weights alone occupy 80 MB per consumer. Split mode partitions active rows and
assigns consumer quotas; weighted frequencies can differ from global sampling
because quotas follow batches rather than partition weight sums.

Sequential loaders default to split populations. Torch interleaves worker
batches, so multiple workers need not emit the original table's global order.
Use one consumer when that order matters.

Native task/result windows and loader batch prefetch remain bounded, but their
memory budgets multiply across consumers. Include maximum payload size,
transformed batch size, metadata, and sampling state in the budget. Additional
workers can increase memory and startup time without improving throughput.

## Distributed sample budgets

Initialize the application's default distributed process group before building
the loader in each rank. `samples_per_epoch` counts global samples. For 1,000
requested samples, three ranks, and batch size 32, the default pads to 334 samples
and 11 batches per rank. With `drop_last=True`, each rank receives 320 samples
and 10 batches, for 960 samples globally. Rank zero warns when the effective
budget differs from the requested one.

An explicit shuffled seed reproduces fresh loaders with identical cache contents,
active metadata, and worker topology. Changing topology can change the sequence.
Construct a new loader after changing the active metadata; an existing loader
uses its exported snapshot. Full option and validation details are in the
[Python API](python-api.md). Use the [benchmark commands](remote-validation.md)
to measure a particular machine and distributed setup.
