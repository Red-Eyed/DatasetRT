---
title: DatasetRT Architecture Overview
description: Understand DatasetRT's public boundaries between Python preprocessing and serialization, Rust cache integrity and sampling, immutable storage, and optional PyTorch loading.
---

# DatasetRT architecture overview

DatasetRT separates application-specific data handling from cache correctness.
Python describes the source and the meaning of its bytes. Rust manages the
storage and sampling contracts used by every application.

## Two connected parts

The payload store contains serialized sample bytes in indexed disk shards. The
separate columnar metadata table contains sample identities, primitive attributes,
and sampling weights. `(cache_id, sample_id)` links a metadata row to a payload.

Python exposes the metadata through Polars for analysis, filtering, and weight
editing. Those operations do not unpack the stored sample records. Rust validates
the accepted population and weights, selects samples, and reads their payloads
with native workers and bounded prefetching. Python then unpacks the delivered
bytes for training. See the [reader measurements](benchmarks.md) for the runtime's
measured performance and workload boundaries.

For multiprocessing loading, workers share an immutable Arrow metadata snapshot
through read-only, file-backed `mmap`. They also map the original cache indexes.
The OS can share those physical pages between processes on one host, reducing
duplicated RAM. Private weights, sampling state, decoded records, and batch buffers
still scale with consumer count; see the [memory guide](runtime-guide.md#share-metadata-and-indexes-with-mmap).

| Responsibility | Owner |
| --- | --- |
| Find source records and perform preprocessing | Your Python source |
| Serialize and decode images, tensors, text, or domain records | Your Python application |
| Publish complete caches and validate storage structure | Rust runtime |
| Validate identities, metadata edits, and sample weights | Rust runtime |
| Select samples and preserve planned reader order | Rust runtime |
| Bound native tasks, results, and prefetch | Rust runtime |
| Express population filters and weights | Polars expressions in Python |
| Transform and collate training batches | Optional PyTorch integration and application callbacks |

## From source to training sample

1. A named iterable source packs Python records or dictionaries with `pickle`
   or `dill`, then yields `CacheInput` with those bytes and primitive metadata.
2. DatasetRT writes payload shards, an index, Arrow metadata, and a manifest.
   A complete manifest marks the cache as published.
3. A dataset loads one or more cache directories. Each physical sample has a
   stable `(cache_id, sample_id)` identity. Treat IDs as opaque values.
4. Future iterators read from the current active population, using either
   sequential order or deterministic weighted sampling with replacement.
5. Python decodes the delivered payload. An optional PyTorch loader performs
   application transforms and batching in configured loading processes.

## Immutable storage, editable sampling

Runtime metadata edits change which cached samples are selected and how much
weight they receive. They do not rewrite payloads or persisted metadata.
Already-created iterators retain their snapshots. New loaders capture the
population when they are constructed.

See the [storage format](storage-format.md), [sampling semantics](determinism.md),
and [runtime guide](runtime-guide.md) for the contracts relevant to applications.
