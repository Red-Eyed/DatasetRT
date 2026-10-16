---
title: DatasetRT vs WebDataset, LMDB, and Hugging Face Datasets
description: Compare dataset storage, preprocessing caches, and sampling responsibilities when choosing DatasetRT, WebDataset, LMDB, or Hugging Face Datasets.
---

# DatasetRT vs WebDataset, LMDB, and Hugging Face Datasets

These tools overlap in dataset I/O but expose different boundaries. Choose based
on how your data is stored, how training selects samples, and which responsibilities
you want the library to own. This is a feature and workflow comparison, not a
cross-library performance benchmark. Upstream documentation was reviewed on
October 16, 2026.

## Compare the main responsibilities

| Tool | Storage and access model | Relevant built-in behavior | A reason to evaluate it |
| --- | --- | --- | --- |
| DatasetRT | Immutable local payload shards with a sample index and a separate Arrow metadata table | Native reads, Rust-owned validation, metadata analysis/filtering/weight edits through Polars, weighted sampling with replacement | Analyze and change the sampling population without unpacking or rewriting payloads |
| WebDataset | Samples grouped in tar shards, with streaming pipelines and local or remote input | Shard-oriented processing, resampling, and optional local caching | Stream large datasets from tar archives or object storage |
| LMDB / py-lmdb | Memory-mapped transactional key-value storage | Transactions, key lookup, cursors, and optional buffer access | Build a custom dataset representation around transactional key-value records |
| Hugging Face Datasets | Arrow-backed map-style datasets or streaming iterable datasets | Dataset loading and transforms; fingerprinted transform caching for regular datasets | Use an integrated loading/preprocessing ecosystem with table-like or streaming access |

For the upstream descriptions, see the
[WebDataset README](https://github.com/webdataset/webdataset),
[py-lmdb documentation](https://lmdb.readthedocs.io/en/release/), and
[Hugging Face dataset access models](https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable).

## When DatasetRT fits

DatasetRT puts the cache publication and sampling contract together. Your source
serializes bytes once. Rust validates cache structure and controls ordered,
bounded reads. You can filter a Polars metadata table or change class weights
before constructing the next iterator or loader.

That combination is useful when experiments reuse payloads but repeatedly change
subsets, class balance, or hard-example weights. See the
[Polars tutorial](polars-sampling.md) for a complete example.

DatasetRT's costs and boundaries matter:

- Cache directories must be available through the filesystem. DatasetRT does not
  provide WebDataset's remote streaming interface.
- You own serialization, decoding, and source versioning. Cache reuse by source
  name does not fingerprint preprocessing code.
- `shuffle=True` means weighted sampling with replacement, so it does not ensure
  each physical sample appears exactly once per epoch.
- Metadata edits require a columnar population table. Replicated loading workers
  retain private sampling state, so increasing workers can increase memory use.

## When another boundary fits better

WebDataset explicitly organizes samples in tar shards and supports sequential
pipelines over local files or streams. Its repository also includes `wids`;
evaluate that separately if you need indexed access. Do not infer that every
WebDataset-related API is restricted to sequential access. See the
[upstream format and library description](https://github.com/webdataset/webdataset#the-webdataset-format).

LMDB gives your application transactions over key-value data. In a custom ML
pipeline, you choose the record schema and the training sampler on top of that
storage API. Its memory mapping can make RSS look large even when mapped clean
pages are reclaimable; see the
[upstream memory discussion](https://lmdb.readthedocs.io/en/release/#memory-usage).

Hugging Face Datasets offers regular datasets with random row access and
streaming iterable datasets with different shuffle behavior. Its transform cache
tracks dataset state through fingerprints, including transform functions and
parameters. That differs from DatasetRT's explicit source-name reuse boundary.
See [Hugging Face cache fingerprints](https://huggingface.co/docs/datasets/about_cache).

For adoption, run the same useful workload on the candidates: identical bytes,
decode/transform work, sample counts, ordering or sampling requirements, and
worker budgets. DatasetRT's [local measurements](benchmarks.md) compare its own
reader configurations; they establish no speed ranking against these libraries.
