---
title: Dataset Caching for Machine Learning
description: Read cached training samples with a fast Rust runtime, and analyze, filter, or weight a separate metadata table without unpacking serialized payloads.
---

# DatasetRT — Dataset Caching for Machine Learning

DatasetRT is a Rust-powered dataset cache for Python machine learning pipelines.
Cache serialized records once, then analyze, filter, or rebalance their separate
metadata without unpacking or rewriting the payloads.

## Start here

Install the package, then follow the [runnable quickstart](cache-training-data.md)
to define an iterable source, pack Python records with pickle, and read them back.

```bash
uv add dataset-rt
```

Or use `pip install dataset-rt`. Install **`dataset-rt`**, import **`dataset_rt`**.
Python 3.10–3.13 is supported; PyTorch is optional.

## Payloads and metadata

| Part | What it stores | What you do with it |
| --- | --- | --- |
| Payload store | Serialized bytes in indexed disk shards; commonly pickled dictionaries, arrays, or nested Python structures | Read selected samples and unpack them for training |
| Metadata table | Sample identities, primitive attributes, and sampling weights | Analyze class distributions, filter the population, and adjust sampling probabilities |

The metadata table is exposed through a **Polars DataFrame** in Python. Payloads
are stored separately and linked by `(cache_id, sample_id)`. Updating metadata
does not require unpickling the records or rewriting their cached bytes.

Rust owns indexed sample reading, native workers, and bounded prefetching.
See the [reader benchmarks](benchmarks.md) for measured throughput, startup time,
and memory use. Your Python code owns unpacking and training transforms.

## Find the guide you need

| You want to… | Start with… |
| --- | --- |
| Create and reuse a cache | [Quickstart: cache your first dataset](cache-training-data.md) |
| Pack dictionaries, arrays, or nested records | [Serialize Python records with pickle or dill](serialization.md) |
| Inspect class counts, filter records, or rebalance sampling | [Analyze, filter, and weight samples](polars-sampling.md) |
| Decode and batch samples for training | [Load batches with PyTorch](pytorch-dataloader.md) |
| Choose worker counts and understand mmap memory sharing | [Runtime, workers, and shared memory](runtime-guide.md) |
| Understand weighted sampling and stream progress | [Sampling and reproducibility](determinism.md) |
| Look up methods and configuration | [Python API reference](python-api.md) |
| Measure your own workload | [Run your own benchmarks](remote-validation.md) |

## Important behavior

- Shuffled reads use weighted multinomial sampling **with replacement**. Samples
  can repeat within a pass, and others may be absent.
- Metadata edits affect future dataset iterators. Construct a new PyTorch loader
  to use an updated population.
- Cache reuse follows the source name. Change that name or cache root when
  preprocessing or source contents change.
- Loader workers share mapped metadata and index pages on the same host.
  Sampling state, decoded records, and prefetched batches still use private RAM.

The sidebar keeps the quickstart, user guides, concepts, API reference, and
benchmarks available throughout the documentation.

## Project links

[Package on PyPI](https://pypi.org/project/dataset-rt/) ·
[Source code on GitHub](https://github.com/Red-Eyed/DatasetRT) ·
[Report an issue on GitHub](https://github.com/Red-Eyed/DatasetRT/issues)
