---
title: Dataset Cache Benchmarks — Startup, Throughput, and Memory
description: Reproduce DatasetRT reader measurements with raw JSON evidence covering native iteration and PyTorch loaders, startup latency, throughput, and process-tree RSS.
---

# Dataset cache benchmarks: startup, throughput, and memory

This local reader benchmark compares DatasetRT native iteration with its PyTorch
loader at zero, one, and two loading workers. It measures a small synthetic
sequential workload. It does not compare DatasetRT with other libraries or
establish training, GPU, network-storage, or production-scale performance.

## Environment and workload

- Measured October 16, 2026 on an Apple M5, 10 logical CPUs, 24 GiB RAM,
  macOS 27.0.1 arm64, using a local temporary filesystem.
- Python 3.11.15, PyTorch 2.14.1, psutil 7.2.2, DatasetRT metadata version 0.4.2.
- Reader and harness from Git base
  [`8945dca`](https://github.com/Red-Eyed/DatasetRT/commit/8945dca5fc84694b3cbc0b3dd56ccd05f7a3b90f).
  The working tree contained documentation and package metadata changes; no reader
  or benchmark code changes. The raw report records both source fingerprints
  and `worktree_dirty: true`.
- Two caches with 4,096 records each, 16 KiB payloads: 8,192 samples and 128 MiB
  of payload per pass. Metadata contains an integer index per record.
- Batch size 64, two native reader threads per consumer, native prefetch 64,
  loader prefetch two batches per worker, sequential sampling, seed 7.
- Each delivered sample is SHA-256 hashed once; no extra transform rounds,
  image decoding, model execution, or compression. Results cross loader process
  boundaries as compact digests rather than full payloads.
- Five trials per configuration, rotating their execution order. Each trial
  creates a fresh dataset/loader and consumes two passes. Loading workers persist
  between passes. This was a developer session rather than an isolated benchmark
  machine; short timings are sensitive to other activity.

## Measured results

All values below are medians of five trials. The complete
[raw JSON report](benchmark-results.json) includes every observation and setting.

| Reader | Construction (ms) | First-pass iterator + first batch (ms) | Second-pass steady samples/s | Second-pass peak process-tree RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Native iteration | 0.45 | 0.69 | 92,090 | 243.5 |
| PyTorch loader, 0 workers | 9.33 | 1.63 | 92,464 | 247.6 |
| PyTorch loader, 1 spawned worker | 9.15 | 475.12 | 86,622 | 510.8 |
| PyTorch loader, 2 spawned workers | 8.98 | 506.41 | 174,082 | 761.6 |

Native construction and loader construction use different boundaries; inspect
the methodology below before comparing those columns. The spawned configurations
show a substantial first-batch cost and a higher process-memory footprint. Two
workers deliver higher steady throughput in this particular second-pass workload;
one worker does not improve on the zero-worker baseline. Re-measure with your
actual decoding, payload sizes, and storage before choosing a topology.

## Reproduce from a checkout

The command generates private temporary caches automatically, validates consumed
record identities/counts, and writes a JSON report. The Just target builds the
extension and synchronizes its development dependencies before running the
existing benchmark script:

```bash
just bench-reader \
  --samples-per-cache 4096 --caches 2 --payload-bytes 16384 \
  --transform-rounds 0 --batch-size 64 \
  --native-workers 2 --prefetch-size 64 \
  --contexts '["spawn"]' --worker-counts '[1,2]' \
  --workloads '["cheap"]' --shuffles '[false]' --repeats 5 \
  --output results/reader.json
```

The harness automatically includes native and zero-worker controls. Process
inspection and multiprocessing shared memory must be permitted on the host.

## Measurement boundaries and limitations

- Construction includes runtime creation, cache loading, a warming sample read,
  epoch configuration, and transform setup. Loader cases additionally include
  `to_torch_dataloader()` and its reconstruction metadata export.
- First-batch latency is the JSON's `first_batch_seconds`, measured from before
  iterator creation through the first delivered batch. It already includes
  `iterator_seconds`; adding that field again would double-count it.
  It excludes construction, interpreter startup, imports,
  and fixture generation. The first pass is called `cold` in the JSON but is
  **not** a cold filesystem-cache test: writing fixtures has already warmed data.
- Steady throughput is delivered samples after the first batch divided by the
  recorded remaining pass time. The second pass is `warm` in the JSON. Per-pass
  throughput excludes construction and first-batch latency.
- The RSS column is the median of each second pass's sampled peak summed RSS
  over the benchmark process and its child processes. It includes interpreter
  and library overhead, may count shared pages repeatedly, and can miss brief
  peaks. It is not incremental reader allocation or a bounded-memory proof.
- Each pass verified 8,192 accepted samples and 134,217,728 payload bytes.
  Synthetic record headers also check physical identity. This run exercises
  multiple caches but does not establish behavior at millions of metadata rows.
- The report's serial-overhead gate passed. Four-worker speedup gates were not
  evaluated because this run did not request four workers. This is not a full
  performance-acceptance run.

For real training, existing caches, CUDA, and distributed runs, use the
[portable benchmark commands](remote-validation.md). Read the
[tool comparison](comparisons.md) for differences in storage and sampling scope.
