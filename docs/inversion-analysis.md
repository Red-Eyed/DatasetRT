# Inversion Analysis

Charlie Munger often pointed to Jacobi's maxim, "Invert, always invert." DatasetRT applies that idea by starting with the failures an ML dataset runtime must make hard, then shaping the architecture as the inverse of those failures.

## Failure List

- Silent sample duplication.
- Nondeterministic epoch order.
- Weight updates applied to the wrong physical sample.
- Readers accepting incomplete or corrupt caches.
- Startup paths that hash or materialize more data than requested.
- Queues that grow without bound under slow consumers.
- Python adapters that accidentally own correctness-sensitive iteration state.
- Payload decoding or framework behavior leaking into the Rust correctness core.
- Metadata tables expanding into Python row objects on dataset-scale paths.
- A shuffled epoch being mistaken for a permutation instead of weighted sampling with replacement.
- Cache identity drifting away from stable `(cache_id, sample_id)` pairs.

## Design Response

DatasetRT makes those failures hard by keeping correctness-sensitive behavior in Rust:

- Rust owns cache identity, sampling, validation, storage layout, bounded queues, and ordering.
- Python stays the ergonomic edge that describes sources, writes metadata, and decodes payload bytes.
- Readers reject incomplete or malformed caches instead of attempting best-effort recovery.
- Weight updates are accepted only after Rust validates exact physical-sample coverage, uniqueness, and positive finite values.
- Startup avoids full shard checksum hashing unless `validate_cache=True` asks for it explicitly.
- Runtime queues are bounded so producer speed cannot silently become unbounded memory growth.
- Unsupported adapter modes should fail clearly rather than produce plausible-looking bad training data.

The useful intuition is that DatasetRT does not try to make every path flexible. It makes the dangerous paths narrow, typed, bounded, and validated before they can affect training.

## Nonpersistent-worker training replay — 2026-10-13

The user reported affected training after nonpersistent workers repeatedly drew
the same short epoch from a larger weighted population. Reconstructing readers
preserves weights and seeds but resets sampling offsets; it does not continue
training coverage. The default was insufficiently guarded, and earlier assistant
advice incorrectly dismissed persistence as a correctness concern.

Loading workers now persist by default; explicitly nonpersistent workers warn
at construction. Use persistent
workers to continue native streams, or zero-worker loading to retain state in the
main process. This warning does not fix replay automatically. The detailed
[failure analysis and default review](inversion-checklists.md#training-failure-analysis-nonpersistent-worker-replay--2026-10-13)
records the cause, impact limits, regression coverage, and remaining epoch-state work.
