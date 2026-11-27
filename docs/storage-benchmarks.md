# Storage layout benchmarks

`tests/test_storage_benchmark.py` runs small real writes and reads in ordinary
pytest. It checks text and valid PNG payloads, metadata, sample counts, native
compression and report generation. It contains no machine-speed assertions.

For performance comparisons, `scripts/bench_storage.py` runs the same harness in
isolated interpreters containing baseline and candidate wheels. Build both from
their respective revisions with the same release profile and toolchain:

```bash
uv run --python 3.11 --extra dev maturin build --release --out /tmp/storage-wheels
```

Pass existing baseline and candidate wheel paths to the runner. It creates and
removes its own environments and fixtures, resolves baseline dependencies and pins those same direct Python dependency
versions in the candidate environment, and requires binary wheels for installation:

```bash
just bench-storage \
    --baseline-wheel /path/to/dataset_rt-0.5.9-platform.whl \
    --candidate-wheel /path/to/dataset_rt-0.5.10-platform.whl \
    --samples-per-cache 256 --repeats 5 \
    --output /tmp/storage-comparison.json
```

Use the actual wheel filenames produced by the build. Alternatively, pass
`--baseline-python` and `--candidate-python` pointing to already prepared
interpreters; their dependencies and build profiles are then your responsibility.
Running without distinct releases is useful as a smoke test but cannot pass
performance acceptance.

## Workloads and measurements

| Profile | Payload |
| --- | --- |
| `text-64` | 64 bytes of deterministic English prose |
| `text-256` | 256 bytes of deterministic English prose |
| `text-4k` | 4096 bytes of deterministic English prose |
| `image-64k` | Approximately 62 KiB, valid 144×144 RGB PNG |
| `image-1m` | Approximately 1 MiB, valid 592×592 RGB PNG |

Images encode seeded pixel noise to represent already compressed image bytes.
The text is synthetic prose, not a representative LLM corpus. Each profile keeps
eight payload variants in memory and streams samples from that fixed pool. This
isolates cache storage rather than image decoding, tokenization or source I/O;
it does not model production content diversity or a dataset larger than RAM.
Per-sample metadata contains identity, provenance, a path, quality and split.

The default matrix uses four caches, `none` and `zstd` codecs, one and four native
workers, prefetch 64, and five paired repetitions. `--codecs '["lz4"]'` adds an
LZ4-only comparison. Profile and worker matrices accept JSON arrays as well.
Pair order alternates baseline/candidate and candidate/baseline. Each trial runs
in a fresh process and removes its files before the next trial.

Reports include:

- Durable write time, payload bytes, stored shard bytes and total cache size.
- Dataset construction time, total read time, first-sample latency and sample
  counts for sequential and shuffled first/warm epochs.
- Process peak RSS and initial RSS. Peak RSS covers the entire trial, including
  fixture generation, writing, reading and correctness verification; it does not
  isolate reader memory or include filesystem cache residency outside the process.
- Physical payload and metadata verification outside timing. Timed reads check
  sample identity and record an order digest; comparisons require identical work
  and deterministic order in each corresponding epoch.
- Raw paired ratios and their medians. Throughput ratios above 1 favor v5;
  storage and memory ratios below 1 favor v5.

Shuffled epochs use weighted sampling with replacement, so the order digest may
contain repeated identities. The separate sequential verification checks every
physical sample exactly once.

## Performance acceptance

Add `--enforce` to return a nonzero exit status when any read-throughput median
regresses beyond `--max-read-regression` (default 0.10). Acceptance requires at
least five complete pairs, v4 baseline caches and v5 candidate caches. Size,
write-throughput and memory ratios are reported without speed-based gates.

Use enough samples to make read intervals meaningful. For example, select only
text with `--profiles '["text-256","text-4k"]' --samples-per-cache 16384`.
Image profiles need fewer samples to exercise comparable byte volumes.

First-pass means a new dataset reader, **not cold disk**. Fixtures have just been
written, and filesystem cache state is uncontrolled. Warm means another epoch on
the same dataset; shuffled order advances deterministically between epochs.
Do not infer cold-cache or production-scale performance from this benchmark.
Run filesystem-cache-controlled trials separately on the intended training host
before relying on cold random-read behavior.

## Local v4 versus v5 results

The [raw report](storage-benchmark-results.json) retains 200 release-build trials
and all 80 read-throughput gates. Measurements used Python 3.11.15 on macOS
27.2.1 arm64, with both wheels built locally using Rust 1.97.1. The v4 baseline
is unmodified 0.5.9 at `bfe84fef1da1aa75ddbdcfd3e82000681cc79b3f`; the candidate
is 0.5.10. Both environments used matching dependency versions, recorded per trial.

These timing measurements precede the later explicit record-layout dispatch
refactor. The refactored release build subsequently passed 40 smoke trials (20 pairs)
covering all five profiles, both codecs and both worker counts, with eight samples
per cache. Those smoke runs verify bytes, metadata and ordering; they are not a
repeat of the five-pair throughput acceptance matrix.

Text trials used 16,384 samples per cache (65,536 total); image trials used 256
per cache (1,024 total). Both used four caches, prefetch 64, no compression and
Zstd level 3, one/four workers, and five alternating pairs. Run the text and image
profile subsets separately with these sample counts to reproduce that matrix.

Ranges below cover the paired medians across the codec/worker combinations.
Throughput ratios above 1 favor v5. RSS is the whole-trial process peak, not an
isolated reader-memory measurement.

| Payload | Total cache reduction | Write throughput | Warm shuffled throughput | Peak RSS ratio |
| --- | --- | --- | --- | --- |
| 64 B text | 35.0–45.7% | 1.09–1.30× | 1.09–1.55× | 1.09–1.13× |
| 256 B text | 30.2–32.0% | 1.17–1.21× | 1.12–1.41× | 1.09–1.13× |
| 4 KiB text | 4.6–17.6% | 1.19–1.28× | 1.10–1.59× | 1.09–1.13× |
| ~64 KiB PNG | 0.322% | 1.01–1.03× | 1.07–1.30× | ~1.00× |
| ~1 MiB PNG | 0.019% | 1.00–1.02× | 1.02–1.20× | 0.99–1.02× |

All read-throughput medians passed the configured 10% regression limit, but
uncompressed sequential text cases included 3.5% and 5.3% slowdowns. First-sample latency also increased. These
are pooled medians of the first sequential sample across repetitions, codecs
and worker counts, in milliseconds:

| Payload | v4 | v5 |
| --- | --- | --- |
| 64 B text | 0.081 | 0.367 |
| 256 B text | 0.080 | 0.373 |
| 4 KiB text | 0.079 | 0.359 |
| ~64 KiB PNG | 0.086 | 0.139 |
| ~1 MiB PNG | 0.208 | 0.267 |

The larger text runs exposed repeated whole-column UTF-8 validation when readers
switched batches. V5 now validates each immutable mapped batch once and shares
only that validation marker; workers still retain one decoded batch each.
Corrupt UTF-8 tests verify that failed validation never enables reuse.

The sentence-sized `text-64` case uses 193–197 bytes of compact JSON metadata
per sample: metadata is about three times the payload. Total cache size falls
35.0% with Zstd and 45.7% without compression. Shards alone shrink 65.1% and
76.1%, respectively; the unchanged Arrow metadata table reduces the overall gain.
That ratio is a controlled example, not a substitute for measuring the actual
metadata fields and their compressibility.

These measurements support text storage savings and improved shuffled throughput
for this synthetic workload. They also show higher text peak RSS and first-sample
cost. They do not establish a universal performance improvement or cold-disk
behavior.
