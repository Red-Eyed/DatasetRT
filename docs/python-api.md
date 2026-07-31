# Python API

The Python API is intentionally thin. It describes what the user wants to cache or read, while the Rust core owns the cache format, validation, sampling, and iterator state.

## CacheInput

```python
class CacheInput(NamedTuple):
    data: bytes | bytearray | memoryview
    metadata: Mapping[str, bool | int | float | str]
```

`data` must be bytes-like. DatasetRT v0.1 does not accept arbitrary Python objects.

`metadata` must be a mapping with string keys and primitive values. Every emitted row must use the same metadata keys and compatible field types.

## CacheSource

```python
class CacheSource(Protocol):
    name: str

    def __iter__(self) -> Iterator[CacheInput]:
        ...
```

A `CacheSource` is a synchronous Python iterable. Python does not create threads or queues. The Rust writer pulls from the iterable and owns the cache construction rules.

## DatasetRuntime

```python
runtime = DatasetRuntime(num_workers=4)
```

`DatasetRuntime` owns exactly `num_workers` reusable Rust threads. Cache loading,
sample reading, and cache writing called through the runtime share that pool. The
worker count is never inferred from the machine and is not repeated on operations.

## DatasetRuntime.write_cache

```python
results = runtime.write_cache(
    source,
    base_cache_dir,
    writer_config=WriterConfig(
        prefetch_size=64,
        max_shard_bytes=...,
        shard_compression=ShardCompression(algo="none", ratio=1.0),
        show_progress=True,
        validate_cache=False,
    ),
)
results = runtime.write_cache(
    [source_a, source_b],
    base_cache_dir,
    writer_config=WriterConfig(...),
)
```

`DatasetRuntime.write_cache` creates immutable caches and returns one result per source.

`path` is always the base cache directory. Python passes this through to Rust as `base_cache_dir`.

Each source is written by Rust under `base_cache_dir / name`. The returned list preserves source order and contains either `CacheWriteSuccess(source_name, path)` or `CacheWriteError(source_name, message)`.

Rust writes each cache into a temporary directory under `base_cache_dir / tmp` first and publishes it by renaming only after the manifest is complete. If one source in a multi-source write fails, its temporary cache is cleaned up and the remaining sources still run.

Per-source failures do not stop the rest of the write. For example, an empty source returns `CacheWriteError(source_name="empty", message="cache source yielded no samples")` while other sources can still return `CacheWriteSuccess`.

```python
for result in results:
    match result:
        case CacheWriteSuccess(source_name, path):
            print(source_name, path)
        case CacheWriteError(source_name, message):
            print(source_name, message)
```

`prefetch_size` bounds the writer's in-flight task and result window. If Python iteration is faster than writing, Rust applies backpressure when that window reaches `min(prefetch_size, num_workers)`. For multi-source writes, the window may span source boundaries while ordered commit preserves deterministic output.

The runtime's fixed worker count limits active jobs. Calls do not create threads or resize the pool.

`show_progress` controls Rust-owned write progress. It is enabled by default and reports committed samples/s and MB/s for the active source. For multi-source writes, it also reports completed sources with an ETA based on wall time and completed source count. Set it to `False` for quiet tests, background jobs, or logging systems that do not want terminal progress output.

`validate_cache` controls checksum validation when the writer reuses an existing cache. It is `False` by default so `DatasetRuntime.from_cache_sources` does not hash every payload shard before handing paths to the dataset loader. Set it to `True` for integrity checks.

Writer configuration is a frozen pydantic model:

```python
CompressionAlgo = Literal["none", "lz4"]

class ShardCompression(BaseModel):
    algo: CompressionAlgo = "none"
    ratio: float = 1.0

class WriterConfig(BaseModel):
    prefetch_size: int = 64
    max_shard_bytes: int = 64 * 1024 * 1024
    shard_compression: ShardCompression = ShardCompression()
    show_progress: bool = True
    validate_cache: bool = False
    profiler: WriterProfilerConfig = WriterProfilerConfig()

class WriterProfilerConfig(BaseModel):
    enabled: bool = False
    path: Path = Path("dataset_rt_profile.json")
```

Python validates the config shape with pydantic, then Rust validates it again before writing. DatasetRT v0.1 supports `algo="none"` with `ratio=1.0` and `algo="lz4"` with a positive finite ratio value.

LZ4 compression is applied per payload record, not to the whole shard as one stream. This keeps random sample access direct: the index locates one compressed record, and Rust decompresses that record before returning the original bytes. The commonly named Rust `zstd` crate is a binding to the C zstd library, so it is not used for v0.1. A pure-Rust zstd implementation may be considered later once its compression path is accepted as stable enough for DatasetRT.

Set `profiler=WriterProfilerConfig(enabled=True, path=Path("profile.json"))` to write a structured timing summary after the write returns, including handled failures such as Ctrl-C. The summary separates Python iterator time (`python_next`), Python-to-Rust extraction (`python_extract`), queue backpressure (`ingress_wait`), Rust metadata/record work, compression, disk writes, finish steps, and publish time.

Rules:

- Every target path must not already contain a completed cache.
- Source names used as subdirectories must be plain path segments.
- `prefetch_size` must be greater than zero.
- `validate_cache=True` verifies existing cache checksums before reuse.
- The source must yield at least one sample.
- Payload data must be bytes-like.
- Metadata schema is inferred from the first sample and validated for every sample.
- Rust assigns `cache_id`, `sample_id`, shard offsets, and checksums.

## DatasetRuntime.cached_dataset

```python
dataset = runtime.cached_dataset(
    [path],
    reader_config=ReaderConfig(
        seed=42,
        prefetch_size=64,
        shuffle=True,
        validate_cache=False,
    ),
)
```

The method loads one or more immutable caches and returns a synchronous `CachedDataset`. The dataset retains the runtime pool internally, so reads remain valid for the dataset's lifetime.

Reader configuration is a frozen pydantic model:

```python
class ReaderConfig(BaseModel):
    seed: int
    prefetch_size: int = 64
    shuffle: bool = True
    validate_cache: bool = False
```

`validate_cache` is `False` by default so dataset construction reads manifests, metadata, and indexes without hashing every shard payload. Set it to `True` when startup should verify metadata, index, and shard checksums before iteration.

Rules:

- Dataset state is owned by Rust.
- Multiple readers may load the same cache concurrently.
- `prefetch_size` controls the bounded Rust reader queue.
- The runtime worker count limits active read jobs.
- `validate_cache=True` verifies metadata, index, and shard checksums during dataset construction.
- With `shuffle=True`, each Python iterator snapshots the current weight vector.
- With `shuffle=False`, samples are emitted once in physical cache order.
- Changing weights after iterator construction affects future shuffled iterators, not existing ones.

## DatasetRuntime.from_cache_sources

```python
result = runtime.from_cache_sources(
    [source_a, source_b],
    base_cache_dir,
    reader_config=ReaderConfig(seed=42),
    writer_config=WriterConfig(),
)

match result:
    case CacheSourcesDatasetSuccess(dataset, results):
        ...
    case CacheSourcesDatasetError(results, message):
        ...
```

The factory asks Rust to generate cache paths, reuse valid existing caches, create missing caches, and then load every successful cache path. It returns `CacheSourcesDatasetSuccess(dataset, results)` when at least one cache was loaded, or `CacheSourcesDatasetError(results, message)` when no cache could be loaded. The `results` list preserves every per-source `CacheWriteSuccess` or `CacheWriteError`, so callers can see what was missing or malformed.

## Samples Metadata

```python
metadata = dataset.samples_metadata()
dataset.set_samples_metadata(metadata)
```

Weights are represented inside a Polars samples metadata `DataFrame`, not as a bare vector. The table is designed to be filtered and edited by metadata:

```text
cache_id | sample_id | <metadata columns...> | weight
```

Required columns:

- `cache_id`: cache position passed to `DatasetRuntime.cached_dataset(...)`.
- `sample_id`: physical row within that cache.
- Metadata columns: one column per metadata field stored in the cache.
- `weight`: positive finite float, default `1.0`.

`samples_metadata` returns a copy. Mutating the Polars frame has no effect until it is passed to `set_samples_metadata`.

`set_samples_metadata` accepts a Polars `DataFrame`. Rust reads and validates the authoritative identity and weight columns:

- Every physical sample must appear exactly once.
- Unknown `(cache_id, sample_id)` pairs are rejected.
- Duplicate `(cache_id, sample_id)` pairs are rejected.
- Every weight must be finite and positive.

Rows may be reordered by Polars operations before calling `set_samples_metadata`; Rust maps by `(cache_id, sample_id)`, not by row position.

Metadata columns are included for ergonomic filtering and auditing. They are not trusted for sample identity; `cache_id` and `sample_id` are the identity fields.

## Iteration

```python
for sample in dataset:
    reveal_type(sample)  # CachedSample
```

`CachedSample` contains:

- `data`: bytes
- `metadata`: dict[str, bool | int | float | str]
- `cache_id`: int
- `sample_id`: int

The epoch length is the physical sample count across all loaded caches.

## PyTorch Adapter

```python
torch_dataset = dataset.to_torch_iterable_dataset()
loader = torch.utils.data.DataLoader(torch_dataset, batch_size=None, num_workers=0)
```

The adapter is available only when PyTorch is installed. It returns a sized `torch.utils.data.IterableDataset` view that yields `CachedSample` objects and implements `__len__`.

DatasetRT does not decode tensors in the adapter. Payload decoding remains in Python, usually inside your collate function, transform, or training loop.

Keep PyTorch `DataLoader(num_workers=0)` with this adapter. Use `DatasetRuntime(num_workers=N)` for parallel cache reads; the adapter rejects PyTorch worker processes to avoid duplicated iterable streams.
