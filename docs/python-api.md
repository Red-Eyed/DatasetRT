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
- With `shuffle=True`, each Python iterator snapshots the current weight vector and the next finite window of the deterministic draw stream.
- With `shuffle=False`, samples are emitted from a cyclic window over active table order.
- Changing weights or metadata after iterator construction affects future iterators, not existing ones.

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
metadata = dataset.get_metadata()
dataset.update_metadata(metadata)
```

Active samples and weights are represented inside a Polars metadata `DataFrame`, not as a bare vector. The table is designed to be filtered and edited by metadata:

```text
cache_id | sample_id | <metadata columns...> | weight
```

Required columns:

- `cache_id`: cache position passed to `DatasetRuntime.cached_dataset(...)`.
- `sample_id`: physical row within that cache.
- Metadata columns: one column per metadata field stored in the cache.
- `weight`: positive finite float, default `1.0`.

`get_metadata` returns a copy. Mutating the Polars frame has no effect until it is passed to `update_metadata`.

`update_metadata` accepts a Polars `DataFrame`. Rust reads and validates the authoritative identity, stored metadata, and weight columns:

- At least one physical sample must be included.
- Unknown `(cache_id, sample_id)` pairs are rejected.
- Stored metadata columns must be present.
- Every weight must be finite and positive.

Rows may be filtered, duplicated, or reordered by Polars operations before calling `update_metadata`. Removed rows are excluded from future iterators. Duplicated rows repeat the same physical sample in the active row space, and non-shuffled iterators follow the active table order exactly.

Additional columns are preserved in the active metadata table. They will be returned by the next `get_metadata()` call.

`dataset.set_epoch_len(n)` changes how many samples future iterators emit before
stopping. The active metadata table remains the sampling population. Ordered
reads continue through the active table cyclically across iterator boundaries;
shuffled reads continue one deterministic multinomial draw stream. Updating
metadata resets the cursor or draw stream without changing epoch length.

For a quick development run:

```python
dataset.update_metadata(dataset.get_metadata().head(100))
```

Metadata columns are included for ergonomic filtering and auditing. They are not trusted for sample identity; `cache_id` and `sample_id` are the identity fields.

<!-- BEGIN GENERATED: Public Python API -->

_Generated from public docstrings in `dataset_rt/api.py`._

### `CacheInput`

One sample yielded by a `CacheSource`.

`data` is an already-serialized payload. `metadata` is stored separately in
Arrow-compatible columns and is available for weighting/filtering.

Fields:
- `data: BytesLike`: Bytes-like payload to store in DatasetRT shards.
- `metadata: Metadata`: Primitive metadata columns for this physical sample.

### `CacheSource`

Synchronous source protocol consumed by the Rust cache writer.

Fields:
- `name: str`: Plain source name used by Rust when generating `base_cache_dir/name`.

#### `CacheSource.__iter__() -> Iterator[CacheInput]`

Yield cache inputs synchronously.

DatasetRT does not use Python threads or queues. Rust pulls from this
iterator and owns bounded prefetching, worker threads, and commits.

### `CacheSourcesDatasetError`

Dataset creation result when no source produced a cache.

Fields:
- `results: list[CacheWriteResult]`: Per-source write outcomes explaining why no dataset was loaded.
- `message: str`: Human-readable summary of the failed dataset creation.

### `CacheSourcesDatasetResult`

```python
CacheSourcesDatasetResult = CacheSourcesDatasetSuccess | CacheSourcesDatasetError
```

Best-effort result returned by `DatasetRuntime.from_cache_sources`.

### `CacheSourcesDatasetSuccess`

Dataset creation result when at least one source produced a cache.

Fields:
- `dataset: CachedDataset`: Dataset loaded from all successful cache writes.
- `results: list[CacheWriteResult]`: Per-source write outcomes, including failures.

### `CacheWriteError`

Failed per-source cache write result.

Fields:
- `source_name: str`: Source name, or a source index label if the name could not be read.
- `message: str`: Human-readable reason the source was not written.

### `CacheWriteResult`

```python
CacheWriteResult = CacheWriteSuccess | CacheWriteError
```

Per-source cache write outcome returned by `DatasetRuntime.write_cache`.

### `CacheWriteSuccess`

Successful per-source cache write result.

Fields:
- `source_name: str`: Source name used to derive the cache path.
- `path: Path`: Published cache directory.

### `CachedDataset`

Synchronous iterable view over Rust-owned dataset state.

Users do not construct this class directly; use
`DatasetRuntime.cached_dataset` or `DatasetRuntime.from_cache_sources`.
Rust owns cache validation, active metadata state, epoch planning, sampling,
bounded prefetching, and iterator cancellation.

Fields:
- `cache_paths: list[Path]`: Immutable cache directories loaded by this dataset, in `cache_id` order.
- `reader_config: ReaderConfig`: Reader configuration used when this dataset was loaded.

#### `CachedDataset.__init__()`

Reject direct construction because every dataset requires a runtime.

#### `CachedDataset.__iter__() -> Iterator[CachedSample]`

Create an iterator from the current active metadata table and epoch length.

Iterator construction snapshots Rust runtime state. Later `set_epoch_len`
or `update_metadata` calls affect future iterators, not this iterator.

With `ReaderConfig.shuffle=True`, Rust creates a deterministic weighted
multinomial stream over active metadata rows. With `shuffle=False`,
iteration follows active metadata table order from the current cyclic
cursor, including duplicate rows.

#### `CachedDataset.__len__() -> int`

Return the number of samples emitted by each future iterator.

This value changes after `set_epoch_len` or `update_metadata`. Existing
iterators keep their own snapshot even if this value changes mid-epoch.

#### `CachedDataset.set_epoch_len(epoch_len: int) -> None`

Set how many samples each future iterator emits before stopping.

`epoch_len` must be at least 1. The active metadata table remains the
sampling population; this method changes only the finite window length.
Existing iterators keep their snapshot.

With `ReaderConfig.shuffle=False`, future iterators continue from the
current cyclic active-row cursor. With `ReaderConfig.shuffle=True`,
future iterators continue from the current multinomial draw stream.
`update_metadata` resets the cursor or draw stream without changing
`epoch_len`.

#### `CachedDataset.to_torch_iterable_dataset() -> SizedTorchIterableDataset`

Return a sized `torch.utils.data.IterableDataset` view.

The adapter yields the same `CachedSample` objects as DatasetRT's
normal iterator and implements `__len__`, so PyTorch consumers can use
it with `DataLoader` while keeping domain decoding in Python.

Raises:
    ImportError: If PyTorch is not installed in the active environment.

#### `CachedDataset.samples_metadata() -> pl.DataFrame`

Compatibility alias for `get_metadata`.

Returns the same active metadata table as `get_metadata`. Prefer
`get_metadata` in new code.

#### `CachedDataset.get_metadata() -> pl.DataFrame`

Return the in-memory metadata table that controls future iterators.

Contract:

- Returns a Polars `DataFrame` copy; editing it does not mutate the
  dataset until the whole frame is passed to `update_metadata`.
- Columns are `cache_id`, `sample_id`, every metadata column stored in
  the cache, `weight`, and any extra columns preserved from the previous
  `update_metadata` call.
- `cache_id` is the cache path position passed to
  `DatasetRuntime.cached_dataset`; `sample_id` is the physical row
  inside that cache.
- Each row is one active sampling row. Duplicate `(cache_id, sample_id)`
  rows are allowed and represent repeated entries for the same physical
  sample.
- `len(dataset)` equals the configured epoch length. By default it
  matches the active row count, including duplicate rows.
- Cache files are not read or rewritten by this method beyond exporting
  the current Rust-owned in-memory table.

#### `CachedDataset.set_samples_metadata(metadata: pl.DataFrame) -> None`

Compatibility alias for `update_metadata`.

Applies the same validation and runtime-only update semantics as
`update_metadata`. Prefer `update_metadata` in new code.

#### `CachedDataset.update_metadata(metadata: pl.DataFrame) -> None`

Replace the Rust-owned active metadata table for future iterators.

Input contract:

- `metadata` must be a Polars `DataFrame`.
- Required columns are `cache_id`, `sample_id`, every metadata column
  stored in the cache, and `weight`.
- `cache_id` and `sample_id` must be non-null integer columns that map
  to known physical cache samples.
- Stored metadata columns must be present with the same Arrow types as
  the immutable cache metadata schema.
- `weight` must be a non-null numeric column, and every value must be
  positive and finite.
- Extra columns are allowed and preserved in runtime memory; they are
  returned by the next `get_metadata` call.

Row semantics:

- Rows absent from `metadata` are removed from the active sampling
  space and excluded from future iterators.
- Duplicate `(cache_id, sample_id)` rows are allowed. Each duplicate is
  a separate active row that points to the same immutable physical
  sample, useful for row-duplication balancing or OHEM.
- `len(dataset)` keeps the current epoch length.
- With `ReaderConfig(shuffle=False)`, future iterators emit active rows
  exactly in table order, including duplicates.
- With `ReaderConfig.shuffle=True`, future iterators sample with
  replacement over active rows using each row's `weight`.

Mutation boundary:

- Rust validates the full table before replacing runtime state; a
  validation error leaves the previous active table intact.
- The update is runtime-only and does not rewrite `metadata.arrow`,
  `index.bin`, shards, or manifests.
- Iterators created before this call keep their existing snapshot;
  iterators created after this call use the new active table.

### `CachedSample`

One sample emitted by `CachedDataset` iteration.

Fields:
- `data: bytes`: Payload bytes loaded from the immutable cache.
- `metadata: dict[str, MetadataValue]`: Metadata row associated with this physical sample.
- `cache_id: int`: Position of the source cache passed to `DatasetRuntime.cached_dataset`.
- `sample_id: int`: Physical sample row within the source cache.

### `DatasetRuntime`

Owner of the fixed Rust worker pool used by DatasetRT operations.

Create one runtime per process or training job, then call its methods to
write caches, load datasets, and iterate samples. The worker count is chosen
once at construction and reused; per-operation APIs do not resize the pool.

#### `DatasetRuntime.__init__(*, num_workers: int)`

Create exactly `num_workers` reusable Rust worker threads.

`num_workers` must be positive. Rust validates the value before creating
the native pool. The pool is owned by this runtime and kept alive by any
datasets loaded through it.

#### `DatasetRuntime.num_workers`

Return the fixed worker count selected when this runtime was created.

#### `DatasetRuntime.write_cache(sources: CacheSource | list[CacheSource], path: str | Path, *, writer_config: WriterConfig = DEFAULT_WRITER_CONFIG) -> list[CacheWriteResult]`

Write one immutable cache per source under `path`.

`sources` may be one `CacheSource` or a list of sources. Each source
must expose a plain `name` and yield `CacheInput` values. Rust writes
source `name` under `path / name`, validates a stable metadata schema,
stores payload bytes in shards, writes `metadata.arrow` and `index.bin`,
then publishes a manifest only after the cache is complete.

Returns one `CacheWriteSuccess` or `CacheWriteError` per source in input
order. Per-source failures are reported as values instead of exceptions
when Rust can handle them cleanly.

#### `DatasetRuntime.cached_dataset(paths: Sequence[str | Path], *, reader_config: ReaderConfig) -> CachedDataset`

Load immutable cache directories into a `CachedDataset`.

`paths` order defines stable `cache_id` values for the dataset.
DatasetRT validates manifests, schemas, metadata/index shape, and shard
lengths while loading. Expensive checksum hashing is controlled by
`reader_config.validate_cache`.

The returned dataset keeps this runtime's Rust worker pool alive and
uses it for every future iterator.

#### `DatasetRuntime.from_cache_sources(sources: CacheSource | list[CacheSource], path: str | Path, *, reader_config: ReaderConfig, writer_config: WriterConfig = DEFAULT_WRITER_CONFIG) -> CacheSourcesDatasetResult`

Create or reuse source caches, then load all successful cache paths.

Existing complete caches under `path / source.name` are reused. Missing
caches are written with `writer_config`. The method then loads every
successful cache with `reader_config`.

Returns `CacheSourcesDatasetSuccess` when at least one cache is loaded;
returns `CacheSourcesDatasetError` when every source failed or no source
was provided. The result always includes per-source write outcomes so
callers can audit partial success.

### `ReaderConfig`

Configuration for Rust-owned dataset reading and sampling.

Fields:
- `seed: int`: Seed used for deterministic shuffled epoch planning.
- `prefetch_size: int`: Capacity of Rust's bounded reader result queue.
- `shuffle: bool`: Whether future iterators use deterministic weighted sampling.
- `validate_cache: bool`: Whether cache checksums are verified while loading metadata and indexes.

### `ShardCompression`

Compression policy requested for payload shards.

`ratio` is part of the explicit policy object so callers and manifests use
the same structured shape. `ratio` is advisory metadata for compressed
algorithms; Rust validates `algo="none"` with `ratio == 1.0`.

Fields:
- `algo: CompressionAlgo`: Compression algorithm to apply independently to each payload record.
- `ratio: float`: Expected compression ratio; `algo='none'` requires exactly `1.0`.

### `SizedTorchIterableDataset`

Sized PyTorch iterable view returned by `to_torch_iterable_dataset`.

#### `SizedTorchIterableDataset.__iter__() -> Iterator[CachedSample]`

Yield cached samples in DatasetRT iterator order.

#### `SizedTorchIterableDataset.__len__() -> int`

Return the physical sample count visible to PyTorch.

### `WriterConfig`

Configuration for Rust-owned cache writing.

Fields:
- `prefetch_size: int`: Maximum number of writer tasks/results buffered by Rust.
- `max_shard_bytes: int`: Target shard byte size before Rust rotates to a new shard.
- `shard_compression: ShardCompression`: Per-record payload compression policy for new shards.
- `show_progress: bool`: Whether Rust renders cache write progress with samples/s and MB/s.
- `validate_cache: bool`: Whether reused existing caches are checksum-validated before loading.
- `profiler: WriterProfilerConfig`: Optional JSON writer-stage profiler configuration.

### `WriterProfilerConfig`

Optional writer profiler output.

Profiling is disabled by default. When enabled, Rust writes a structured
JSON summary at `path` after successful writes and handled failures such as
Ctrl-C.

Fields:
- `enabled: bool`: Whether Rust collects and writes writer-stage timing statistics.
- `path: Path`: JSON summary path used when profiling is enabled.

<!-- END GENERATED: Public Python API -->

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

The default epoch length is the active metadata row count.
`dataset.set_epoch_len(n)` changes how many samples future iterators emit.
Each iterator snapshots the active metadata table and epoch length when it is
created, so `set_epoch_len` and `update_metadata` affect future iterators and do
not rewrite an iterator that is already mid-epoch.

## PyTorch Adapter

```python
torch_dataset = dataset.to_torch_iterable_dataset()
loader = torch.utils.data.DataLoader(torch_dataset, batch_size=None, num_workers=0)
```

The adapter is available only when PyTorch is installed. It returns a sized `torch.utils.data.IterableDataset` view that yields `CachedSample` objects and implements `__len__`.

DatasetRT does not decode tensors in the adapter. Payload decoding remains in Python, usually inside your collate function, transform, or training loop.

Keep PyTorch `DataLoader(num_workers=0)` with this adapter. Use `DatasetRuntime(num_workers=N)` for parallel cache reads; the adapter rejects PyTorch worker processes to avoid duplicated iterable streams.
