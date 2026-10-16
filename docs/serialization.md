---
title: Cache Python Records with Pickle or Dill
description: Pack Python dictionaries, arrays, and nested training records with pickle or dill in an iterable DatasetRT source, and unpack them in your read transform.
---

# Cache Python records with pickle or dill

The usual data source is a named Python iterable. It preprocesses one sample,
packs its Python record or dictionary with `pickle` or `dill`, and yields
`CacheInput(data=packed_record, metadata=...)`. Your read transform unpacks
`sample.data` into the structure used by training.

This lets a record contain large arrays, tensors, and nested dictionaries while
DatasetRT's storage and sampling remain independent of those Python types.
Use the [complete caching example](cache-training-data.md) or the
[PyTorch transform example](pytorch-dataloader.md) to see both boundaries.

## Packing and unpacking

`pickle` is included with Python. Pair `pickle.dumps(record)` in your source
with `pickle.loads(sample.data)` when reading. The examples use an explicit
`TypedDict` shape and validate the decoded record once with a Pydantic
`TypeAdapter` before using it.

For application objects that require `dill`, install it with `uv add dill` and
pair `dill.dumps(record)` with `dill.loads(sample.data)`. Use the same serializer
and compatible application types in writers and readers. Custom classes need
to remain available when their stored objects are decoded.

Pack records as the source yields them rather than building a serialized copy
of the full dataset in memory. Per-sample packing and unpacking allocate memory
for that sample; budget for the largest decoded record as well as bounded
prefetch and batch buffers. Validate complex record shapes at your decoding
boundary according to your application's schema.

## Two serialization layers

DatasetRT has two serialization layers:

- User/project serialization into a bytes payload.
- DatasetRT cache serialization into immutable cache files.

Your data source performs the first layer. Pickle and dill are common choices;
encoded images, NumPy `.npy`, MessagePack, protobuf, or application-specific
formats can also supply the payload bytes.

Rust always owns the second layer:

- Payload bytes are placed into shards by Rust.
- Metadata is typed and written by Rust.
- `metadata.arrow`, `index.bin`, shard files, and `manifest.json` are written by Rust.
- Checksums are calculated and verified by Rust.
- Deserialization from cache shards back into Python `bytes` is performed by Rust.

This boundary keeps DatasetRT framework-independent while still allowing project-specific payload formats. The cache runtime does not need to know whether the bytes represent JPEG, NumPy `.npy`, Arrow IPC, MessagePack, protobuf, or a custom format.

## Payload contract

`CacheInput.data` accepts:

- `bytes`
- `bytearray`
- `memoryview`

DatasetRT accepts the serialized bytes rather than the original Python object.
Your source chooses the serializer and produces the bytes before yielding
`CacheInput`; DatasetRT returns those bytes without invoking pickle or dill.

## Metadata Contract

Metadata is not an opaque payload. It is part of the cache’s queryable, sample-level identity and sampling surface, so Rust stores it separately as Arrow-compatible primitive columns.

Copy primitive labels, split names, or scores needed for population selection
into `CacheInput.metadata`. Keep the full record in `CacheInput.data`. This lets
Polars filter and weight samples without deserializing every large payload.
