# Format v4 compatibility fixtures

Generated locally using the unmodified DatasetRT 0.5.9 release wheel built from
the repository before the v5 changes, with one worker and `show_progress=False`.
Each named cache uses `NoCompression()`, `Lz4Fast()`, or `Zstd()` respectively.

Each source yields:

```python
CacheInput(b"alpha", {"index": 0, "split": "train"})
CacheInput(b"beta", {"index": 1, "split": "train"})
```

These artifacts retain genuine v4 metadata envelopes and checksums. Tests must
read them directly rather than relabel v5 manifests as v4.
