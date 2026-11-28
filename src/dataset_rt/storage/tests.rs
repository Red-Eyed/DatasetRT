use super::*;
use crate::test_support::TestDirectory;
use arrow_schema::{Field, Schema};

/// Mismatched Arrow types cannot be interpreted as the manifest's primitive metadata kind.
#[test]
fn metadata_readers_reject_missing_and_mistyped_columns() -> CacheResult<()> {
    let kinds = [
        MetadataKind::Bool,
        MetadataKind::Int,
        MetadataKind::Float,
        MetadataKind::String,
    ];
    for kind in kinds {
        let wrong: ArrayRef = match kind {
            MetadataKind::Int => Arc::new(BooleanArray::from(vec![true])),
            _ => Arc::new(Int64Array::from(vec![1])),
        };
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                wrong.data_type().clone(),
                false,
            )])),
            vec![wrong],
        )?;
        let schema = vec![MetadataField {
            name: "value".to_string(),
            kind: kind.clone(),
        }];
        let error = read_metadata_row(&batch, &schema, 0)
            .err()
            .ok_or(CacheError::WorkerFailed)?;
        assert!(error.to_string().contains("metadata 'value' is not"));
        let wrong_value = match kind {
            MetadataKind::Int => MetadataValue::Bool(true),
            _ => MetadataValue::Int(1),
        };
        assert!(matches!(
            build_array(0, &kind, &[vec![wrong_value]]),
            Err(CacheError::InvalidInput(message)) if message == "metadata kind drifted"
        ));
        assert!(matches!(
            build_array(1, &kind, &[Vec::new()]),
            Err(CacheError::InvalidInput(message)) if message == "metadata column is out of range"
        ));
    }
    let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let schema = vec![MetadataField {
        name: "missing".to_string(),
        kind: MetadataKind::Int,
    }];
    assert!(read_metadata_row(&batch, &schema, 0).is_err());
    assert!(read_u64(&[0; 7]).is_err());
    assert!(read_index_entry(&[0; 23]).is_err());
    Ok(())
}

/// A failed shard open reports its path, and repeated fixture construction never overwrites data.
#[test]
fn builder_and_shard_creation_failures_preserve_existing_paths() -> CacheResult<()> {
    let directory = TestDirectory::new()?;
    let path = directory.path().join("existing");
    std::fs::create_dir(&path)?;
    assert!(CacheBuilder::create(path, "test".to_string(), 1024, ShardCompression::None).is_err());
    let path = directory.path().join("blocked-shard");
    let _builder = CacheBuilder::create(
        path.clone(),
        "test".to_string(),
        1024,
        ShardCompression::None,
    )?;
    std::fs::create_dir(path.join("shards").join("000001.bin"))?;
    let error = OpenShard::create(&path, 1)
        .err()
        .ok_or(CacheError::WorkerFailed)?;
    assert!(error.to_string().contains("create shard"));
    assert!(!path.join("manifest.json").exists());
    Ok(())
}

struct FailingWriter {
    fail_write: bool,
}

impl Write for FailingWriter {
    /// Fail only the operation selected by the fixture.
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.fail_write {
            return Err(std::io::Error::other("injected manifest write failure"));
        }
        Ok(bytes.len())
    }

    /// Exercise explicit flush errors independently of serialization errors.
    fn flush(&mut self) -> std::io::Result<()> {
        Err(std::io::Error::other("injected manifest flush failure"))
    }
}

/// A completed fixture supplies a real manifest without duplicating format defaults.
fn completed_cache(path: PathBuf) -> CacheResult<Manifest> {
    let mut builder = CacheBuilder::create(path, "test".to_string(), 1024, ShardCompression::None)?;
    builder.push_sample(b"payload".to_vec(), BTreeMap::new())?;
    Ok(builder.finish()?.0)
}

/// Buffered tail writes and explicit flush failures cannot masquerade as successful manifests.
#[test]
fn manifest_output_failures_are_returned() -> CacheResult<()> {
    let directory = TestDirectory::new()?;
    let manifest = completed_cache(directory.path().join("cache"))?;
    let mut buffered = BufWriter::new(FailingWriter { fail_write: true });
    let error = write_manifest_contents(&mut buffered, &manifest)
        .err()
        .ok_or(CacheError::WorkerFailed)?;
    assert!(error
        .to_string()
        .contains("injected manifest write failure"));
    let error = write_manifest_contents(&mut FailingWriter { fail_write: false }, &manifest)
        .err()
        .ok_or(CacheError::WorkerFailed)?;
    assert!(error
        .to_string()
        .contains("injected manifest flush failure"));
    Ok(())
}

/// File creation failures retain the publication-marker path and never produce a valid cache.
#[test]
fn failed_manifest_creation_has_context() -> CacheResult<()> {
    let directory = TestDirectory::new()?;
    let path = directory.path().join("cache");
    let mut builder = CacheBuilder::create(
        path.clone(),
        "test".to_string(),
        1024,
        ShardCompression::None,
    )?;
    builder.push_sample(b"payload".to_vec(), BTreeMap::new())?;
    std::fs::create_dir(path.join("manifest.json"))?;
    let error = builder.finish().err().ok_or(CacheError::WorkerFailed)?;
    assert!(error.to_string().contains("write manifest"));
    assert!(error
        .to_string()
        .contains(&path.join("manifest.json").display().to_string()));
    assert!(load_cache(path, false, 0).is_err());
    Ok(())
}

/// Chunked table output remains byte-deterministic and validates through the production reader.
#[test]
fn streamed_cache_is_deterministic_and_valid() -> CacheResult<()> {
    let directory = TestDirectory::new()?;
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    for path in [&first, &second] {
        completed_cache(path.clone())?;
        let cache = load_cache(path.clone(), true, 0)?;
        assert_eq!(cache.sample_count(), 1);
        let sample = cache.read_sample_with_cache(0, &mut ShardReaderCache::new())?;
        assert_eq!(sample.data, b"payload");
        assert!(sample.metadata.is_empty());
    }
    for name in [
        "manifest.json",
        "metadata.arrow",
        "index.bin",
        "shards/000000.bin",
    ] {
        assert_eq!(
            std::fs::read(first.join(name))?,
            std::fs::read(second.join(name))?
        );
    }
    Ok(())
}

/// Random reads cross batch/cache boundaries without initializing a cache-wide row vector.
#[test]
fn payload_only_metadata_is_lazy_and_worker_bounded() -> CacheResult<()> {
    let directory = TestDirectory::new()?;
    let mut caches = Vec::new();
    for name in ["first", "second"] {
        let path = directory.path().join(name);
        let mut builder =
            CacheBuilder::create(path.clone(), name.to_string(), 1024, ShardCompression::None)?;
        for row in 0..65_537 {
            builder.push_sample(
                Vec::new(),
                BTreeMap::from([("index".to_string(), MetadataValue::Int(row))]),
            )?;
        }
        builder.finish()?;
        let cache = load_cache(path, false, 0)?;
        assert!(cache.metadata_index.get().is_none());
        caches.push(cache);
    }
    let mut readers = ShardReaderCache::new();
    for row in [65_536, 0, 65_535, 12, 65_536] {
        for cache in &caches {
            let sample = cache.read_sample_with_cache(row, &mut readers)?;
            assert!(sample.data.is_empty());
            assert_eq!(sample.metadata, vec![MetadataValue::Int(row as i64)]);
            assert!(cache
                .metadata_rows
                .lock()
                .map_err(|_| CacheError::WorkerFailed)?
                .is_none());
            let batch = readers
                .metadata_batch
                .as_ref()
                .ok_or(CacheError::WorkerFailed)?;
            assert!(Arc::ptr_eq(
                &batch.source,
                cache.metadata_index.get().ok_or(CacheError::WorkerFailed)?
            ));
        }
    }
    let first = caches.first().ok_or(CacheError::WorkerFailed)?;
    first.read_sample_with_cache(0, &mut readers)?;
    let reopened = load_cache(first.path.clone(), false, 0)?;
    reopened.read_sample_with_cache(0, &mut readers)?;
    assert!(Arc::ptr_eq(
        &readers
            .metadata_batch
            .as_ref()
            .ok_or(CacheError::WorkerFailed)?
            .source,
        reopened
            .metadata_index
            .get()
            .ok_or(CacheError::WorkerFailed)?
    ));
    let validated = load_cache(directory.path().join("first"), true, 0)?;
    assert!(validated
        .metadata_rows
        .lock()
        .map_err(|_| CacheError::WorkerFailed)?
        .is_none());
    Ok(())
}

/// V4 fixtures were produced by the unmodified 0.5.9 writer, not by the v5 implementation.
#[test]
fn existing_v4_codecs_remain_readable() -> CacheResult<()> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/format-v4");
    for name in ["plain", "lz4", "zstd"] {
        let cache = load_cache(directory.join(name), true, 0)?;
        assert!(matches!(cache.manifest, Manifest::V4(_)));
        for (row, expected) in [b"alpha".as_slice(), b"beta".as_slice()]
            .into_iter()
            .enumerate()
        {
            let sample = cache.read_sample_with_cache(row, &mut ShardReaderCache::new())?;
            assert_eq!(sample.data, expected);
            assert_eq!(
                sample.metadata,
                vec![
                    MetadataValue::Int(row as i64),
                    MetadataValue::String("train".to_string())
                ]
            );
        }
        let wire = serde_json::to_value(cache.manifest)?;
        assert_eq!(wire.get("format_version"), Some(&serde_json::json!(4)));
    }
    Ok(())
}
