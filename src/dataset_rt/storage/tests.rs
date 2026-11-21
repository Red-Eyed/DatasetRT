use super::*;
use crate::test_support::TestDirectory;

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
