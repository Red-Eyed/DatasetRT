use super::*;
use crate::test_support::TestDirectory;

/// Disable observability while retaining the production two-job scheduling window.
pub(super) fn writer_config(pool: Arc<WorkerPool>) -> CacheResult<WriterConfig> {
    Ok(WriterConfig {
        pool,
        max_shard_bytes: MaxShardBytes::new(1024)?,
        prefetch_size: PrefetchSize::new(2)?,
        num_workers: NumWorkers::new(2)?,
        shard_compression: ShardCompression::None,
        show_progress: false,
        validate_cache: false,
        reuse_existing: false,
        profiler: WriterProfilerConfig {
            enabled: false,
            path: PathBuf::new(),
        },
    })
}

/// Produce distinguishable samples without relying on disk or worker completion timing.
fn sample(sequence: u64) -> WriterInput {
    WriterInput {
        data: sequence.to_le_bytes().to_vec(),
        metadata: BTreeMap::new(),
    }
}

/// A future completion holds its credit until the delayed first sample commits.
#[test]
fn reordered_sample_holds_its_scheduling_credit() -> CacheResult<()> {
    let directory = TestDirectory::new()?;
    let path = directory.path().join("cache");
    let config = writer_config(WorkerPool::new(NumWorkers::new(2)?)?)?;
    let builder = CacheBuilder::create(
        path.clone(),
        "test".to_string(),
        1024,
        ShardCompression::None,
    )?;
    let mut pipeline = SampleWritePipeline::new(
        config.pool.clone(),
        builder,
        WriteProgress::new(false, "test", None),
        "test".to_string(),
        WriterProfiler::new(&config.profiler),
        config.prefetch_size,
        config.num_workers,
    )?;
    for sequence in [1, 0] {
        pipeline.result_sender.send(Ok(SerializedSample {
            sequence,
            input: sample(sequence),
        }))?;
    }
    pipeline.in_flight = 2;
    pipeline.commit_completed(pipeline.result_receiver.recv()??)?;
    assert_eq!(pipeline.pending.len(), 1);
    assert_eq!(pipeline.next_sequence, 0);
    pipeline.wait_for_credit(|receiver| receiver.recv()?)?;
    assert_eq!(pipeline.next_sequence, 2);
    assert!(pipeline.pending.is_empty());
    assert_eq!(pipeline.in_flight, 0);
    pipeline.builder.finish()?;
    let cache = load_cache(path, true, 0)?;
    assert_eq!(cache.sample_count(), 2);
    let mut readers = crate::storage::ShardReaderCache::new();
    for sequence in 0..2 {
        assert_eq!(
            cache.read_sample_with_cache(sequence, &mut readers)?.data,
            (sequence as u64).to_le_bytes()
        );
    }
    Ok(())
}

/// Both sides of the rename are synced, and an uncertain publication remains inspectable.
#[test]
fn publication_sync_failures_retain_completed_cache() -> CacheResult<()> {
    for fail_at in [None, Some(0), Some(1)] {
        let directory = TestDirectory::new()?;
        let source = directory.path().join("tmp").join("cache");
        let destination = directory.path().join("cache");
        let mut builder = CacheBuilder::create(
            source.clone(),
            "test".to_string(),
            1024,
            ShardCompression::None,
        )?;
        builder.push_sample(b"sample".to_vec(), BTreeMap::new())?;
        builder.finish()?;
        let mut synced = Vec::new();
        let result = publish_cache_with_sync(&source, &destination, |path| {
            assert!(!source.exists());
            assert!(destination.join("manifest.json").is_file());
            let position = synced.len();
            synced.push(path.to_path_buf());
            if fail_at == Some(position) {
                return Err(std::io::Error::other("injected directory sync failure").into());
            }
            sync_directory(path)
        });
        match fail_at {
            None => result?,
            Some(_) => {
                let error = result.err().ok_or(CacheError::WorkerFailed)?;
                let message = error.to_string();
                assert!(message.contains("publication durability is unconfirmed"));
                assert!(message.contains("injected directory sync failure"));
                assert!(message.contains(&destination.display().to_string()));
            }
        }
        assert_eq!(synced.first().map(PathBuf::as_path), source.parent());
        if fail_at != Some(0) {
            assert_eq!(synced.get(1).map(PathBuf::as_path), destination.parent());
        }
        assert_eq!(load_cache(destination, true, 0)?.sample_count(), 1);
    }
    Ok(())
}
