use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyByteArrayMethods, PyDict, PyIterator, PyList, PyString, PyTuple};
use sha2::{Digest, Sha256};

use crate::storage::{load_cache, CacheBuilder};
use crate::types::{
    CacheError, CacheResult, CompressionAlgo, MaxShardBytes, MetadataValue, NumThreads,
    PrefetchSize, ShardCompression,
};

#[derive(Clone)]
struct WriterConfig {
    max_shard_bytes: MaxShardBytes,
    prefetch_size: PrefetchSize,
    num_threads: NumThreads,
    shard_compression: ShardCompression,
    reuse_existing: bool,
}

struct QueuedSample {
    sequence: u64,
    input: WriterInput,
}

struct WriterInput {
    data: Vec<u8>,
    metadata: BTreeMap<String, MetadataValue>,
}

struct SerializedSample {
    sequence: u64,
    input: WriterInput,
}

enum CommitMessage {
    Sample(SerializedSample),
    Abort(String),
}

pub fn write_cache(
    sources: Bound<'_, PyAny>,
    base_cache_dir: String,
    max_shard_bytes: u64,
    prefetch_size: usize,
    num_threads: usize,
    shard_compression: Bound<'_, PyAny>,
    reuse_existing: bool,
) -> CacheResult<Vec<String>> {
    let config = WriterConfig {
        max_shard_bytes: MaxShardBytes::new(max_shard_bytes)?,
        prefetch_size: PrefetchSize::new(prefetch_size)?,
        num_threads: NumThreads::new(num_threads)?,
        shard_compression: extract_shard_compression(&shard_compression)?,
        reuse_existing,
    };
    let base_cache_dir = PathBuf::from(base_cache_dir);

    if let Ok(source_list) = sources.cast::<PyList>() {
        return write_source_list(source_list, &base_cache_dir, &config);
    }

    let path = write_one_source(sources, &base_cache_dir, 0, &config)?;
    Ok(vec![path_to_string(&path)])
}

fn write_source_list(
    sources: &Bound<'_, PyList>,
    base_cache_dir: &Path,
    config: &WriterConfig,
) -> CacheResult<Vec<String>> {
    let mut written = Vec::with_capacity(sources.len());
    let mut seen_paths = BTreeSet::new();

    for (index, source) in sources.iter().enumerate() {
        let path = match write_one_source(source, base_cache_dir, index, config) {
            Ok(path) => path,
            Err(error) => {
                cleanup_published_caches(&written)?;
                return Err(error);
            }
        };
        if !seen_paths.insert(path.clone()) {
            cleanup_published_caches(&written)?;
            return Err(CacheError::InvalidInput(format!(
                "duplicate generated cache path: {}",
                path.display()
            )));
        }
        written.push(path);
    }

    Ok(written
        .iter()
        .map(|path| path_to_string(path))
        .collect::<Vec<_>>())
}

fn write_one_source(
    source: Bound<'_, PyAny>,
    base_cache_dir: &Path,
    source_index: usize,
    config: &WriterConfig,
) -> CacheResult<PathBuf> {
    let source_name = extract_source_name(&source)?;
    let cache_path = cache_path_for_source(base_cache_dir, &source_name, source_index);
    if config.reuse_existing && cache_path.exists() {
        load_cache(cache_path.clone())?;
        return Ok(cache_path);
    }

    let temp_path = temp_cache_path_for_source(base_cache_dir, &source_name, source_index);
    prepare_cache_paths(&cache_path, &temp_path)?;
    let builder = match CacheBuilder::create(
        temp_path.clone(),
        source_name,
        config.max_shard_bytes.as_u64(),
        config.shard_compression.clone(),
    ) {
        Ok(builder) => builder,
        Err(error) => {
            cleanup_temp_cache(&temp_path)?;
            return Err(error);
        }
    };
    let iterator = PyIterator::from_object(&source).map_err(py_error)?;
    if let Err(error) = write_with_pipeline(iterator, builder, config) {
        cleanup_temp_cache(&temp_path)?;
        return Err(error);
    }
    if let Err(error) = publish_cache(&temp_path, &cache_path) {
        cleanup_temp_cache(&temp_path)?;
        return Err(error);
    }
    Ok(cache_path)
}

fn write_with_pipeline(
    iterator: Bound<'_, PyIterator>,
    builder: CacheBuilder,
    config: &WriterConfig,
) -> CacheResult<()> {
    let (input_sender, input_receiver) = bounded(config.prefetch_size.as_usize());
    let (commit_sender, commit_receiver) = bounded(config.prefetch_size.as_usize());
    let commit_handle = spawn_commit_thread(builder, commit_receiver);
    let worker_handles =
        spawn_serialization_workers(input_receiver, commit_sender.clone(), config.num_threads);

    let ingestion_result = ingest_python_samples(iterator, &input_sender, &commit_sender);
    drop(input_sender);
    drop(commit_sender);

    let workers_result = join_workers(worker_handles);
    let commit_result = join_commit(commit_handle);

    ingestion_result?;
    workers_result?;
    commit_result
}

fn ingest_python_samples(
    iterator: Bound<'_, PyIterator>,
    sender: &Sender<QueuedSample>,
    abort_sender: &Sender<CommitMessage>,
) -> CacheResult<()> {
    for (sequence, item) in iterator.enumerate() {
        let sequence = u64::try_from(sequence).map_err(|_| {
            CacheError::InvalidInput("sample sequence does not fit in u64".to_string())
        })?;
        let item = match item
            .map_err(py_error)
            .and_then(|item| extract_cache_input(&item))
        {
            Ok(input) => input,
            Err(error) => {
                let _ = abort_sender.send(CommitMessage::Abort(error.to_string()));
                return Err(error);
            }
        };
        sender
            .send(QueuedSample {
                sequence,
                input: item,
            })
            .map_err(|_| CacheError::WorkerFailed)?;
    }
    Ok(())
}

fn spawn_serialization_workers(
    input_receiver: Receiver<QueuedSample>,
    commit_sender: Sender<CommitMessage>,
    num_threads: NumThreads,
) -> Vec<thread::JoinHandle<CacheResult<()>>> {
    (0..num_threads.as_usize())
        .map(|_| {
            let input_receiver = input_receiver.clone();
            let commit_sender = commit_sender.clone();
            thread::spawn(move || run_serialization_worker(input_receiver, commit_sender))
        })
        .collect()
}

fn run_serialization_worker(
    input_receiver: Receiver<QueuedSample>,
    commit_sender: Sender<CommitMessage>,
) -> CacheResult<()> {
    for queued in input_receiver {
        let serialized = SerializedSample {
            sequence: queued.sequence,
            input: queued.input,
        };
        if commit_sender
            .send(CommitMessage::Sample(serialized))
            .is_err()
        {
            return Ok(());
        }
    }
    Ok(())
}

fn spawn_commit_thread(
    builder: CacheBuilder,
    receiver: Receiver<CommitMessage>,
) -> thread::JoinHandle<CacheResult<()>> {
    thread::spawn(move || commit_serialized_samples(builder, receiver))
}

fn commit_serialized_samples(
    mut builder: CacheBuilder,
    receiver: Receiver<CommitMessage>,
) -> CacheResult<()> {
    let mut pending = BTreeMap::new();
    let mut next_sequence = 0_u64;

    for message in receiver {
        match message {
            CommitMessage::Sample(sample) => {
                pending.insert(sample.sequence, sample);
                commit_ready_samples(&mut builder, &mut pending, &mut next_sequence)?;
            }
            CommitMessage::Abort(message) => return Err(CacheError::InvalidInput(message)),
        }
    }

    if !pending.is_empty() {
        return Err(CacheError::WorkerFailed);
    }

    builder.finish()?;
    Ok(())
}

fn commit_ready_samples(
    builder: &mut CacheBuilder,
    pending: &mut BTreeMap<u64, SerializedSample>,
    next_sequence: &mut u64,
) -> CacheResult<()> {
    while let Some(sample) = pending.remove(next_sequence) {
        builder.push_sample(sample.input.data, sample.input.metadata)?;
        *next_sequence = next_sequence.checked_add(1).ok_or_else(|| {
            CacheError::InvalidInput("sample sequence overflowed u64".to_string())
        })?;
    }
    Ok(())
}

fn join_workers(handles: Vec<thread::JoinHandle<CacheResult<()>>>) -> CacheResult<()> {
    for handle in handles {
        join_worker(handle)?;
    }
    Ok(())
}

fn join_worker(handle: thread::JoinHandle<CacheResult<()>>) -> CacheResult<()> {
    handle.join().map_err(|_| CacheError::WorkerFailed)?
}

fn join_commit(handle: thread::JoinHandle<CacheResult<()>>) -> CacheResult<()> {
    handle.join().map_err(|_| CacheError::WorkerFailed)?
}

fn extract_shard_compression(value: &Bound<'_, PyAny>) -> CacheResult<ShardCompression> {
    let algo = value
        .getattr("algo")
        .map_err(py_error)?
        .extract::<String>()
        .map_err(py_error)?;
    let ratio = value
        .getattr("ratio")
        .map_err(py_error)?
        .extract::<f64>()
        .map_err(py_error)?;
    ShardCompression::new(CompressionAlgo::from_name(&algo)?, ratio)
}

fn extract_source_name(source: &Bound<'_, PyAny>) -> CacheResult<String> {
    let name = source
        .getattr("name")
        .map_err(py_error)?
        .extract::<String>()
        .map_err(py_error)?;
    validate_source_name(&name)?;
    Ok(name)
}

fn validate_source_name(name: &str) -> CacheResult<()> {
    if name.is_empty() {
        return Err(CacheError::InvalidInput(
            "CacheSource.name must not be empty".to_string(),
        ));
    }

    let mut components = Path::new(name).components();
    let first = components.next().ok_or_else(|| {
        CacheError::InvalidInput("CacheSource.name must be a plain path segment".to_string())
    })?;
    if components.next().is_some() || !matches!(first, Component::Normal(_)) {
        return Err(CacheError::InvalidInput(
            "CacheSource.name must be a plain path segment".to_string(),
        ));
    }
    Ok(())
}

fn cache_path_for_source(base_cache_dir: &Path, source_name: &str, source_index: usize) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(source_index.to_le_bytes());
    hasher.update(source_name.as_bytes());
    let digest = hex::encode(hasher.finalize());
    let short_digest = digest.chars().take(12).collect::<String>();
    base_cache_dir.join(format!("{source_name}_{short_digest}"))
}

fn temp_cache_path_for_source(
    base_cache_dir: &Path,
    source_name: &str,
    source_index: usize,
) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(b"tmp");
    hasher.update(source_index.to_le_bytes());
    hasher.update(source_name.as_bytes());
    let digest = hex::encode(hasher.finalize());
    let short_digest = digest.chars().take(12).collect::<String>();
    base_cache_dir.join(format!(".{source_name}_{short_digest}.tmp"))
}

fn prepare_cache_paths(cache_path: &Path, temp_path: &Path) -> CacheResult<()> {
    if cache_path.exists() {
        return Err(CacheError::InvalidInput(format!(
            "cache path already exists: {}",
            cache_path.display()
        )));
    }
    if temp_path.exists() {
        return Err(CacheError::InvalidInput(format!(
            "temporary cache path already exists: {}",
            temp_path.display()
        )));
    }
    Ok(())
}

fn cleanup_temp_cache(temp_path: &Path) -> CacheResult<()> {
    if temp_path.exists() {
        fs::remove_dir_all(temp_path)?;
    }
    Ok(())
}

fn cleanup_published_caches(paths: &[PathBuf]) -> CacheResult<()> {
    for path in paths {
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

fn publish_cache(temp_path: &Path, cache_path: &Path) -> CacheResult<()> {
    fs::rename(temp_path, cache_path)?;
    Ok(())
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn extract_cache_input(item: &Bound<'_, PyAny>) -> CacheResult<WriterInput> {
    let tuple = item.cast::<PyTuple>().map_err(|_| {
        CacheError::InvalidInput("CacheSource must yield CacheInput tuples".to_string())
    })?;
    if tuple.len() != 2 {
        return Err(CacheError::InvalidInput(
            "CacheInput must contain data and metadata".to_string(),
        ));
    }

    let data = extract_data(&tuple.get_item(0).map_err(py_error)?)?;
    let metadata = extract_metadata(&tuple.get_item(1).map_err(py_error)?)?;
    Ok(WriterInput { data, metadata })
}

fn extract_data(value: &Bound<'_, PyAny>) -> CacheResult<Vec<u8>> {
    // Project-specific serialization must happen before CacheInput; here we only accept buffers.
    PyByteArray::from(value)
        .map(|bytes| bytes.to_vec())
        .map_err(|_| CacheError::InvalidInput("data must be bytes-like".to_string()))
}

fn extract_metadata(value: &Bound<'_, PyAny>) -> CacheResult<BTreeMap<String, MetadataValue>> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| CacheError::InvalidInput("metadata must be a dict".to_string()))?;
    let mut metadata = BTreeMap::new();

    for (key, value) in dict.iter() {
        let key = key
            .cast::<PyString>()
            .map_err(|_| CacheError::InvalidInput("metadata keys must be strings".to_string()))?
            .to_str()
            .map_err(py_error)?
            .to_string();
        let value = extract_metadata_value(&value)?;
        metadata.insert(key, value);
    }

    Ok(metadata)
}

fn extract_metadata_value(value: &Bound<'_, PyAny>) -> CacheResult<MetadataValue> {
    if let Ok(value) = value.extract::<bool>() {
        return Ok(MetadataValue::Bool(value));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(MetadataValue::Int(value));
    }
    if let Ok(value) = value.extract::<f64>() {
        return Ok(MetadataValue::Float(value));
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(MetadataValue::String(value));
    }
    Err(CacheError::InvalidInput(
        "metadata values must be bool, int, float, or str".to_string(),
    ))
}

fn py_error(error: PyErr) -> CacheError {
    CacheError::Python(error.to_string())
}
