use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyByteArrayMethods, PyDict, PyIterator, PyList, PyString, PyTuple};

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
    show_progress: bool,
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

struct WriteProgress {
    bar: Option<ProgressBar>,
    started_at: Instant,
    items: u64,
    bytes: u64,
}

enum CommitMessage {
    Sample(SerializedSample),
    Abort(String),
}

pub fn write_cache(
    sources: Bound<'_, PyAny>,
    base_cache_dir: String,
    writer_config: Bound<'_, PyAny>,
    reuse_existing: bool,
) -> CacheResult<Vec<(String, String, String)>> {
    let config = extract_writer_config(&writer_config, reuse_existing)?;
    let base_cache_dir = PathBuf::from(base_cache_dir);

    if let Ok(source_list) = sources.cast::<PyList>() {
        return write_source_list(source_list, &base_cache_dir, &config);
    }

    Ok(vec![write_one_source_result(
        sources,
        &base_cache_dir,
        0,
        &config,
    )])
}

fn write_source_list(
    sources: &Bound<'_, PyList>,
    base_cache_dir: &Path,
    config: &WriterConfig,
) -> CacheResult<Vec<(String, String, String)>> {
    ensure_unique_source_paths(sources, base_cache_dir)?;

    Ok(sources
        .iter()
        .enumerate()
        .map(|(index, source)| write_one_source_result(source, base_cache_dir, index, config))
        .collect())
}

fn ensure_unique_source_paths(
    sources: &Bound<'_, PyList>,
    base_cache_dir: &Path,
) -> CacheResult<()> {
    let mut seen_paths = BTreeSet::new();

    for (index, source) in sources.iter().enumerate() {
        let source_name = match extract_source_name(&source) {
            Ok(source_name) => source_name,
            Err(_) => continue,
        };
        let cache_path = cache_path_for_source(base_cache_dir, &source_name, index);
        if !seen_paths.insert(cache_path.clone()) {
            return Err(CacheError::InvalidInput(format!(
                "duplicate generated cache path: {}",
                cache_path.display()
            )));
        }
    }

    Ok(())
}

fn write_one_source_result(
    source: Bound<'_, PyAny>,
    base_cache_dir: &Path,
    source_index: usize,
    config: &WriterConfig,
) -> (String, String, String) {
    let source_name = match extract_source_name(&source) {
        Ok(source_name) => source_name,
        Err(error) => return error_write_result(source_label(source_index), error),
    };

    match write_named_source(source, base_cache_dir, source_index, config, &source_name) {
        Ok(path) => success_write_result(source_name, path),
        Err(error) => error_write_result(source_name, error),
    }
}

fn write_named_source(
    source: Bound<'_, PyAny>,
    base_cache_dir: &Path,
    source_index: usize,
    config: &WriterConfig,
    source_name: &str,
) -> CacheResult<PathBuf> {
    let cache_path = cache_path_for_source(base_cache_dir, source_name, source_index);
    if config.reuse_existing && cache_path.exists() {
        load_cache(cache_path.clone())?;
        return Ok(cache_path);
    }

    let temp_path = temp_cache_path_for_source(base_cache_dir, source_name, source_index);
    prepare_cache_paths(&cache_path, &temp_path)?;
    let builder = match CacheBuilder::create(
        temp_path.clone(),
        source_name.to_string(),
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
    if let Err(error) = write_with_pipeline(iterator, builder, source_name, config) {
        cleanup_temp_cache(&temp_path)?;
        return Err(error);
    }
    if let Err(error) = publish_cache(&temp_path, &cache_path) {
        cleanup_temp_cache(&temp_path)?;
        return Err(error);
    }
    Ok(cache_path)
}

fn success_write_result(source_name: String, path: PathBuf) -> (String, String, String) {
    ("success".to_string(), source_name, path_to_string(&path))
}

fn error_write_result(source_name: String, error: CacheError) -> (String, String, String) {
    ("error".to_string(), source_name, error.to_string())
}

fn source_label(source_index: usize) -> String {
    format!("source[{source_index}]")
}

fn write_with_pipeline(
    iterator: Bound<'_, PyIterator>,
    builder: CacheBuilder,
    source_name: &str,
    config: &WriterConfig,
) -> CacheResult<()> {
    let (input_sender, input_receiver) = bounded(config.prefetch_size.as_usize());
    let (commit_sender, commit_receiver) = bounded(config.prefetch_size.as_usize());
    let progress = WriteProgress::new(config.show_progress, source_name);
    let commit_handle = spawn_commit_thread(builder, commit_receiver, progress);
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
    progress: WriteProgress,
) -> thread::JoinHandle<CacheResult<()>> {
    thread::spawn(move || commit_serialized_samples(builder, receiver, progress))
}

fn commit_serialized_samples(
    mut builder: CacheBuilder,
    receiver: Receiver<CommitMessage>,
    mut progress: WriteProgress,
) -> CacheResult<()> {
    let mut pending = BTreeMap::new();
    let mut next_sequence = 0_u64;

    for message in receiver {
        match message {
            CommitMessage::Sample(sample) => {
                pending.insert(sample.sequence, sample);
                commit_ready_samples(
                    &mut builder,
                    &mut pending,
                    &mut next_sequence,
                    &mut progress,
                )?;
            }
            CommitMessage::Abort(message) => return Err(CacheError::InvalidInput(message)),
        }
    }

    if !pending.is_empty() {
        return Err(CacheError::WorkerFailed);
    }

    builder.finish()?;
    progress.finish();
    Ok(())
}

fn commit_ready_samples(
    builder: &mut CacheBuilder,
    pending: &mut BTreeMap<u64, SerializedSample>,
    next_sequence: &mut u64,
    progress: &mut WriteProgress,
) -> CacheResult<()> {
    while let Some(sample) = pending.remove(next_sequence) {
        let byte_len = sample.input.data.len();
        builder.push_sample(sample.input.data, sample.input.metadata)?;
        progress.record_sample(byte_len);
        *next_sequence = next_sequence.checked_add(1).ok_or_else(|| {
            CacheError::InvalidInput("sample sequence overflowed u64".to_string())
        })?;
    }
    Ok(())
}

impl WriteProgress {
    fn new(enabled: bool, source_name: &str) -> Self {
        let bar = enabled.then(|| create_progress_bar(source_name));
        Self {
            bar,
            started_at: Instant::now(),
            items: 0,
            bytes: 0,
        }
    }

    fn record_sample(&mut self, byte_len: usize) {
        self.items = self.items.saturating_add(1);
        let byte_len = u64::try_from(byte_len).unwrap_or(u64::MAX);
        self.bytes = self.bytes.saturating_add(byte_len);

        if let Some(bar) = &self.bar {
            bar.inc(1);
            bar.set_message(self.message());
        }
    }

    fn finish(&self) {
        if let Some(bar) = &self.bar {
            bar.finish_with_message(self.message());
        }
    }

    fn message(&self) -> String {
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64().max(0.001);
        let items_per_second = self.items as f64 / elapsed_seconds;
        let megabytes_per_second = self.bytes as f64 / 1_000_000.0 / elapsed_seconds;
        format!(
            "{} items | {:.1} items/s | {:.1} MB/s",
            self.items, items_per_second, megabytes_per_second
        )
    }
}

fn create_progress_bar(source_name: &str) -> ProgressBar {
    let bar = ProgressBar::new_spinner();
    bar.set_prefix(format!("writing {source_name}"));
    bar.enable_steady_tick(Duration::from_millis(100));
    // Progress rendering is observability only; an invalid style must never fail cache writes.
    if let Ok(style) = ProgressStyle::with_template("{spinner:.green} {prefix} {msg}") {
        bar.set_style(style);
    }
    bar
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

fn extract_writer_config(
    value: &Bound<'_, PyAny>,
    reuse_existing: bool,
) -> CacheResult<WriterConfig> {
    let max_shard_bytes = value
        .getattr("max_shard_bytes")
        .map_err(py_error)?
        .extract::<u64>()
        .map_err(py_error)?;
    let prefetch_size = value
        .getattr("prefetch_size")
        .map_err(py_error)?
        .extract::<usize>()
        .map_err(py_error)?;
    let num_threads = value
        .getattr("num_threads")
        .map_err(py_error)?
        .extract::<usize>()
        .map_err(py_error)?;
    let shard_compression = value.getattr("shard_compression").map_err(py_error)?;
    let show_progress = value
        .getattr("show_progress")
        .map_err(py_error)?
        .extract::<bool>()
        .map_err(py_error)?;

    Ok(WriterConfig {
        max_shard_bytes: MaxShardBytes::new(max_shard_bytes)?,
        prefetch_size: PrefetchSize::new(prefetch_size)?,
        num_threads: NumThreads::new(num_threads)?,
        shard_compression: extract_shard_compression(&shard_compression)?,
        show_progress,
        reuse_existing,
    })
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

fn cache_path_for_source(
    base_cache_dir: &Path,
    source_name: &str,
    _source_index: usize,
) -> PathBuf {
    base_cache_dir.join(source_name)
}

fn temp_cache_path_for_source(
    base_cache_dir: &Path,
    source_name: &str,
    _source_index: usize,
) -> PathBuf {
    base_cache_dir.join("tmp").join(source_name)
}

fn prepare_cache_paths(cache_path: &Path, temp_path: &Path) -> CacheResult<()> {
    if cache_path.exists() {
        return Err(CacheError::InvalidInput(format!(
            "cache path already exists: {}",
            cache_path.display()
        )));
    }
    if temp_path.exists() {
        fs::remove_dir_all(temp_path)?;
    }
    Ok(())
}

fn cleanup_temp_cache(temp_path: &Path) -> CacheResult<()> {
    if temp_path.exists() {
        fs::remove_dir_all(temp_path)?;
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
