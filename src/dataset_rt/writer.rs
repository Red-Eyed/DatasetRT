use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use indicatif::MultiProgress;
use pyo3::exceptions::{PyKeyboardInterrupt, PySystemExit};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyByteArrayMethods, PyDict, PyIterator, PyList, PyString, PyTuple};

use crate::storage::{load_cache, CacheBuilder, FinishStats, PushSampleStats};
use crate::types::{
    CacheError, CacheResult, CompressionAlgo, MaxShardBytes, MetadataValue, NumWorkers,
    PrefetchSize, ShardCompression,
};
use crate::worker_pool;

#[path = "writer/progress.rs"]
mod progress;

#[path = "writer/pipeline.rs"]
mod pipeline;

#[path = "writer/profiler.rs"]
mod profiler;

use pipeline::write_source_list;
use profiler::{ProfileStage, WriterProfiler, WriterProfilerConfig};
use progress::WriteProgress;

#[derive(Clone)]
struct WriterConfig {
    max_shard_bytes: MaxShardBytes,
    prefetch_size: PrefetchSize,
    num_workers: NumWorkers,
    shard_compression: ShardCompression,
    show_progress: bool,
    validate_cache: bool,
    reuse_existing: bool,
    profiler: WriterProfilerConfig,
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

type CacheWriteRecord = (String, String, String);

pub fn write_cache(
    sources: Bound<'_, PyAny>,
    base_cache_dir: String,
    num_workers: usize,
    writer_config: Bound<'_, PyAny>,
    reuse_existing: bool,
) -> CacheResult<Vec<(String, String, String)>> {
    let num_workers = NumWorkers::new(num_workers)?;
    worker_pool::initialize(num_workers.as_usize())?;
    let config = extract_writer_config(&writer_config, num_workers, reuse_existing)?;
    let base_cache_dir = PathBuf::from(base_cache_dir);
    let profiler = WriterProfiler::new(&config.profiler);

    let result = if let Ok(source_list) = sources.cast::<PyList>() {
        write_source_list(source_list, &base_cache_dir, &config, profiler.clone())
    } else {
        write_one_source_result(sources, &base_cache_dir, 0, &config, None, profiler.clone())
            .map(|record| vec![record])
    };

    finish_with_profile(result, &profiler)
}

fn write_one_source_result(
    source: Bound<'_, PyAny>,
    base_cache_dir: &Path,
    source_index: usize,
    config: &WriterConfig,
    multi_progress: Option<&MultiProgress>,
    profiler: WriterProfiler,
) -> CacheResult<(String, String, String)> {
    let source_name = match extract_source_name(&source) {
        Ok(source_name) => source_name,
        // User interrupts are process-level control flow, not per-source data errors.
        Err(CacheError::KeyboardInterrupt(message)) => {
            return Err(CacheError::KeyboardInterrupt(message));
        }
        Err(CacheError::SystemExit(message)) => {
            return Err(CacheError::SystemExit(message));
        }
        Err(error) => return Ok(error_write_result(source_label(source_index), error)),
    };

    match write_named_source(
        source,
        base_cache_dir,
        source_index,
        config,
        &source_name,
        multi_progress,
        profiler,
    ) {
        Ok(path) => Ok(success_write_result(source_name, path)),
        // Preserve Ctrl-C/SystemExit so Python callers can stop the whole write immediately.
        Err(CacheError::KeyboardInterrupt(message)) => Err(CacheError::KeyboardInterrupt(message)),
        Err(CacheError::SystemExit(message)) => Err(CacheError::SystemExit(message)),
        Err(error) => Ok(error_write_result(source_name, error)),
    }
}

fn write_named_source(
    source: Bound<'_, PyAny>,
    base_cache_dir: &Path,
    source_index: usize,
    config: &WriterConfig,
    source_name: &str,
    multi_progress: Option<&MultiProgress>,
    profiler: WriterProfiler,
) -> CacheResult<PathBuf> {
    let cache_path = cache_path_for_source(base_cache_dir, source_name, source_index);
    if config.reuse_existing && cache_path.exists() {
        if config.validate_cache {
            load_cache(cache_path.clone(), true)?;
        }
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
    if let Err(error) = write_with_pipeline(
        iterator,
        builder,
        source_name,
        config,
        multi_progress,
        profiler.clone(),
    ) {
        cleanup_temp_cache(&temp_path)?;
        return Err(error);
    }
    let publish_started_at = Instant::now();
    if let Err(error) = publish_cache(&temp_path, &cache_path) {
        profiler.record(
            source_name,
            ProfileStage::Publish,
            publish_started_at.elapsed(),
        );
        cleanup_temp_cache(&temp_path)?;
        return Err(error);
    }
    profiler.record(
        source_name,
        ProfileStage::Publish,
        publish_started_at.elapsed(),
    );
    Ok(cache_path)
}

fn success_write_result(source_name: String, path: PathBuf) -> (String, String, String) {
    ("success".to_string(), source_name, path_to_string(&path))
}

fn error_write_result(source_name: String, error: CacheError) -> (String, String, String) {
    error_write_message(source_name, error.to_string())
}

fn error_write_message(source_name: String, message: String) -> (String, String, String) {
    ("error".to_string(), source_name, message)
}

fn finish_with_profile<T>(result: CacheResult<T>, profiler: &WriterProfiler) -> CacheResult<T> {
    let profile_result = profiler.finish();
    match result {
        Ok(value) => {
            profile_result?;
            Ok(value)
        }
        Err(error) => {
            // Preserve the writer failure or Python control-flow exception.
            // Profiling is diagnostic, so a failed best-effort flush must not
            // hide Ctrl-C, SystemExit, or the original cache-writing error.
            let _ = profile_result;
            Err(error)
        }
    }
}

fn source_label(source_index: usize) -> String {
    format!("source[{source_index}]")
}

fn write_with_pipeline(
    iterator: Bound<'_, PyIterator>,
    builder: CacheBuilder,
    source_name: &str,
    config: &WriterConfig,
    multi_progress: Option<&MultiProgress>,
    profiler: WriterProfiler,
) -> CacheResult<()> {
    let progress = WriteProgress::new(config.show_progress, source_name, multi_progress);
    let mut pipeline = SampleWritePipeline::new(
        builder,
        progress,
        source_name.to_string(),
        profiler.clone(),
        config.prefetch_size,
        config.num_workers,
    );
    ingest_python_samples(iterator, &mut pipeline, source_name, &profiler)?;
    pipeline.finish()
}

fn ingest_python_samples(
    mut iterator: Bound<'_, PyIterator>,
    pipeline: &mut SampleWritePipeline,
    source_name: &str,
    profiler: &WriterProfiler,
) -> CacheResult<()> {
    let mut sequence = 0_u64;
    loop {
        // Rust can spend a long time ingesting samples; poll Python signals explicitly.
        iterator.py().check_signals().map_err(py_error)?;
        let next_started_at = Instant::now();
        let Some(item) = iterator.next() else {
            profiler.record(
                source_name,
                ProfileStage::PythonNext,
                next_started_at.elapsed(),
            );
            return Ok(());
        };
        profiler.record(
            source_name,
            ProfileStage::PythonNext,
            next_started_at.elapsed(),
        );
        let current_sequence = sequence;
        let extract_started_at = Instant::now();
        let item = item
            .map_err(py_error)
            .and_then(|item| extract_cache_input(&item))?;
        profiler.record_bytes(
            source_name,
            ProfileStage::PythonExtract,
            extract_started_at.elapsed(),
            u64::try_from(item.data.len()).unwrap_or(u64::MAX),
        );
        let send_started_at = Instant::now();
        pipeline.submit(QueuedSample {
            sequence: current_sequence,
            input: item,
        })?;
        profiler.record(
            source_name,
            ProfileStage::IngressWait,
            send_started_at.elapsed(),
        );
        sequence = sequence.checked_add(1).ok_or_else(|| {
            CacheError::InvalidInput("sample sequence overflowed u64".to_string())
        })?;
    }
}

fn record_push_sample_stats(profiler: &WriterProfiler, source_name: &str, stats: &PushSampleStats) {
    profiler.record(
        source_name,
        ProfileStage::MetadataValidate,
        stats.metadata_validate,
    );
    profiler.record_bytes(
        source_name,
        ProfileStage::Compression,
        stats.compression,
        stats.uncompressed_bytes,
    );
    profiler.record_bytes(
        source_name,
        ProfileStage::RecordEncode,
        stats.record_encode,
        stats.stored_bytes,
    );
    profiler.record_bytes(
        source_name,
        ProfileStage::DiskWrite,
        stats.disk_write,
        stats.stored_bytes,
    );
}

fn record_finish_stats(profiler: &WriterProfiler, source_name: &str, stats: FinishStats) {
    profiler.record(source_name, ProfileStage::ShardFlush, stats.shard_flush);
    profiler.record(source_name, ProfileStage::FinishMetadata, stats.metadata);
    profiler.record(source_name, ProfileStage::FinishIndex, stats.index);
    profiler.record(source_name, ProfileStage::FinishManifest, stats.manifest);
}

struct SampleWritePipeline {
    builder: CacheBuilder,
    result_sender: Sender<CacheResult<SerializedSample>>,
    result_receiver: Receiver<CacheResult<SerializedSample>>,
    pending: BTreeMap<u64, SerializedSample>,
    next_sequence: u64,
    in_flight: usize,
    parallelism: usize,
    progress: WriteProgress,
    source_name: String,
    profiler: WriterProfiler,
}

impl SampleWritePipeline {
    /// Create a bounded per-write task window over the process-wide worker pool.
    fn new(
        builder: CacheBuilder,
        progress: WriteProgress,
        source_name: String,
        profiler: WriterProfiler,
        prefetch_size: PrefetchSize,
        num_workers: NumWorkers,
    ) -> Self {
        let parallelism = prefetch_size.as_usize().min(num_workers.as_usize());
        let (result_sender, result_receiver) = bounded(parallelism);
        Self {
            builder,
            result_sender,
            result_receiver,
            pending: BTreeMap::new(),
            next_sequence: 0,
            in_flight: 0,
            parallelism,
            progress,
            source_name,
            profiler,
        }
    }

    /// Submit one finite serialization job after freeing an operation-local credit.
    fn submit(&mut self, queued: QueuedSample) -> CacheResult<()> {
        if self.in_flight == self.parallelism {
            self.complete_one()?;
        }
        let result_sender = self.result_sender.clone();
        worker_pool::submit(result_sender, move || {
            Ok(SerializedSample {
                sequence: queued.sequence,
                input: queued.input,
            })
        })?;
        self.in_flight += 1;
        Ok(())
    }

    /// Drain all submitted jobs, commit in source order, and publish writer metadata.
    fn finish(mut self) -> CacheResult<()> {
        while self.in_flight > 0 {
            self.complete_one()?;
        }
        if !self.pending.is_empty() {
            return Err(CacheError::WorkerFailed);
        }
        let (_, stats) = self.builder.finish()?;
        record_finish_stats(&self.profiler, &self.source_name, stats);
        self.progress.finish();
        Ok(())
    }

    /// Receive one completed task and advance every now-contiguous sample.
    fn complete_one(&mut self) -> CacheResult<()> {
        let sample = receive_worker_result(&self.result_receiver)?;
        self.in_flight = self
            .in_flight
            .checked_sub(1)
            .ok_or(CacheError::WorkerFailed)?;
        if self.pending.insert(sample.sequence, sample).is_some() {
            return Err(CacheError::WorkerFailed);
        }
        commit_ready_samples(
            &mut self.builder,
            &mut self.pending,
            &mut self.next_sequence,
            &mut self.progress,
            &self.source_name,
            &self.profiler,
        )
    }
}

/// Wait for one finite writer job while keeping Python signal handling responsive.
fn receive_worker_result<T>(receiver: &Receiver<CacheResult<T>>) -> CacheResult<T> {
    loop {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(result) => return result,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                Python::attach(|py| py.check_signals()).map_err(py_error)?;
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                return Err(CacheError::WorkerFailed);
            }
        }
    }
}

fn commit_ready_samples(
    builder: &mut CacheBuilder,
    pending: &mut BTreeMap<u64, SerializedSample>,
    next_sequence: &mut u64,
    progress: &mut WriteProgress,
    source_name: &str,
    profiler: &WriterProfiler,
) -> CacheResult<()> {
    while let Some(sample) = pending.remove(next_sequence) {
        let byte_len = sample.input.data.len();
        let stats = builder.push_sample(sample.input.data, sample.input.metadata)?;
        record_push_sample_stats(profiler, source_name, &stats);
        progress.record_sample(byte_len);
        *next_sequence = next_sequence.checked_add(1).ok_or_else(|| {
            CacheError::InvalidInput("sample sequence overflowed u64".to_string())
        })?;
    }
    Ok(())
}

fn extract_writer_config(
    value: &Bound<'_, PyAny>,
    num_workers: NumWorkers,
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
    let shard_compression = value.getattr("shard_compression").map_err(py_error)?;
    let show_progress = value
        .getattr("show_progress")
        .map_err(py_error)?
        .extract::<bool>()
        .map_err(py_error)?;
    let validate_cache = value
        .getattr("validate_cache")
        .map_err(py_error)?
        .extract::<bool>()
        .map_err(py_error)?;

    Ok(WriterConfig {
        max_shard_bytes: MaxShardBytes::new(max_shard_bytes)?,
        prefetch_size: PrefetchSize::new(prefetch_size)?,
        num_workers,
        shard_compression: extract_shard_compression(&shard_compression)?,
        show_progress,
        validate_cache,
        reuse_existing,
        profiler: extract_writer_profiler_config(value)?,
    })
}

fn extract_writer_profiler_config(value: &Bound<'_, PyAny>) -> CacheResult<WriterProfilerConfig> {
    let profiler = match value.getattr("profiler") {
        Ok(profiler) => profiler,
        Err(_) => return Ok(WriterProfilerConfig::disabled()),
    };
    let enabled = profiler
        .getattr("enabled")
        .map_err(py_error)?
        .extract::<bool>()
        .map_err(py_error)?;
    let path = profiler
        .getattr("path")
        .map_err(py_error)?
        .str()
        .map_err(py_error)?
        .to_str()
        .map_err(py_error)?
        .to_string();
    Ok(WriterProfilerConfig {
        enabled,
        path: PathBuf::from(path),
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
    let message = error.to_string();
    // These exceptions should keep their Python control-flow semantics.
    if Python::attach(|py| error.is_instance_of::<PyKeyboardInterrupt>(py)) {
        return CacheError::KeyboardInterrupt(message);
    }
    if Python::attach(|py| error.is_instance_of::<PySystemExit>(py)) {
        return CacheError::SystemExit(message);
    }
    CacheError::Python(message)
}
