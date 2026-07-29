use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, SendTimeoutError, Sender};
use indicatif::MultiProgress;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList};

use super::progress::{SourceListProgress, WriteProgress};
use super::{
    cache_path_for_source, cleanup_temp_cache, error_write_message, error_write_result,
    extract_cache_input, extract_source_name, join_workers, prepare_cache_paths, publish_cache,
    py_error, source_label, success_write_result, CacheWriteRecord, WriterConfig, WriterInput,
};
use crate::storage::{load_cache, CacheBuilder};
use crate::types::{CacheError, CacheResult};

struct SourceWriteStart {
    source_index: usize,
    source_name: String,
    cache_path: PathBuf,
    temp_path: PathBuf,
}

struct QueuedPipelineMessage {
    sequence: u64,
    event: PipelineEvent,
}

struct SerializedPipelineMessage {
    sequence: u64,
    event: PipelineEvent,
}

enum PipelineEvent {
    SourceResult(usize, CacheWriteRecord),
    BeginSource(SourceWriteStart),
    Sample(WriterInput),
    EndSource,
    AbortSource(String),
}

pub(super) fn write_source_list(
    sources: &Bound<'_, PyList>,
    base_cache_dir: &Path,
    config: &WriterConfig,
) -> CacheResult<Vec<CacheWriteRecord>> {
    ensure_unique_source_paths(sources, base_cache_dir)?;

    let (input_sender, input_receiver) = bounded(config.prefetch_size.as_usize());
    let (output_sender, output_receiver) = bounded(config.prefetch_size.as_usize());
    let progress = SourceListProgress::new(config.show_progress, sources.len());
    let commit_handle =
        spawn_batch_commit_thread(output_receiver, config.clone(), progress, sources.len());
    let worker_handles =
        spawn_pipeline_workers(input_receiver, output_sender.clone(), config.num_threads);

    let ingestion_result = ingest_source_list(sources, base_cache_dir, config, &input_sender);
    drop(input_sender);
    drop(output_sender);

    let workers_result = join_workers(worker_handles);
    let commit_result = join_batch_commit(commit_handle);

    ingestion_result?;
    workers_result?;
    complete_write_results(commit_result?)
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

fn ingest_source_list(
    sources: &Bound<'_, PyList>,
    base_cache_dir: &Path,
    config: &WriterConfig,
    sender: &Sender<QueuedPipelineMessage>,
) -> CacheResult<()> {
    let mut ingress = PipelineIngress::new(sender);

    for (index, source) in sources.iter().enumerate() {
        source.py().check_signals().map_err(py_error)?;
        ingest_one_source(source, base_cache_dir, index, config, &mut ingress)?;
    }

    Ok(())
}

fn ingest_one_source(
    source: Bound<'_, PyAny>,
    base_cache_dir: &Path,
    source_index: usize,
    config: &WriterConfig,
    ingress: &mut PipelineIngress<'_>,
) -> CacheResult<()> {
    let source_name = match extract_source_name(&source) {
        Ok(source_name) => source_name,
        Err(CacheError::KeyboardInterrupt(message)) => {
            return Err(CacheError::KeyboardInterrupt(message));
        }
        Err(CacheError::SystemExit(message)) => {
            return Err(CacheError::SystemExit(message));
        }
        Err(error) => {
            return ingress.send_source_result(
                source_index,
                error_write_result(source_label(source_index), error),
            );
        }
    };

    let cache_path = cache_path_for_source(base_cache_dir, &source_name, source_index);
    if config.reuse_existing && cache_path.exists() {
        return send_existing_cache_result(ingress, source_index, source_name, cache_path);
    }

    let temp_path = super::temp_cache_path_for_source(base_cache_dir, &source_name, source_index);
    if let Err(error) = prepare_cache_paths(&cache_path, &temp_path) {
        return ingress.send_source_result(source_index, error_write_result(source_name, error));
    }

    let start = SourceWriteStart {
        source_index,
        source_name,
        cache_path,
        temp_path,
    };
    ingress.send_begin_source(start)?;

    let iterator = match PyIterator::from_object(&source).map_err(py_error) {
        Ok(iterator) => iterator,
        Err(CacheError::KeyboardInterrupt(message)) => {
            return Err(CacheError::KeyboardInterrupt(message));
        }
        Err(CacheError::SystemExit(message)) => {
            return Err(CacheError::SystemExit(message));
        }
        Err(error) => return ingress.send_abort_source(error.to_string()),
    };

    match ingest_source_samples(iterator, ingress) {
        Ok(()) => ingress.send_end_source(),
        Err(CacheError::KeyboardInterrupt(message)) => Err(CacheError::KeyboardInterrupt(message)),
        Err(CacheError::SystemExit(message)) => Err(CacheError::SystemExit(message)),
        Err(error) => ingress.send_abort_source(error.to_string()),
    }
}

fn send_existing_cache_result(
    ingress: &mut PipelineIngress<'_>,
    source_index: usize,
    source_name: String,
    cache_path: PathBuf,
) -> CacheResult<()> {
    let result = match load_cache(cache_path.clone()) {
        Ok(_) => success_write_result(source_name, cache_path),
        Err(error) => error_write_result(source_name, error),
    };
    ingress.send_source_result(source_index, result)
}

fn ingest_source_samples(
    mut iterator: Bound<'_, PyIterator>,
    ingress: &mut PipelineIngress<'_>,
) -> CacheResult<()> {
    loop {
        iterator.py().check_signals().map_err(py_error)?;
        let Some(item) = iterator.next() else {
            return Ok(());
        };
        let input = item
            .map_err(py_error)
            .and_then(|item| extract_cache_input(&item))?;
        ingress.send_sample(input)?;
    }
}

struct PipelineIngress<'a> {
    sender: &'a Sender<QueuedPipelineMessage>,
    next_sequence: u64,
}

impl<'a> PipelineIngress<'a> {
    fn new(sender: &'a Sender<QueuedPipelineMessage>) -> Self {
        Self {
            sender,
            next_sequence: 0,
        }
    }

    fn send_source_result(
        &mut self,
        source_index: usize,
        result: CacheWriteRecord,
    ) -> CacheResult<()> {
        self.send_event(PipelineEvent::SourceResult(source_index, result))
    }

    fn send_begin_source(&mut self, start: SourceWriteStart) -> CacheResult<()> {
        self.send_event(PipelineEvent::BeginSource(start))
    }

    fn send_sample(&mut self, input: WriterInput) -> CacheResult<()> {
        self.send_event(PipelineEvent::Sample(input))
    }

    fn send_end_source(&mut self) -> CacheResult<()> {
        self.send_event(PipelineEvent::EndSource)
    }

    fn send_abort_source(&mut self, message: String) -> CacheResult<()> {
        self.send_event(PipelineEvent::AbortSource(message))
    }

    fn send_event(&mut self, event: PipelineEvent) -> CacheResult<()> {
        let message = QueuedPipelineMessage {
            sequence: self.next_sequence,
            event,
        };
        self.next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
            CacheError::InvalidInput("pipeline sequence overflowed u64".to_string())
        })?;
        send_pipeline_message(self.sender, message)
    }
}

fn send_pipeline_message(
    sender: &Sender<QueuedPipelineMessage>,
    mut message: QueuedPipelineMessage,
) -> CacheResult<()> {
    loop {
        match sender.send_timeout(message, Duration::from_millis(100)) {
            Ok(()) => return Ok(()),
            Err(SendTimeoutError::Timeout(returned_message)) => {
                // The input queue is the memory boundary; keep Ctrl-C responsive at that boundary.
                Python::attach(|py| py.check_signals()).map_err(py_error)?;
                message = returned_message;
            }
            Err(SendTimeoutError::Disconnected(_)) => return Err(CacheError::WorkerFailed),
        }
    }
}

fn spawn_pipeline_workers(
    input_receiver: Receiver<QueuedPipelineMessage>,
    output_sender: Sender<SerializedPipelineMessage>,
    num_threads: crate::types::NumThreads,
) -> Vec<thread::JoinHandle<CacheResult<()>>> {
    (0..num_threads.as_usize())
        .map(|_| {
            let input_receiver = input_receiver.clone();
            let output_sender = output_sender.clone();
            thread::spawn(move || run_pipeline_worker(input_receiver, output_sender))
        })
        .collect()
}

fn run_pipeline_worker(
    input_receiver: Receiver<QueuedPipelineMessage>,
    output_sender: Sender<SerializedPipelineMessage>,
) -> CacheResult<()> {
    for queued in input_receiver {
        let serialized = SerializedPipelineMessage {
            sequence: queued.sequence,
            event: queued.event,
        };
        if output_sender.send(serialized).is_err() {
            return Ok(());
        }
    }
    Ok(())
}

fn spawn_batch_commit_thread(
    receiver: Receiver<SerializedPipelineMessage>,
    config: WriterConfig,
    progress: SourceListProgress,
    total_sources: usize,
) -> thread::JoinHandle<CacheResult<Vec<Option<CacheWriteRecord>>>> {
    thread::spawn(move || commit_pipeline_messages(receiver, config, progress, total_sources))
}

fn commit_pipeline_messages(
    receiver: Receiver<SerializedPipelineMessage>,
    config: WriterConfig,
    progress: SourceListProgress,
    total_sources: usize,
) -> CacheResult<Vec<Option<CacheWriteRecord>>> {
    let mut processor = PipelineCommitter::new(config, progress, total_sources);
    let mut pending = BTreeMap::new();
    let mut next_sequence = 0_u64;

    for message in receiver {
        pending.insert(message.sequence, message.event);
        processor.commit_ready_events(&mut pending, &mut next_sequence)?;
    }

    if !pending.is_empty() {
        return Err(CacheError::WorkerFailed);
    }

    processor.cleanup_unfinished_source();
    processor.finish();
    Ok(processor.results)
}

struct PipelineCommitter {
    config: WriterConfig,
    progress: SourceListProgress,
    results: Vec<Option<CacheWriteRecord>>,
    active: Option<ActiveSourceWrite>,
}

impl PipelineCommitter {
    fn new(config: WriterConfig, progress: SourceListProgress, total_sources: usize) -> Self {
        Self {
            config,
            progress,
            results: vec![None; total_sources],
            active: None,
        }
    }

    fn commit_ready_events(
        &mut self,
        pending: &mut BTreeMap<u64, PipelineEvent>,
        next_sequence: &mut u64,
    ) -> CacheResult<()> {
        while let Some(event) = pending.remove(next_sequence) {
            self.commit_event(event)?;
            *next_sequence = next_sequence.checked_add(1).ok_or_else(|| {
                CacheError::InvalidInput("pipeline sequence overflowed u64".to_string())
            })?;
        }
        Ok(())
    }

    fn commit_event(&mut self, event: PipelineEvent) -> CacheResult<()> {
        match event {
            PipelineEvent::SourceResult(source_index, result) => {
                self.record_result(source_index, result)
            }
            PipelineEvent::BeginSource(start) => self.begin_source(start),
            PipelineEvent::Sample(input) => self.push_sample(input),
            PipelineEvent::EndSource => self.end_source(),
            PipelineEvent::AbortSource(message) => self.abort_source(message),
        }
    }

    fn record_result(&mut self, source_index: usize, result: CacheWriteRecord) -> CacheResult<()> {
        let slot = self
            .results
            .get_mut(source_index)
            .ok_or(CacheError::WorkerFailed)?;
        if slot.is_some() {
            return Err(CacheError::WorkerFailed);
        }
        *slot = Some(result);
        self.progress.record_source();
        Ok(())
    }

    fn begin_source(&mut self, start: SourceWriteStart) -> CacheResult<()> {
        if self.active.is_some() {
            return Err(CacheError::WorkerFailed);
        }
        self.active = Some(ActiveSourceWrite::new(
            start,
            &self.config,
            self.progress.multi_progress(),
        ));
        Ok(())
    }

    fn push_sample(&mut self, input: WriterInput) -> CacheResult<()> {
        let active = self.active.as_mut().ok_or(CacheError::WorkerFailed)?;
        active.push_sample(input);
        Ok(())
    }

    fn end_source(&mut self) -> CacheResult<()> {
        let active = self.active.take().ok_or(CacheError::WorkerFailed)?;
        let source_index = active.source_index();
        let result = active.finish();
        self.record_result(source_index, result)
    }

    fn abort_source(&mut self, message: String) -> CacheResult<()> {
        let mut active = self.active.take().ok_or(CacheError::WorkerFailed)?;
        active.mark_failed(CacheError::InvalidInput(message));
        let source_index = active.source_index();
        let result = active.error_result();
        self.record_result(source_index, result)
    }

    fn cleanup_unfinished_source(&mut self) {
        if let Some(mut active) = self.active.take() {
            active.mark_failed(CacheError::WorkerFailed);
        }
    }

    fn finish(&self) {
        self.progress.finish();
    }
}

struct ActiveSourceWrite {
    start: SourceWriteStart,
    builder: Option<CacheBuilder>,
    progress: WriteProgress,
    failure: Option<String>,
}

impl ActiveSourceWrite {
    fn new(
        start: SourceWriteStart,
        config: &WriterConfig,
        multi_progress: Option<&MultiProgress>,
    ) -> Self {
        let progress = WriteProgress::new(config.show_progress, &start.source_name, multi_progress);
        let builder = CacheBuilder::create(
            start.temp_path.clone(),
            start.source_name.clone(),
            config.max_shard_bytes.as_u64(),
            config.shard_compression.clone(),
        );
        match builder {
            Ok(builder) => Self {
                start,
                builder: Some(builder),
                progress,
                failure: None,
            },
            Err(error) => {
                let mut active = Self {
                    start,
                    builder: None,
                    progress,
                    failure: Some(error.to_string()),
                };
                active.cleanup_temp_cache();
                active
            }
        }
    }

    fn source_index(&self) -> usize {
        self.start.source_index
    }

    fn push_sample(&mut self, input: WriterInput) {
        if self.failure.is_some() {
            return;
        }
        let Some(builder) = self.builder.as_mut() else {
            self.mark_failed(CacheError::WorkerFailed);
            return;
        };

        let byte_len = input.data.len();
        if let Err(error) = builder.push_sample(input.data, input.metadata) {
            self.mark_failed(error);
            return;
        }
        self.progress.record_sample(byte_len);
    }

    fn finish(mut self) -> CacheWriteRecord {
        if self.failure.is_some() {
            return self.error_result();
        }

        let Some(builder) = self.builder.take() else {
            self.mark_failed(CacheError::WorkerFailed);
            return self.error_result();
        };

        if let Err(error) = builder.finish() {
            self.mark_failed(error);
            return self.error_result();
        }
        self.progress.finish();

        if let Err(error) = publish_cache(&self.start.temp_path, &self.start.cache_path) {
            self.mark_failed(error);
            return self.error_result();
        }

        success_write_result(self.start.source_name, self.start.cache_path)
    }

    fn error_result(mut self) -> CacheWriteRecord {
        self.builder.take();
        self.cleanup_temp_cache();
        let message = self
            .failure
            .take()
            .unwrap_or_else(|| CacheError::WorkerFailed.to_string());
        error_write_message(self.start.source_name, message)
    }

    fn mark_failed(&mut self, error: CacheError) {
        if self.failure.is_none() {
            self.failure = Some(error.to_string());
        }
        self.builder.take();
        self.cleanup_temp_cache();
    }

    fn cleanup_temp_cache(&mut self) {
        if let Err(error) = cleanup_temp_cache(&self.start.temp_path) {
            let message = match self.failure.take() {
                Some(message) => format!("{message}; cleanup failed: {error}"),
                None => error.to_string(),
            };
            self.failure = Some(message);
        }
    }
}

fn join_batch_commit(
    handle: thread::JoinHandle<CacheResult<Vec<Option<CacheWriteRecord>>>>,
) -> CacheResult<Vec<Option<CacheWriteRecord>>> {
    handle.join().map_err(|_| CacheError::WorkerFailed)?
}

fn complete_write_results(
    results: Vec<Option<CacheWriteRecord>>,
) -> CacheResult<Vec<CacheWriteRecord>> {
    results.into_iter().map(require_write_result).collect()
}

fn require_write_result(result: Option<CacheWriteRecord>) -> CacheResult<CacheWriteRecord> {
    result.ok_or(CacheError::WorkerFailed)
}
