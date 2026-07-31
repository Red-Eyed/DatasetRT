use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crossbeam_channel::{bounded, Receiver, Sender};
use indicatif::MultiProgress;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList};

use super::progress::{SourceListProgress, WriteProgress};
use super::{
    cache_path_for_source, cleanup_temp_cache, error_write_message, error_write_result,
    extract_cache_input, extract_source_name, prepare_cache_paths, publish_cache, py_error,
    receive_worker_result, record_finish_stats, record_push_sample_stats, source_label,
    success_write_result, CacheWriteRecord, ProfileStage, WriterConfig, WriterInput,
    WriterProfiler,
};
use crate::storage::{load_cache, CacheBuilder};
use crate::types::{CacheError, CacheResult};
use crate::worker_pool;

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
    profiler: WriterProfiler,
) -> CacheResult<Vec<CacheWriteRecord>> {
    ensure_unique_source_paths(sources, base_cache_dir)?;

    let progress = SourceListProgress::new(config.show_progress, sources.len());
    let committer =
        PipelineCommitter::new(config.clone(), progress, sources.len(), profiler.clone());
    let mut pipeline =
        SourceListWritePipeline::new(committer, config.prefetch_size, config.num_workers);
    if let Err(error) =
        ingest_source_list(sources, base_cache_dir, config, &mut pipeline, &profiler)
    {
        pipeline.finish_incomplete();
        return Err(error);
    }
    complete_write_results(pipeline.finish()?)
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
    pipeline: &mut SourceListWritePipeline,
    profiler: &WriterProfiler,
) -> CacheResult<()> {
    let mut ingress = PipelineIngress::new(pipeline);

    for (index, source) in sources.iter().enumerate() {
        source.py().check_signals().map_err(py_error)?;
        ingest_one_source(
            source,
            base_cache_dir,
            index,
            config,
            &mut ingress,
            profiler,
        )?;
    }

    Ok(())
}

fn ingest_one_source(
    source: Bound<'_, PyAny>,
    base_cache_dir: &Path,
    source_index: usize,
    config: &WriterConfig,
    ingress: &mut PipelineIngress<'_>,
    profiler: &WriterProfiler,
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
        return send_existing_cache_result(ingress, config, source_index, source_name, cache_path);
    }

    let temp_path = super::temp_cache_path_for_source(base_cache_dir, &source_name, source_index);
    if let Err(error) = prepare_cache_paths(&cache_path, &temp_path) {
        return ingress.send_source_result(source_index, error_write_result(source_name, error));
    }

    let start = SourceWriteStart {
        source_index,
        source_name: source_name.clone(),
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

    match ingest_source_samples(iterator, ingress, &source_name, profiler) {
        Ok(()) => ingress.send_end_source(),
        Err(CacheError::KeyboardInterrupt(message)) => Err(CacheError::KeyboardInterrupt(message)),
        Err(CacheError::SystemExit(message)) => Err(CacheError::SystemExit(message)),
        Err(error) => ingress.send_abort_source(error.to_string()),
    }
}

fn send_existing_cache_result(
    ingress: &mut PipelineIngress<'_>,
    config: &WriterConfig,
    source_index: usize,
    source_name: String,
    cache_path: PathBuf,
) -> CacheResult<()> {
    let result = existing_cache_result(config, source_name, cache_path);
    ingress.send_source_result(source_index, result)
}

fn existing_cache_result(
    config: &WriterConfig,
    source_name: String,
    cache_path: PathBuf,
) -> CacheWriteRecord {
    if !config.validate_cache {
        return success_write_result(source_name, cache_path);
    }

    match load_cache(cache_path.clone(), true) {
        Ok(_) => success_write_result(source_name, cache_path),
        Err(error) => error_write_result(source_name, error),
    }
}

fn ingest_source_samples(
    mut iterator: Bound<'_, PyIterator>,
    ingress: &mut PipelineIngress<'_>,
    source_name: &str,
    profiler: &WriterProfiler,
) -> CacheResult<()> {
    loop {
        iterator.py().check_signals().map_err(py_error)?;
        let next_started_at = std::time::Instant::now();
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
        let extract_started_at = std::time::Instant::now();
        let input = item
            .map_err(py_error)
            .and_then(|item| extract_cache_input(&item))?;
        profiler.record_bytes(
            source_name,
            ProfileStage::PythonExtract,
            extract_started_at.elapsed(),
            u64::try_from(input.data.len()).unwrap_or(u64::MAX),
        );
        let send_started_at = std::time::Instant::now();
        ingress.send_sample(input)?;
        profiler.record(
            source_name,
            ProfileStage::IngressWait,
            send_started_at.elapsed(),
        );
    }
}

struct PipelineIngress<'a> {
    pipeline: &'a mut SourceListWritePipeline,
    next_sequence: u64,
}

impl<'a> PipelineIngress<'a> {
    fn new(pipeline: &'a mut SourceListWritePipeline) -> Self {
        Self {
            pipeline,
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
        let next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
            CacheError::InvalidInput("pipeline sequence overflowed u64".to_string())
        })?;
        self.pipeline.submit(message)?;
        self.next_sequence = next_sequence;
        Ok(())
    }
}

struct SourceListWritePipeline {
    result_sender: Sender<CacheResult<SerializedPipelineMessage>>,
    result_receiver: Receiver<CacheResult<SerializedPipelineMessage>>,
    committer: Option<PipelineCommitter>,
    pending: BTreeMap<u64, PipelineEvent>,
    next_sequence: u64,
    in_flight: usize,
    parallelism: usize,
}

impl SourceListWritePipeline {
    /// Create bounded operation state for finite source-list writer jobs.
    fn new(
        committer: PipelineCommitter,
        prefetch_size: crate::types::PrefetchSize,
        num_workers: crate::types::NumWorkers,
    ) -> Self {
        let parallelism = prefetch_size.as_usize().min(num_workers.as_usize());
        let (result_sender, result_receiver) = bounded(parallelism);
        Self {
            result_sender,
            result_receiver,
            committer: Some(committer),
            pending: BTreeMap::new(),
            next_sequence: 0,
            in_flight: 0,
            parallelism,
        }
    }

    /// Submit one finite pipeline event after freeing an operation-local credit.
    fn submit(&mut self, queued: QueuedPipelineMessage) -> CacheResult<()> {
        if self.in_flight == self.parallelism {
            self.complete_one()?;
        }
        let result_sender = self.result_sender.clone();
        worker_pool::submit(result_sender, move || {
            Ok(SerializedPipelineMessage {
                sequence: queued.sequence,
                event: queued.event,
            })
        })?;
        self.in_flight += 1;
        Ok(())
    }

    /// Drain all jobs and return one ordered result slot per requested source.
    fn finish(mut self) -> CacheResult<Vec<Option<CacheWriteRecord>>> {
        while self.in_flight > 0 {
            self.complete_one()?;
        }
        if !self.pending.is_empty() {
            return Err(CacheError::WorkerFailed);
        }

        let processor = self.committer.take().ok_or(CacheError::WorkerFailed)?;
        processor.finish();
        Ok(processor.results)
    }

    /// Drain accepted jobs, preserve completed sources, and clean an interrupted source.
    fn finish_incomplete(mut self) {
        while self.in_flight > 0 {
            if self.complete_one().is_err() {
                break;
            }
        }
        if let Some(mut processor) = self.committer.take() {
            processor.cleanup_unfinished_source();
            processor.finish();
        }
    }

    /// Receive one completed task and commit every now-contiguous event.
    fn complete_one(&mut self) -> CacheResult<()> {
        let message = receive_worker_result(&self.result_receiver)?;
        self.in_flight = self
            .in_flight
            .checked_sub(1)
            .ok_or(CacheError::WorkerFailed)?;
        if self
            .pending
            .insert(message.sequence, message.event)
            .is_some()
        {
            return Err(CacheError::WorkerFailed);
        }
        self.committer
            .as_mut()
            .ok_or(CacheError::WorkerFailed)?
            .commit_ready_events(&mut self.pending, &mut self.next_sequence)
    }
}

impl Drop for SourceListWritePipeline {
    fn drop(&mut self) {
        if let Some(processor) = self.committer.as_mut() {
            processor.cleanup_unfinished_source();
            processor.finish();
        }
    }
}

struct PipelineCommitter {
    config: WriterConfig,
    progress: SourceListProgress,
    profiler: WriterProfiler,
    results: Vec<Option<CacheWriteRecord>>,
    active: Option<ActiveSourceWrite>,
}

impl PipelineCommitter {
    fn new(
        config: WriterConfig,
        progress: SourceListProgress,
        total_sources: usize,
        profiler: WriterProfiler,
    ) -> Self {
        Self {
            config,
            progress,
            profiler,
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
            self.profiler.clone(),
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
    profiler: WriterProfiler,
    failure: Option<String>,
}

impl ActiveSourceWrite {
    fn new(
        start: SourceWriteStart,
        config: &WriterConfig,
        multi_progress: Option<&MultiProgress>,
        profiler: WriterProfiler,
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
                profiler,
                failure: None,
            },
            Err(error) => {
                let mut active = Self {
                    start,
                    builder: None,
                    progress,
                    profiler,
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
        let stats = match builder.push_sample(input.data, input.metadata) {
            Ok(stats) => stats,
            Err(error) => {
                self.mark_failed(error);
                return;
            }
        };
        record_push_sample_stats(&self.profiler, &self.start.source_name, &stats);
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

        let finish_stats = match builder.finish() {
            Ok((_, stats)) => stats,
            Err(error) => {
                self.mark_failed(error);
                return self.error_result();
            }
        };
        record_finish_stats(&self.profiler, &self.start.source_name, finish_stats);
        self.progress.finish();

        let publish_started_at = std::time::Instant::now();
        if let Err(error) = publish_cache(&self.start.temp_path, &self.start.cache_path) {
            self.profiler.record(
                &self.start.source_name,
                ProfileStage::Publish,
                publish_started_at.elapsed(),
            );
            self.mark_failed(error);
            return self.error_result();
        }
        self.profiler.record(
            &self.start.source_name,
            ProfileStage::Publish,
            publish_started_at.elapsed(),
        );

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

fn complete_write_results(
    results: Vec<Option<CacheWriteRecord>>,
) -> CacheResult<Vec<CacheWriteRecord>> {
    results.into_iter().map(require_write_result).collect()
}

fn require_write_result(result: Option<CacheWriteRecord>) -> CacheResult<CacheWriteRecord> {
    result.ok_or(CacheError::WorkerFailed)
}
