use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::types::{CacheError, CacheResult};

/// Thread-safe writer-stage profiler.
///
/// The writer records only aggregate timing data here. The profiler writes one
/// JSON summary at the top-level writer boundary so cache-writing correctness
/// does not depend on per-event profiler I/O.
#[derive(Clone)]
pub(super) struct WriterProfiler {
    inner: Arc<ProfilerInner>,
}

/// User-facing profiler settings extracted from Python's `WriterConfig`.
#[derive(Clone)]
pub(super) struct WriterProfilerConfig {
    pub(super) enabled: bool,
    pub(super) path: PathBuf,
}

/// Coarse timing buckets for answering where writer time is spent.
#[derive(Clone, Copy)]
pub(super) enum ProfileStage {
    PythonNext,
    PythonExtract,
    IngressWait,
    MetadataValidate,
    Compression,
    RecordEncode,
    DiskWrite,
    ShardFlush,
    FinishMetadata,
    FinishIndex,
    FinishManifest,
    Publish,
}

struct ProfilerInner {
    path: ProfilerPath,
    started_at: Instant,
    sources: Mutex<BTreeMap<String, SourceStats>>,
}

enum ProfilerPath {
    Disabled,
    Enabled(PathBuf),
}

#[derive(Default, Serialize)]
struct SourceStats {
    stages: BTreeMap<&'static str, StageStats>,
}

#[derive(Default, Serialize)]
struct StageStats {
    calls: u64,
    total_ns: u128,
    bytes: u64,
}

#[derive(Serialize)]
struct ProfileSummary<'a> {
    format_version: u32,
    elapsed_ns: u128,
    sources: Vec<SourceSummary<'a>>,
}

#[derive(Serialize)]
struct SourceSummary<'a> {
    source_name: &'a str,
    stages: Vec<StageSummary>,
}

#[derive(Serialize)]
struct StageSummary {
    name: &'static str,
    calls: u64,
    total_ns: u128,
    bytes: u64,
}

impl WriterProfiler {
    /// Create either an enabled aggregate profiler or a zero-cost disabled profiler.
    pub(super) fn new(config: &WriterProfilerConfig) -> Self {
        let path = if config.enabled {
            ProfilerPath::Enabled(config.path.clone())
        } else {
            ProfilerPath::Disabled
        };
        Self {
            inner: Arc::new(ProfilerInner {
                path,
                started_at: Instant::now(),
                sources: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    /// Accumulate one duration-only stage observation for a source.
    pub(super) fn record(&self, source_name: &str, stage: ProfileStage, duration: Duration) {
        self.record_bytes(source_name, stage, duration, 0);
    }

    /// Accumulate one stage observation and the payload bytes it handled.
    pub(super) fn record_bytes(
        &self,
        source_name: &str,
        stage: ProfileStage,
        duration: Duration,
        bytes: u64,
    ) {
        if !self.is_enabled() {
            return;
        }

        let mut sources = match self.inner.sources.lock() {
            Ok(sources) => sources,
            Err(_) => return,
        };
        let source = sources.entry(source_name.to_string()).or_default();
        let stats = source.stages.entry(stage.as_str()).or_default();
        stats.calls = stats.calls.saturating_add(1);
        stats.total_ns = stats.total_ns.saturating_add(duration.as_nanos());
        stats.bytes = stats.bytes.saturating_add(bytes);
    }

    /// Write the final JSON summary.
    ///
    /// Callers decide whether this error can replace the main writer result.
    /// In particular, Ctrl-C should preserve the original Python control-flow
    /// exception after making a best-effort profile flush.
    pub(super) fn finish(&self) -> CacheResult<()> {
        let ProfilerPath::Enabled(path) = &self.inner.path else {
            return Ok(());
        };

        let sources = self
            .inner
            .sources
            .lock()
            .map_err(|_| CacheError::WorkerFailed)?;
        let summary = profile_summary(self.inner.started_at.elapsed(), &sources);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        serde_json::to_writer_pretty(BufWriter::new(file), &summary)?;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        matches!(self.inner.path, ProfilerPath::Enabled(_))
    }
}

impl WriterProfilerConfig {
    pub(super) fn disabled() -> Self {
        Self {
            enabled: false,
            path: PathBuf::from("dataset_rt_profile.json"),
        }
    }
}

impl ProfileStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::PythonNext => "python_next",
            Self::PythonExtract => "python_extract",
            Self::IngressWait => "ingress_wait",
            Self::MetadataValidate => "metadata_validate",
            Self::Compression => "compression",
            Self::RecordEncode => "record_encode",
            Self::DiskWrite => "disk_write",
            Self::ShardFlush => "shard_flush",
            Self::FinishMetadata => "finish_metadata",
            Self::FinishIndex => "finish_index",
            Self::FinishManifest => "finish_manifest",
            Self::Publish => "publish",
        }
    }
}

fn profile_summary<'a>(
    elapsed: Duration,
    sources: &'a BTreeMap<String, SourceStats>,
) -> ProfileSummary<'a> {
    ProfileSummary {
        format_version: 1,
        elapsed_ns: elapsed.as_nanos(),
        sources: sources
            .iter()
            .map(|(source_name, stats)| source_summary(source_name, stats))
            .collect(),
    }
}

fn source_summary<'a>(source_name: &'a str, stats: &SourceStats) -> SourceSummary<'a> {
    SourceSummary {
        source_name,
        stages: stats
            .stages
            .iter()
            .map(|(name, stage)| StageSummary {
                name,
                calls: stage.calls,
                total_ns: stage.total_ns,
                bytes: stage.bytes,
            })
            .collect(),
    }
}
