//! Resolve embedded or Arrow metadata without exposing version rules to storage orchestration.

use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatch;

use super::record::{DecodedRecord, RecordLayout};
use super::{arrow_type, read_metadata_row, LoadedCache};
use crate::arrow_metadata::mapped::MappedMetadataIndex;
use crate::types::{CacheError, CacheResult, CacheSample, MetadataField, MetadataValue};

/// Keep legacy full-row validation opt-in and Arrow-only metadata lazy at construction.
pub(super) fn prepare_rows(
    layout: RecordLayout,
    path: &Path,
    schema: &[MetadataField],
    validate: bool,
) -> CacheResult<Option<Vec<Vec<MetadataValue>>>> {
    std::fs::metadata(path).map_err(|error| {
        CacheError::from(error).context(format!("inspect metadata file '{}'", path.display()))
    })?;
    if !validate {
        return Ok(None);
    }
    match layout {
        RecordLayout::LegacyEnvelope | RecordLayout::RecordEnvelope => {
            Ok(Some(super::read_metadata_file(path, schema)?))
        }
        RecordLayout::PayloadOnly => Ok(None),
    }
}

/// Full validation of Arrow-only metadata stays streamed; embedded rows were prepared above.
pub(super) fn validate_prepared(cache: &LoadedCache, validate: bool) -> CacheResult<()> {
    if !validate {
        return Ok(());
    }
    match cache.record_layout {
        RecordLayout::LegacyEnvelope | RecordLayout::RecordEnvelope => Ok(()),
        RecordLayout::PayloadOnly => validate_file(cache),
    }
}

/// Resolve typed metadata provenance without consulting manifest versions on the read path.
pub(super) fn materialize(
    decoded: DecodedRecord,
    cache: &LoadedCache,
    sample_index: usize,
    cached: &mut Option<CachedMetadataBatch>,
) -> CacheResult<CacheSample> {
    match decoded {
        DecodedRecord::ArrowPayload(data) => Ok(CacheSample {
            data,
            metadata: read_row(cache, sample_index, cached)?,
        }),
        DecodedRecord::Embedded(sample) => {
            if cache.validate_embedded_metadata {
                cache.with_metadata_rows(|rows| {
                    let metadata = rows.get(sample_index).ok_or_else(|| {
                        CacheError::InvalidCache(format!(
                            "metadata row {sample_index} is out of range"
                        ))
                    })?;
                    validate_redundant_metadata(metadata, &sample.metadata, sample_index)
                })?;
            }
            Ok(sample)
        }
    }
}

/// A worker retains one decoded batch total, even when alternating among thousands of caches.
pub(super) struct CachedMetadataBatch {
    pub(super) source: Arc<MappedMetadataIndex>,
    index: usize,
    batch: RecordBatch,
}

/// Initialize only batch descriptors and a mapping, with no shared mutable decoder.
fn index(cache: &LoadedCache) -> CacheResult<&Arc<MappedMetadataIndex>> {
    if let Some(index) = cache.metadata_index.get() {
        return Ok(index);
    }
    let index =
        MappedMetadataIndex::open(&cache.path.join("metadata.arrow"), cache.sample_count())?;
    // Concurrent first readers may build descriptors independently; only one mapping survives.
    let _ = cache.metadata_index.set(Arc::new(index));
    cache.metadata_index.get().ok_or(CacheError::WorkerFailed)
}

/// Convert only the requested row after validating its containing Arrow batch.
pub(super) fn read_row(
    cache: &LoadedCache,
    sample: usize,
    cached: &mut Option<CachedMetadataBatch>,
) -> CacheResult<Vec<MetadataValue>> {
    let mut read = || -> CacheResult<Vec<MetadataValue>> {
        let source = index(cache)?;
        let (batch_index, row) = source.location(sample)?;
        if !matches!(cached, Some(value) if Arc::ptr_eq(&value.source, source) && value.index == batch_index)
        {
            // Release the old batch before decoding another, including any alignment copies.
            *cached = None;
            let batch = source.batch(batch_index)?;
            validate_batch(&batch, &cache.manifest.fields().metadata_schema)?;
            *cached = Some(CachedMetadataBatch {
                source: Arc::clone(source),
                index: batch_index,
                batch,
            });
        }
        let batch = &cached.as_ref().ok_or(CacheError::WorkerFailed)?.batch;
        if row >= batch.num_rows() {
            return Err(CacheError::InvalidCache(format!(
                "metadata row {row} is out of batch range"
            )));
        }
        read_metadata_row(batch, &cache.manifest.fields().metadata_schema, row)
    };
    read().map_err(|error| {
        error.context(format!(
            "read metadata sample_id={sample} from '{}'",
            cache.path.join("metadata.arrow").display()
        ))
    })
}

/// Full validation streams batches, preserving bounded row memory even when explicitly requested.
pub(super) fn validate_file(cache: &LoadedCache) -> CacheResult<()> {
    index(cache)?;
    for batch in cache.metadata_batches()? {
        validate_batch(&batch?, &cache.manifest.fields().metadata_schema)?;
    }
    Ok(())
}

/// Reject schema drift and nulls before any unchecked Arrow cell accessor can run.
pub(super) fn validate_batch(batch: &RecordBatch, schema: &[MetadataField]) -> CacheResult<()> {
    if batch.num_columns() != schema.len() {
        return Err(CacheError::InvalidCache(format!(
            "metadata column count does not match manifest schema: expected {}, actual {}",
            schema.len(),
            batch.num_columns()
        )));
    }
    for ((actual, column), expected) in batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .zip(schema)
    {
        if actual.name() != &expected.name {
            return Err(CacheError::InvalidCache(format!(
                "metadata column name does not match '{}': actual '{}'",
                expected.name,
                actual.name()
            )));
        }
        if column.data_type() != &arrow_type(&expected.kind) || column.null_count() != 0 {
            return Err(CacheError::InvalidCache(format!(
                "metadata '{}' has invalid type or null values: expected {:?} with no nulls, actual {:?} with {} nulls",
                expected.name, arrow_type(&expected.kind), column.data_type(), column.null_count()
            )));
        }
    }
    Ok(())
}

/// Compare the legacy embedded copy only when full validation is enabled.
fn validate_redundant_metadata(
    metadata: &[MetadataValue],
    embedded_metadata: &[MetadataValue],
    sample_index: usize,
) -> CacheResult<()> {
    if metadata != embedded_metadata {
        return Err(CacheError::InvalidCache(format!(
            "embedded metadata does not match metadata.arrow for sample {sample_index}"
        )));
    }
    Ok(())
}

impl LoadedCache {
    /// Lazily load `metadata.arrow` for validation or compatibility paths that need full rows.
    pub fn with_metadata_rows<T>(
        &self,
        read_rows: impl FnOnce(&[Vec<MetadataValue>]) -> CacheResult<T>,
    ) -> CacheResult<T> {
        let mut guard = self
            .metadata_rows
            .lock()
            .map_err(|_| CacheError::WorkerFailed)?;
        if guard.is_none() {
            *guard = Some(super::read_metadata_file(
                &self.path.join("metadata.arrow"),
                &self.manifest.fields().metadata_schema,
            )?);
        }
        let rows = guard
            .as_deref()
            .ok_or_else(|| CacheError::InvalidCache("metadata rows are unavailable".to_string()))?;
        read_rows(rows)
    }
}
