use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;
use arrow_schema::DataType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::types::{
    native_result, CacheError, CacheId, CacheResult, CacheSample, MetadataField, MetadataKind,
    MetadataValue, ShardCompression,
};

pub const FORMAT_VERSION: u32 = 5;
mod index;
mod manifest;
mod metadata;
mod record;
use record::{EncodedRecord, RecordLayout};
mod write_tables;
use write_tables::CacheTablesWriter;

pub mod migration;
#[cfg(test)]
mod tests;
use index::MappedIndex;
pub use manifest::{Manifest, ManifestFields, ManifestV3};
const INDEX_ROW_BYTES: usize = 24;
const MAX_CACHED_SHARD_READERS: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardManifest {
    pub name: String,
    pub uncompressed_byte_len: u64,
    pub byte_len: u64,
    pub compression: ShardCompression,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct IndexEntry {
    pub shard_id: u64,
    pub offset: u64,
    pub byte_len: u64,
}

#[derive(Debug)]
pub struct LoadedCache {
    pub cache_id: CacheId,
    pub manifest: Manifest,
    metadata_rows: Mutex<Option<Vec<Vec<MetadataValue>>>>,
    pub index: MappedIndex,
    pub path: PathBuf,
    validate_embedded_metadata: bool,
    record_layout: RecordLayout,
    metadata_index: OnceLock<Arc<crate::arrow_metadata::mapped::MappedMetadataIndex>>,
}

pub struct ShardReaderCache {
    readers: BTreeMap<PathBuf, BufReader<File>>,
    recent_paths: VecDeque<PathBuf>,
    max_readers: usize,
    metadata_batch: Option<metadata::CachedMetadataBatch>,
}

impl LoadedCache {
    pub fn sample_count(&self) -> usize {
        self.index.len()
    }

    /// Open `metadata.arrow` as Arrow batches without materializing Rust row values.
    pub fn metadata_batches(
        &self,
    ) -> CacheResult<crate::arrow_metadata::mapped::MappedMetadataReader> {
        crate::arrow_metadata::mapped::MappedMetadataReader::open(&self.path.join("metadata.arrow"))
            .map_err(|error| {
                error.context(format!(
                    "open metadata batches '{}'",
                    self.path.join("metadata.arrow").display()
                ))
            })
    }

    /// Read one sample record using caller-owned shard handles for repeated random access.
    pub fn read_sample_with_cache(
        &self,
        sample_index: usize,
        readers: &mut ShardReaderCache,
    ) -> CacheResult<CacheSample> {
        self.read_sample_record(sample_index, readers)
            .map_err(|error| {
                error.context(format!(
                    "read sample cache_id={} sample_id={sample_index} in '{}'",
                    self.cache_id.as_u64(),
                    self.path.display()
                ))
            })
    }

    /// Attach shard addressing to failures without changing record validation or allocation.
    fn read_sample_record(
        &self,
        sample_index: usize,
        readers: &mut ShardReaderCache,
    ) -> CacheResult<CacheSample> {
        let entry = self.index.get(sample_index)?;
        let shard = self
            .manifest
            .fields()
            .shards
            .get(entry.shard_id as usize)
            .ok_or_else(|| {
                CacheError::InvalidCache(format!("shard id {} is out of range", entry.shard_id))
            })?;
        let shard_path = self.path.join("shards").join(&shard.name);
        let byte_len = usize::try_from(entry.byte_len).map_err(|_| {
            CacheError::InvalidCache("payload byte length does not fit in usize".to_string())
        })?;
        let end = entry.offset.checked_add(entry.byte_len).ok_or_else(|| {
            CacheError::InvalidCache("record offset and length overflowed u64".to_string())
        })?;
        let reader = readers
            .reader_for(&shard_path)
            .map_err(|error| error.context(format!("open shard '{}'", shard_path.display())))?;
        if end > shard.byte_len {
            return Err(CacheError::InvalidCache(
                format!("record extends beyond shard length: shard='{}', offset={}, length={}, shard length={}", shard_path.display(), entry.offset, entry.byte_len, shard.byte_len),
            ));
        }
        let mut record = Vec::new();
        record.try_reserve_exact(byte_len).map_err(|error| {
            CacheError::InvalidCache(format!(
                "cannot allocate record of {byte_len} bytes: {error}"
            ))
        })?;
        record.resize(byte_len, 0);
        let address = || {
            format!(
                "read shard '{}' at offset {} for {} bytes",
                shard_path.display(),
                entry.offset,
                entry.byte_len
            )
        };
        reader
            .seek(std::io::SeekFrom::Start(entry.offset))
            .map_err(|error| CacheError::from(error).context(address()))?;
        reader
            .read_exact(&mut record)
            .map_err(|error| CacheError::from(error).context(address()))?;
        let decoded = self
            .record_layout
            .decode(record, &self.manifest.fields().metadata_schema, shard)
            .map_err(|error| error.context(address()))?;
        metadata::materialize(decoded, self, sample_index, &mut readers.metadata_batch)
    }
}

impl ShardReaderCache {
    /// Create a bounded worker-local shard reader cache for random sample materialization.
    pub fn new() -> Self {
        Self {
            readers: BTreeMap::new(),
            recent_paths: VecDeque::new(),
            max_readers: MAX_CACHED_SHARD_READERS,
            metadata_batch: None,
        }
    }

    /// Return an open reader for a shard path, opening and evicting handles as needed.
    fn reader_for(&mut self, path: &Path) -> CacheResult<&mut BufReader<File>> {
        if self.readers.contains_key(path) {
            self.mark_recent(path);
            return self.reader_mut(path);
        }

        self.open_reader(path)?;
        self.reader_mut(path)
    }

    /// Record recent use so repeated shard reads stay open.
    fn mark_recent(&mut self, path: &Path) {
        self.recent_paths
            .retain(|existing_path| existing_path.as_path() != path);
        self.recent_paths.push_back(path.to_path_buf());
    }

    /// Open a shard reader after evicting the oldest cached reader when at capacity.
    fn open_reader(&mut self, path: &Path) -> CacheResult<()> {
        self.evict_if_full();
        self.readers
            .insert(path.to_path_buf(), BufReader::new(File::open(path)?));
        self.recent_paths.push_back(path.to_path_buf());
        Ok(())
    }

    /// Keep the cache below its file descriptor budget.
    fn evict_if_full(&mut self) {
        while self.readers.len() >= self.max_readers {
            let Some(path) = self.recent_paths.pop_front() else {
                break;
            };
            self.readers.remove(&path);
        }
    }

    /// Fetch a cached reader after the cache has guaranteed it exists.
    fn reader_mut(&mut self, path: &Path) -> CacheResult<&mut BufReader<File>> {
        self.readers
            .get_mut(path)
            .ok_or_else(|| CacheError::InvalidCache("shard reader is unavailable".to_string()))
    }
}

impl Default for ShardReaderCache {
    /// Create the default bounded cache used by reader workers.
    fn default() -> Self {
        Self::new()
    }
}

pub struct CacheBuilder {
    path: PathBuf,
    source_name: String,
    max_shard_bytes: u64,
    tables: Option<CacheTablesWriter>,
    shards: Vec<ShardManifest>,
    shard_compression: ShardCompression,
    current_shard: OpenShard,
    record_layout: RecordLayout,
}

/// Timing data returned by one sample write.
///
/// Storage returns plain measurements instead of writing profiler output
/// itself; the writer layer decides whether and where to publish them.
#[derive(Default)]
pub struct PushSampleStats {
    pub metadata_validate: Duration,
    pub compression: Duration,
    pub record_encode: Duration,
    pub disk_write: Duration,
    pub uncompressed_bytes: u64,
    pub stored_bytes: u64,
}

/// Timing data returned by final cache materialization.
///
/// These stages are grouped around durable side effects: shard flush,
/// metadata/index files, and the manifest publication marker.
#[derive(Default)]
pub struct FinishStats {
    pub shard_flush: Duration,
    pub metadata: Duration,
    pub index: Duration,
    pub manifest: Duration,
}

struct OpenShard {
    id: u64,
    name: String,
    path: PathBuf,
    writer: BufWriter<File>,
    byte_len: u64,
    uncompressed_byte_len: u64,
    hasher: Sha256,
}

impl CacheBuilder {
    /// Create unpublished cache artifacts owned exclusively by this write operation.
    pub fn create(
        path: PathBuf,
        source_name: String,
        max_shard_bytes: u64,
        shard_compression: ShardCompression,
    ) -> CacheResult<Self> {
        if path.exists() {
            return Err(CacheError::InvalidInput(format!(
                "cache path already exists: {}",
                path.display()
            )));
        }

        fs::create_dir_all(path.join("shards"))?;
        let current_shard = OpenShard::create(&path, 0)?;

        Ok(Self {
            path,
            source_name,
            max_shard_bytes,
            tables: None,
            shards: Vec::new(),
            record_layout: RecordLayout::for_write(&shard_compression),
            shard_compression,
            current_shard,
        })
    }

    /// Commit one validated sample in physical order without retaining dataset-wide rows.
    pub fn push_sample(
        &mut self,
        data: Vec<u8>,
        metadata: BTreeMap<String, MetadataValue>,
    ) -> CacheResult<PushSampleStats> {
        let metadata_started_at = Instant::now();
        if self.tables.is_none() {
            self.tables = Some(CacheTablesWriter::create(&self.path, &metadata)?);
        }
        let tables = self.tables.as_mut().ok_or(CacheError::WorkerFailed)?;
        let metadata_row = tables.metadata_row(metadata)?;
        let metadata_validate = metadata_started_at.elapsed();
        let EncodedRecord {
            bytes: record,
            uncompressed_byte_len,
            compression,
            record_encode,
        } = self.record_layout.encode(
            data,
            &tables.schema,
            &metadata_row,
            &self.shard_compression,
        )?;
        let stored_byte_len = u64::try_from(record.len()).map_err(|_| {
            CacheError::InvalidInput("stored payload byte length does not fit in u64".to_string())
        })?;
        self.rotate_shard_if_needed(stored_byte_len)?;

        let disk_started_at = Instant::now();
        let entry = self
            .current_shard
            .write_payload(&record, uncompressed_byte_len)?;
        self.tables
            .as_mut()
            .ok_or(CacheError::WorkerFailed)?
            .push(entry, metadata_row)?;
        let disk_write = disk_started_at.elapsed();
        Ok(PushSampleStats {
            metadata_validate,
            compression,
            record_encode,
            disk_write,
            uncompressed_bytes: uncompressed_byte_len,
            stored_bytes: stored_byte_len,
        })
    }

    /// Publish the manifest only after every referenced file and directory entry is synced.
    pub fn finish(mut self) -> CacheResult<(Manifest, FinishStats)> {
        let tables = self.tables.take().ok_or_else(|| {
            CacheError::InvalidInput("cache source yielded no samples".to_string())
        })?;

        let shard_started_at = Instant::now();
        let final_shard = self.current_shard.finish(&self.shard_compression)?;
        let shard_flush = shard_started_at.elapsed();
        self.shards.push(final_shard);

        let tables = tables.finish()?;
        sync_directory(&self.path.join("shards"))?;
        sync_directory(&self.path)?;
        let manifest_data = ManifestV3 {
            cache_id: cache_id_from_name(&self.source_name),
            fields: ManifestFields {
                source_name: self.source_name,
                sample_count: tables.sample_count,
                metadata_schema: tables.schema,
                metadata_sha256: tables.metadata_sha256,
                index_sha256: tables.index_sha256,
                shards: self.shards,
                extra: BTreeMap::new(),
            },
        };
        let manifest = Manifest::for_layout(self.record_layout, manifest_data);
        // The manifest is the publication marker: readers reject caches until it exists.
        let manifest_started_at = Instant::now();
        write_manifest(&self.path, &manifest)?;
        sync_directory(&self.path)?;
        let manifest_time = manifest_started_at.elapsed();
        Ok((
            manifest,
            FinishStats {
                shard_flush,
                metadata: tables.metadata_time,
                index: tables.index_time,
                manifest: manifest_time,
            },
        ))
    }

    fn rotate_shard_if_needed(&mut self, next_payload_len: u64) -> CacheResult<()> {
        if self.current_shard.byte_len == 0 {
            return Ok(());
        }
        if next_payload_len
            <= self
                .max_shard_bytes
                .saturating_sub(self.current_shard.byte_len)
        {
            return Ok(());
        }

        let next_id = self
            .current_shard
            .id
            .checked_add(1)
            .ok_or_else(|| CacheError::InvalidInput("shard id overflowed u64".to_string()))?;
        let closed = std::mem::replace(
            &mut self.current_shard,
            OpenShard::create(&self.path, next_id)?,
        );
        self.shards.push(closed.finish(&self.shard_compression)?);
        Ok(())
    }
}

impl OpenShard {
    fn create(cache_path: &Path, id: u64) -> CacheResult<Self> {
        let name = format!("{id:06}.bin");
        let path = cache_path.join("shards").join(&name);
        let file = File::create(&path).map_err(|error| {
            CacheError::from(error).context(format!("create shard '{}'", path.display()))
        })?;
        Ok(Self {
            id,
            name,
            path,
            writer: BufWriter::new(file),
            byte_len: 0,
            uncompressed_byte_len: 0,
            hasher: Sha256::new(),
        })
    }

    fn write_payload(
        &mut self,
        data: &[u8],
        uncompressed_byte_len: u64,
    ) -> CacheResult<IndexEntry> {
        let offset = self.byte_len;
        let byte_len = self
            .byte_len
            .checked_add(u64::try_from(data.len()).map_err(|_| {
                CacheError::InvalidInput(
                    "stored payload byte length does not fit in u64".to_string(),
                )
            })?)
            .ok_or_else(|| {
                CacheError::InvalidInput("shard byte length overflowed u64".to_string())
            })?;
        let uncompressed_byte_len = self
            .uncompressed_byte_len
            .checked_add(uncompressed_byte_len)
            .ok_or_else(|| {
                CacheError::InvalidInput(
                    "shard uncompressed byte length overflowed u64".to_string(),
                )
            })?;
        self.writer.write_all(data)?;
        self.hasher.update(data);
        self.byte_len = byte_len;
        self.uncompressed_byte_len = uncompressed_byte_len;
        Ok(IndexEntry {
            shard_id: self.id,
            offset,
            byte_len: data.len() as u64,
        })
    }

    fn finish(mut self, compression: &ShardCompression) -> CacheResult<ShardManifest> {
        self.writer
            .flush()
            .and_then(|()| self.writer.get_ref().sync_all())
            .map_err(|error| {
                CacheError::from(error).context(format!("finish shard '{}'", self.path.display()))
            })?;
        Ok(ShardManifest {
            name: self.name,
            uncompressed_byte_len: self.uncompressed_byte_len,
            byte_len: self.byte_len,
            compression: compression.clone(),
            sha256: hex::encode(self.hasher.finalize()),
        })
    }
}

/// Resolve legacy identity only at the loading boundary; v3 IDs come from disk.
pub fn load_cache(
    path: PathBuf,
    validate_cache: bool,
    position: usize,
) -> CacheResult<LoadedCache> {
    let path = std::path::absolute(&path).map_err(CacheError::from)?;
    load_cache_inner(path.clone(), validate_cache, position)
        .map_err(|error| error.context(format!("load cache '{}'", path.display())))
}

/// Validate published cache files without expanding metadata on ordinary restarts.
fn load_cache_inner(
    path: PathBuf,
    validate_cache: bool,
    position: usize,
) -> CacheResult<LoadedCache> {
    let manifest = read_manifest(&path)?;
    let cache_id = manifest.resolve_cache_id(position)?;

    let metadata_path = path.join("metadata.arrow");
    let index_path = path.join("index.bin");
    if validate_cache {
        verify_file_checksum(&metadata_path, &manifest.fields().metadata_sha256)?;
        verify_file_checksum(&index_path, &manifest.fields().index_sha256)?;
    }
    verify_shards(&path, &manifest, validate_cache)?;

    let index = MappedIndex::open(&index_path)?;
    let record_layout = manifest.record_layout();
    let metadata_rows = metadata::prepare_rows(
        record_layout,
        &metadata_path,
        &manifest.fields().metadata_schema,
        validate_cache,
    )?;
    validate_loaded_shapes(&manifest, metadata_rows.as_deref(), &index)?;

    let cache = LoadedCache {
        cache_id,
        manifest,
        metadata_rows: Mutex::new(metadata_rows),
        index,
        path,
        validate_embedded_metadata: validate_cache,
        record_layout,
        metadata_index: OnceLock::new(),
    };
    metadata::validate_prepared(&cache, validate_cache)?;
    Ok(cache)
}

/// Propagate serialization, final flush, and durability failures for the publication marker.
fn write_manifest(path: &Path, manifest: &Manifest) -> CacheResult<()> {
    let manifest_path = path.join("manifest.json");
    let write = || -> CacheResult<()> {
        let mut writer = BufWriter::new(File::create(&manifest_path)?);
        write_manifest_contents(&mut writer, manifest)?;
        writer.get_ref().sync_all()?;
        Ok(())
    };
    write().map_err(|error| error.context(format!("write manifest '{}'", manifest_path.display())))
}

/// Finish buffered output explicitly so drop cannot hide a failed final write.
fn write_manifest_contents(writer: &mut impl Write, manifest: &Manifest) -> CacheResult<()> {
    serde_json::to_writer_pretty(&mut *writer, manifest)?;
    writer.flush()?;
    Ok(())
}

/// Persist directory entries before acknowledging cache creation or publication.
pub(crate) fn sync_directory(path: &Path) -> CacheResult<()> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| {
            CacheError::from(error).context(format!("sync directory '{}'", path.display()))
        })
}

/// Open the publication marker directly so access failures retain their OS cause.
fn read_manifest(path: &Path) -> CacheResult<Manifest> {
    let manifest_path = path.join("manifest.json");
    let file = File::open(&manifest_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CacheError::InvalidCache(format!(
                "missing manifest '{}': {error}. Each cache directory must directly contain manifest.json; check the supplied path and whether cache writing completed",
                manifest_path.display()
            ))
        } else {
            CacheError::from(error).context(format!(
                "open manifest '{}'; pass a cache directory containing manifest.json",
                manifest_path.display()
            ))
        }
    })?;
    serde_json::from_reader(BufReader::new(file)).map_err(|error| {
        CacheError::from(error).context(format!("parse manifest '{}'", manifest_path.display()))
    })
}

/// Hash exact UTF-8 names with a fixed, portable 63-bit manifest identity rule.
pub fn cache_id_from_name(name: &str) -> u64 {
    let digest = Sha256::digest(name.as_bytes());
    let mut prefix = [0_u8; 8];
    for (target, byte) in prefix.iter_mut().zip(digest.iter()) {
        *target = *byte;
    }
    u64::from_be_bytes(prefix) & i64::MAX as u64
}

fn read_metadata_file(
    path: &Path,
    schema: &[MetadataField],
) -> CacheResult<Vec<Vec<MetadataValue>>> {
    native_result("metadata Arrow validation", || {
        let file = File::open(path)?;
        let reader = FileReader::try_new(BufReader::new(file), None)?;
        let mut rows = Vec::new();

        for batch in reader {
            let batch = batch?;
            metadata::validate_batch(&batch, schema)?;
            for row_index in 0..batch.num_rows() {
                rows.push(read_metadata_row(&batch, schema, row_index)?);
            }
        }

        Ok(rows)
    })
    .map_err(|error| error.context(format!("read metadata file '{}'", path.display())))
}

fn read_metadata_row(
    batch: &RecordBatch,
    schema: &[MetadataField],
    row_index: usize,
) -> CacheResult<Vec<MetadataValue>> {
    schema
        .iter()
        .enumerate()
        .map(|(column_index, field)| {
            let column = batch.columns().get(column_index).ok_or_else(|| {
                CacheError::InvalidCache("metadata column is missing".to_string())
            })?;
            match field.kind {
                MetadataKind::Bool => {
                    let array =
                        column
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .ok_or_else(|| {
                                CacheError::InvalidCache(format!(
                                    "metadata '{}' is not bool",
                                    field.name
                                ))
                            })?;
                    Ok(MetadataValue::Bool(array.value(row_index)))
                }
                MetadataKind::Int => {
                    let array = column
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            CacheError::InvalidCache(format!(
                                "metadata '{}' is not int",
                                field.name
                            ))
                        })?;
                    Ok(MetadataValue::Int(array.value(row_index)))
                }
                MetadataKind::Float => {
                    let array =
                        column
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| {
                                CacheError::InvalidCache(format!(
                                    "metadata '{}' is not float",
                                    field.name
                                ))
                            })?;
                    Ok(MetadataValue::Float(array.value(row_index)))
                }
                MetadataKind::String => {
                    let array = column
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            CacheError::InvalidCache(format!(
                                "metadata '{}' is not string",
                                field.name
                            ))
                        })?;
                    Ok(MetadataValue::String(array.value(row_index).to_string()))
                }
            }
        })
        .collect()
}

fn arrow_type(kind: &MetadataKind) -> DataType {
    match kind {
        MetadataKind::Bool => DataType::Boolean,
        MetadataKind::Int => DataType::Int64,
        MetadataKind::Float => DataType::Float64,
        MetadataKind::String => DataType::Utf8,
    }
}

fn build_array(
    column_index: usize,
    kind: &MetadataKind,
    rows: &[Vec<MetadataValue>],
) -> CacheResult<ArrayRef> {
    match kind {
        MetadataKind::Bool => {
            let values = rows
                .iter()
                .map(|row| match metadata_cell(row, column_index)? {
                    MetadataValue::Bool(value) => Ok(*value),
                    _ => Err(CacheError::InvalidInput(
                        "metadata kind drifted".to_string(),
                    )),
                })
                .collect::<CacheResult<Vec<_>>>()?;
            Ok(Arc::new(BooleanArray::from(values)))
        }
        MetadataKind::Int => {
            let values = rows
                .iter()
                .map(|row| match metadata_cell(row, column_index)? {
                    MetadataValue::Int(value) => Ok(*value),
                    _ => Err(CacheError::InvalidInput(
                        "metadata kind drifted".to_string(),
                    )),
                })
                .collect::<CacheResult<Vec<_>>>()?;
            Ok(Arc::new(Int64Array::from(values)))
        }
        MetadataKind::Float => {
            let values = rows
                .iter()
                .map(|row| match metadata_cell(row, column_index)? {
                    MetadataValue::Float(value) => Ok(*value),
                    _ => Err(CacheError::InvalidInput(
                        "metadata kind drifted".to_string(),
                    )),
                })
                .collect::<CacheResult<Vec<_>>>()?;
            Ok(Arc::new(Float64Array::from(values)))
        }
        MetadataKind::String => {
            let values = rows
                .iter()
                .map(|row| match metadata_cell(row, column_index)? {
                    MetadataValue::String(value) => Ok(value.as_str()),
                    _ => Err(CacheError::InvalidInput(
                        "metadata kind drifted".to_string(),
                    )),
                })
                .collect::<CacheResult<Vec<_>>>()?;
            Ok(Arc::new(StringArray::from(values)))
        }
    }
}

fn metadata_cell(row: &[MetadataValue], column_index: usize) -> CacheResult<&MetadataValue> {
    row.get(column_index)
        .ok_or_else(|| CacheError::InvalidInput("metadata column is out of range".to_string()))
}

fn read_index_entry(bytes: &[u8]) -> CacheResult<IndexEntry> {
    let shard_id = read_u64(index_field(bytes, 0, 8)?)?;
    let offset = read_u64(index_field(bytes, 8, 16)?)?;
    let byte_len = read_u64(index_field(bytes, 16, 24)?)?;
    Ok(IndexEntry {
        shard_id,
        offset,
        byte_len,
    })
}

fn index_field(bytes: &[u8], start: usize, end: usize) -> CacheResult<&[u8]> {
    record_range(bytes, start, end)
}

fn record_range(bytes: &[u8], start: usize, end: usize) -> CacheResult<&[u8]> {
    bytes
        .get(start..end)
        .ok_or_else(|| CacheError::InvalidCache("invalid binary record".to_string()))
}

fn read_u64(bytes: &[u8]) -> CacheResult<u64> {
    let fixed: [u8; 8] = bytes
        .try_into()
        .map_err(|_| CacheError::InvalidCache("invalid index integer".to_string()))?;
    Ok(u64::from_le_bytes(fixed))
}

fn verify_file_checksum(path: &Path, expected: &str) -> CacheResult<()> {
    let actual = hash_file(path)?;
    if actual != expected {
        return Err(CacheError::InvalidCache(format!(
            "checksum mismatch for {}: expected {expected}, actual {actual}. Restore this file from a known-good copy or rebuild the cache",
            path.display()
        )));
    }
    Ok(())
}

fn verify_shards(path: &Path, manifest: &Manifest, validate_cache: bool) -> CacheResult<()> {
    for shard in &manifest.fields().shards {
        let shard_path = path.join("shards").join(&shard.name);
        let metadata = fs::metadata(&shard_path).map_err(|error| {
            CacheError::from(error).context(format!("inspect shard '{}'", shard_path.display()))
        })?;
        if metadata.len() != shard.byte_len {
            return Err(CacheError::InvalidCache(format!(
                "shard length mismatch for {}: expected {} bytes, actual {} bytes",
                shard_path.display(),
                shard.byte_len,
                metadata.len()
            )));
        }
        if validate_cache {
            verify_file_checksum(&shard_path, &shard.sha256)?;
        }
    }
    Ok(())
}

fn validate_loaded_shapes(
    manifest: &Manifest,
    metadata_rows: Option<&[Vec<MetadataValue>]>,
    index: &MappedIndex,
) -> CacheResult<()> {
    let sample_count = manifest.fields().sample_count as usize;
    if let Some(rows) = metadata_rows {
        if rows.len() != sample_count {
            return Err(CacheError::InvalidCache(format!(
                "metadata row count does not match manifest: expected {sample_count}, actual {}",
                rows.len()
            )));
        }
    }
    if index.len() != sample_count {
        return Err(CacheError::InvalidCache(format!(
            "index row count does not match manifest: expected {sample_count}, actual {}",
            index.len()
        )));
    }
    for entry in index.iter() {
        let entry = entry?;
        if manifest
            .fields()
            .shards
            .get(entry.shard_id as usize)
            .is_none()
        {
            return Err(CacheError::InvalidCache(format!(
                "index references missing shard {}",
                entry.shard_id
            )));
        }
    }
    Ok(())
}

fn hash_file(path: &Path) -> CacheResult<String> {
    let mut file = BufReader::new(File::open(path).map_err(|error| {
        CacheError::from(error).context(format!("open file for checksum '{}'", path.display()))
    })?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            CacheError::from(error).context(format!("read file for checksum '{}'", path.display()))
        })?;
        if read == 0 {
            break;
        }
        let chunk = buffer.get(..read).ok_or_else(|| {
            CacheError::InvalidCache("hash buffer read is out of range".to_string())
        })?;
        hasher.update(chunk);
    }

    Ok(hex::encode(hasher.finalize()))
}
