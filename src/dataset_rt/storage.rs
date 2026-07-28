use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::compression::{compress_payload, decompress_payload};
use crate::types::{
    CacheError, CacheResult, CacheSample, MetadataField, MetadataKind, MetadataValue,
    ShardCompression,
};

const FORMAT_VERSION: u32 = 1;
const INDEX_ROW_BYTES: usize = 24;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    pub format_version: u32,
    pub source_name: String,
    pub sample_count: u64,
    pub metadata_schema: Vec<MetadataField>,
    pub metadata_sha256: String,
    pub index_sha256: String,
    pub shards: Vec<ShardManifest>,
}

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

#[derive(Clone, Debug)]
pub struct LoadedCache {
    pub manifest: Manifest,
    pub metadata_rows: Vec<Vec<MetadataValue>>,
    pub index: Vec<IndexEntry>,
    pub path: PathBuf,
}

impl LoadedCache {
    pub fn sample_count(&self) -> usize {
        self.index.len()
    }

    pub fn read_sample(&self, sample_index: usize) -> CacheResult<CacheSample> {
        let entry = self.index.get(sample_index).ok_or_else(|| {
            CacheError::InvalidCache(format!("sample index {sample_index} is out of range"))
        })?;
        let shard = self
            .manifest
            .shards
            .get(entry.shard_id as usize)
            .ok_or_else(|| {
                CacheError::InvalidCache(format!("shard id {} is out of range", entry.shard_id))
            })?;
        let shard_path = self.path.join("shards").join(&shard.name);
        let mut file = File::open(shard_path)?;
        let mut reader = BufReader::new(&mut file);
        let byte_len = usize::try_from(entry.byte_len).map_err(|_| {
            CacheError::InvalidCache("payload byte length does not fit in usize".to_string())
        })?;
        let mut data = vec![0; byte_len];
        use std::io::Seek;
        reader.seek(std::io::SeekFrom::Start(entry.offset))?;
        reader.read_exact(&mut data)?;
        let data = decompress_payload(&data, &shard.compression)?;

        let metadata = self.metadata_rows.get(sample_index).ok_or_else(|| {
            CacheError::InvalidCache(format!("metadata row {sample_index} is out of range"))
        })?;

        Ok(CacheSample {
            data,
            metadata: metadata.clone(),
        })
    }
}

pub struct CacheBuilder {
    path: PathBuf,
    source_name: String,
    max_shard_bytes: u64,
    schema: Vec<MetadataField>,
    metadata_rows: Vec<Vec<MetadataValue>>,
    index: Vec<IndexEntry>,
    shards: Vec<ShardManifest>,
    shard_compression: ShardCompression,
    current_shard: OpenShard,
}

struct OpenShard {
    id: u64,
    name: String,
    writer: BufWriter<File>,
    byte_len: u64,
    uncompressed_byte_len: u64,
    hasher: Sha256,
}

impl CacheBuilder {
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
            schema: Vec::new(),
            metadata_rows: Vec::new(),
            index: Vec::new(),
            shards: Vec::new(),
            shard_compression,
            current_shard,
        })
    }

    pub fn push_sample(
        &mut self,
        data: Vec<u8>,
        metadata: BTreeMap<String, MetadataValue>,
    ) -> CacheResult<()> {
        let metadata_row = self.validate_metadata(metadata)?;
        let uncompressed_byte_len = u64::try_from(data.len()).map_err(|_| {
            CacheError::InvalidInput("payload byte length does not fit in u64".to_string())
        })?;
        let stored_data = compress_payload(data, &self.shard_compression);
        let stored_byte_len = u64::try_from(stored_data.len()).map_err(|_| {
            CacheError::InvalidInput("stored payload byte length does not fit in u64".to_string())
        })?;
        self.rotate_shard_if_needed(stored_byte_len)?;

        let entry = self
            .current_shard
            .write_payload(&stored_data, uncompressed_byte_len)?;
        self.index.push(entry);
        self.metadata_rows.push(metadata_row);
        Ok(())
    }

    pub fn finish(mut self) -> CacheResult<Manifest> {
        if self.index.is_empty() {
            return Err(CacheError::InvalidInput(
                "cache source yielded no samples".to_string(),
            ));
        }

        let final_shard = self.current_shard.finish(&self.shard_compression)?;
        self.shards.push(final_shard);

        let metadata_sha256 = write_metadata_file(&self.path, &self.schema, &self.metadata_rows)?;
        let index_sha256 = write_index_file(&self.path, &self.index)?;
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            source_name: self.source_name,
            sample_count: self.index.len() as u64,
            metadata_schema: self.schema,
            metadata_sha256,
            index_sha256,
            shards: self.shards,
        };
        // The manifest is the publication marker: readers reject caches until it exists.
        write_manifest(&self.path, &manifest)?;
        Ok(manifest)
    }

    fn rotate_shard_if_needed(&mut self, next_payload_len: u64) -> CacheResult<()> {
        if self.current_shard.byte_len == 0 {
            return Ok(());
        }
        if self.current_shard.byte_len + next_payload_len <= self.max_shard_bytes {
            return Ok(());
        }

        let next_id = self.current_shard.id + 1;
        let closed = std::mem::replace(
            &mut self.current_shard,
            OpenShard::create(&self.path, next_id)?,
        );
        self.shards.push(closed.finish(&self.shard_compression)?);
        Ok(())
    }

    fn validate_metadata(
        &mut self,
        metadata: BTreeMap<String, MetadataValue>,
    ) -> CacheResult<Vec<MetadataValue>> {
        if self.schema.is_empty() {
            self.schema = metadata
                .iter()
                .map(|(name, value)| MetadataField {
                    name: name.clone(),
                    kind: value.kind(),
                })
                .collect();
        }

        if metadata.len() != self.schema.len() {
            return Err(CacheError::InvalidInput(
                "metadata keys must be identical for every sample".to_string(),
            ));
        }

        self.schema
            .iter()
            .map(|field| match metadata.get(&field.name) {
                Some(value) if value.kind() == field.kind => Ok(value.clone()),
                Some(value) => Err(CacheError::InvalidInput(format!(
                    "metadata field '{}' expected {}, got {}",
                    field.name,
                    field.kind,
                    value.kind()
                ))),
                None => Err(CacheError::InvalidInput(format!(
                    "metadata field '{}' is missing",
                    field.name
                ))),
            })
            .collect()
    }
}

impl OpenShard {
    fn create(cache_path: &Path, id: u64) -> CacheResult<Self> {
        let name = format!("{id:06}.bin");
        let file = File::create(cache_path.join("shards").join(&name))?;
        Ok(Self {
            id,
            name,
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
        self.writer.write_all(data)?;
        self.hasher.update(data);
        self.byte_len += u64::try_from(data.len()).map_err(|_| {
            CacheError::InvalidInput("stored payload byte length does not fit in u64".to_string())
        })?;
        self.uncompressed_byte_len = self
            .uncompressed_byte_len
            .checked_add(uncompressed_byte_len)
            .ok_or_else(|| {
                CacheError::InvalidInput(
                    "shard uncompressed byte length overflowed u64".to_string(),
                )
            })?;
        Ok(IndexEntry {
            shard_id: self.id,
            offset,
            byte_len: data.len() as u64,
        })
    }

    fn finish(mut self, compression: &ShardCompression) -> CacheResult<ShardManifest> {
        self.writer.flush()?;
        Ok(ShardManifest {
            name: self.name,
            uncompressed_byte_len: self.uncompressed_byte_len,
            byte_len: self.byte_len,
            compression: compression.clone(),
            sha256: hex::encode(self.hasher.finalize()),
        })
    }
}

pub fn load_cache(path: PathBuf) -> CacheResult<LoadedCache> {
    let manifest = read_manifest(&path)?;
    validate_manifest_version(&manifest)?;

    let metadata_path = path.join("metadata.arrow");
    let index_path = path.join("index.bin");
    verify_file_checksum(&metadata_path, &manifest.metadata_sha256)?;
    verify_file_checksum(&index_path, &manifest.index_sha256)?;
    verify_shards(&path, &manifest)?;

    let metadata_rows = read_metadata_file(&metadata_path, &manifest.metadata_schema)?;
    let index = read_index_file(&index_path)?;
    validate_loaded_shapes(&manifest, &metadata_rows, &index)?;

    Ok(LoadedCache {
        manifest,
        metadata_rows,
        index,
        path,
    })
}

fn write_manifest(path: &Path, manifest: &Manifest) -> CacheResult<()> {
    let manifest_path = path.join("manifest.json");
    let file = File::create(manifest_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), manifest)?;
    Ok(())
}

fn read_manifest(path: &Path) -> CacheResult<Manifest> {
    let manifest_path = path.join("manifest.json");
    if !manifest_path.exists() {
        return Err(CacheError::InvalidCache(format!(
            "missing manifest: {}",
            manifest_path.display()
        )));
    }
    let file = File::open(manifest_path)?;
    Ok(serde_json::from_reader(BufReader::new(file))?)
}

fn validate_manifest_version(manifest: &Manifest) -> CacheResult<()> {
    if manifest.format_version != FORMAT_VERSION {
        return Err(CacheError::InvalidCache(format!(
            "unsupported format version {}",
            manifest.format_version
        )));
    }
    Ok(())
}

fn write_metadata_file(
    path: &Path,
    schema: &[MetadataField],
    rows: &[Vec<MetadataValue>],
) -> CacheResult<String> {
    let arrow_schema = Arc::new(Schema::new(
        schema
            .iter()
            .map(|field| Field::new(&field.name, arrow_type(&field.kind), false))
            .collect::<Vec<_>>(),
    ));
    let columns = schema
        .iter()
        .enumerate()
        .map(|(column_index, field)| build_array(column_index, &field.kind, rows))
        .collect::<CacheResult<Vec<_>>>()?;
    let batch = RecordBatch::try_new(arrow_schema.clone(), columns)?;
    let metadata_path = path.join("metadata.arrow");
    let file = File::create(&metadata_path)?;
    let mut writer = FileWriter::try_new(BufWriter::new(file), &arrow_schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    hash_file(&metadata_path)
}

fn read_metadata_file(
    path: &Path,
    schema: &[MetadataField],
) -> CacheResult<Vec<Vec<MetadataValue>>> {
    let file = File::open(path)?;
    let reader = FileReader::try_new(BufReader::new(file), None)?;
    let mut rows = Vec::new();

    for batch in reader {
        let batch = batch?;
        for row_index in 0..batch.num_rows() {
            rows.push(read_metadata_row(&batch, schema, row_index)?);
        }
    }

    Ok(rows)
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
            let column = batch.column(column_index);
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

fn write_index_file(path: &Path, index: &[IndexEntry]) -> CacheResult<String> {
    let index_path = path.join("index.bin");
    let mut file = BufWriter::new(File::create(&index_path)?);
    for entry in index {
        file.write_all(&entry.shard_id.to_le_bytes())?;
        file.write_all(&entry.offset.to_le_bytes())?;
        file.write_all(&entry.byte_len.to_le_bytes())?;
    }
    file.flush()?;
    hash_file(&index_path)
}

fn read_index_file(path: &Path) -> CacheResult<Vec<IndexEntry>> {
    let bytes = fs::read(path)?;
    if bytes.len() % INDEX_ROW_BYTES != 0 {
        return Err(CacheError::InvalidCache(
            "index length is not divisible by row size".to_string(),
        ));
    }

    bytes
        .chunks_exact(INDEX_ROW_BYTES)
        .map(read_index_entry)
        .collect()
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
    bytes
        .get(start..end)
        .ok_or_else(|| CacheError::InvalidCache("invalid index row".to_string()))
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
            "checksum mismatch for {}",
            path.display()
        )));
    }
    Ok(())
}

fn verify_shards(path: &Path, manifest: &Manifest) -> CacheResult<()> {
    for shard in &manifest.shards {
        let shard_path = path.join("shards").join(&shard.name);
        let metadata = fs::metadata(&shard_path)?;
        if metadata.len() != shard.byte_len {
            return Err(CacheError::InvalidCache(format!(
                "shard length mismatch for {}",
                shard_path.display()
            )));
        }
        verify_file_checksum(&shard_path, &shard.sha256)?;
    }
    Ok(())
}

fn validate_loaded_shapes(
    manifest: &Manifest,
    metadata_rows: &[Vec<MetadataValue>],
    index: &[IndexEntry],
) -> CacheResult<()> {
    let sample_count = manifest.sample_count as usize;
    if metadata_rows.len() != sample_count {
        return Err(CacheError::InvalidCache(
            "metadata row count does not match manifest".to_string(),
        ));
    }
    if index.len() != sample_count {
        return Err(CacheError::InvalidCache(
            "index row count does not match manifest".to_string(),
        ));
    }
    for entry in index {
        if manifest.shards.get(entry.shard_id as usize).is_none() {
            return Err(CacheError::InvalidCache(format!(
                "index references missing shard {}",
                entry.shard_id
            )));
        }
    }
    Ok(())
}

fn hash_file(path: &Path) -> CacheResult<String> {
    let mut file = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
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
