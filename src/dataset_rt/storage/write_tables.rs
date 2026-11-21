//! Stream final cache artifacts while retaining at most one metadata batch.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{Field, Schema};

use super::{arrow_type, build_array, hash_file, IndexEntry};
use crate::types::{native_result, CacheError, CacheResult, MetadataField, MetadataValue};

const MAX_BATCH_ROWS: usize = 65_536;
const MAX_BATCH_BYTES: usize = 8 * 1024 * 1024;

pub(super) struct CacheTablesWriter {
    pub schema: Vec<MetadataField>,
    path: PathBuf,
    metadata: FileWriter<BufWriter<File>>,
    index: BufWriter<File>,
    rows: Vec<Vec<MetadataValue>>,
    row_bytes: usize,
    sample_count: u64,
}

pub(super) struct FinishedTables {
    pub schema: Vec<MetadataField>,
    pub sample_count: u64,
    pub metadata_sha256: String,
    pub index_sha256: String,
    pub metadata_time: Duration,
    pub index_time: Duration,
}

impl CacheTablesWriter {
    /// Freeze even an empty first-row schema and open the final table artifacts.
    pub fn create(path: &Path, metadata: &BTreeMap<String, MetadataValue>) -> CacheResult<Self> {
        let schema = metadata
            .iter()
            .map(|(name, value)| MetadataField {
                name: name.clone(),
                kind: value.kind(),
            })
            .collect::<Vec<_>>();
        let arrow_schema = Arc::new(Schema::new(
            schema
                .iter()
                .map(|field| Field::new(&field.name, arrow_type(&field.kind), false))
                .collect::<Vec<_>>(),
        ));
        let metadata_path = path.join("metadata.arrow");
        let index_path = path.join("index.bin");
        let metadata = native_result("create metadata Arrow writer", || {
            Ok(FileWriter::try_new(
                BufWriter::new(File::create(&metadata_path)?),
                &arrow_schema,
            )?)
        })
        .map_err(|error| error.context(format!("create '{}'", metadata_path.display())))?;
        let index = BufWriter::new(File::create(&index_path).map_err(|error| {
            CacheError::from(error).context(format!("create '{}'", index_path.display()))
        })?);
        Ok(Self {
            schema,
            path: path.to_path_buf(),
            metadata,
            index,
            rows: Vec::new(),
            row_bytes: 0,
            sample_count: 0,
        })
    }

    /// Reject schema drift before a sample reaches either the shard or table artifacts.
    pub fn metadata_row(
        &self,
        mut metadata: BTreeMap<String, MetadataValue>,
    ) -> CacheResult<Vec<MetadataValue>> {
        if metadata.len() != self.schema.len() {
            return Err(CacheError::InvalidInput(
                "metadata keys must be identical for every sample".to_string(),
            ));
        }
        self.schema
            .iter()
            .map(|field| match metadata.remove(&field.name) {
                Some(value) if value.kind() == field.kind => Ok(value),
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

    /// Append the index immediately; bound retained rows by count and owned cell bytes.
    pub fn push(&mut self, entry: IndexEntry, row: Vec<MetadataValue>) -> CacheResult<()> {
        let row_bytes = metadata_row_bytes(&row)?;
        if !self.rows.is_empty()
            && (self.rows.len() >= MAX_BATCH_ROWS
                || row_bytes > MAX_BATCH_BYTES.saturating_sub(self.row_bytes))
        {
            self.flush_metadata()?;
        }
        self.index
            .write_all(&entry.shard_id.to_le_bytes())
            .and_then(|()| self.index.write_all(&entry.offset.to_le_bytes()))
            .and_then(|()| self.index.write_all(&entry.byte_len.to_le_bytes()))
            .map_err(|error| {
                CacheError::from(error).context(format!(
                    "append index '{}'",
                    self.path.join("index.bin").display()
                ))
            })?;
        self.row_bytes = self.row_bytes.checked_add(row_bytes).ok_or_else(|| {
            CacheError::InvalidInput("metadata batch byte length overflowed usize".to_string())
        })?;
        self.rows.push(row);
        self.sample_count = self
            .sample_count
            .checked_add(1)
            .ok_or_else(|| CacheError::InvalidInput("sample count overflowed u64".to_string()))?;
        if self.rows.len() >= MAX_BATCH_ROWS || self.row_bytes >= MAX_BATCH_BYTES {
            self.flush_metadata()?;
        }
        Ok(())
    }

    /// Finish and sync both artifacts before their checksums can enter the manifest.
    pub fn finish(mut self) -> CacheResult<FinishedTables> {
        if self.sample_count == 0 {
            return Err(CacheError::InvalidInput(
                "cache source yielded no samples".to_string(),
            ));
        }
        let started = Instant::now();
        self.flush_metadata()?;
        let metadata_path = self.path.join("metadata.arrow");
        native_result("finish metadata Arrow writer", || {
            self.metadata.finish()?;
            self.metadata.get_mut().flush()?;
            self.metadata.get_ref().get_ref().sync_all()?;
            Ok(())
        })
        .map_err(|error| error.context(format!("finish '{}'", metadata_path.display())))?;
        let metadata_sha256 = hash_file(&metadata_path)?;
        let metadata_time = started.elapsed();

        let started = Instant::now();
        let index_path = self.path.join("index.bin");
        self.index
            .flush()
            .and_then(|()| self.index.get_ref().sync_all())
            .map_err(|error| {
                CacheError::from(error).context(format!("finish '{}'", index_path.display()))
            })?;
        let index_sha256 = hash_file(&index_path)?;
        Ok(FinishedTables {
            schema: self.schema,
            sample_count: self.sample_count,
            metadata_sha256,
            index_sha256,
            metadata_time,
            index_time: started.elapsed(),
        })
    }

    /// Serialize one bounded batch with explicit row counts for zero-column metadata.
    fn flush_metadata(&mut self) -> CacheResult<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        native_result("write metadata Arrow batch", || {
            let columns = self
                .schema
                .iter()
                .enumerate()
                .map(|(column, field)| build_array(column, &field.kind, &self.rows))
                .collect::<CacheResult<Vec<_>>>()?;
            let options = RecordBatchOptions::new().with_row_count(Some(self.rows.len()));
            let batch = RecordBatch::try_new_with_options(
                self.metadata.schema().clone(),
                columns,
                &options,
            )?;
            self.metadata.write(&batch)?;
            Ok(())
        })
        .map_err(|error| {
            error.context(format!(
                "write '{}'",
                self.path.join("metadata.arrow").display()
            ))
        })?;
        self.rows.clear();
        self.row_bytes = 0;
        Ok(())
    }
}

/// Include primitive cell storage and string allocations in the ingestion budget.
fn metadata_row_bytes(row: &[MetadataValue]) -> CacheResult<usize> {
    let mut bytes = std::mem::size_of_val(row)
        .checked_add(std::mem::size_of::<Vec<MetadataValue>>())
        .ok_or_else(|| {
            CacheError::InvalidInput("metadata row byte length overflowed usize".to_string())
        })?;
    for value in row {
        if let MetadataValue::String(value) = value {
            if value.len() > i32::MAX as usize {
                return Err(CacheError::InvalidInput(
                    "metadata string exceeds Arrow UTF-8 offset limit".to_string(),
                ));
            }
            bytes = bytes.checked_add(value.capacity()).ok_or_else(|| {
                CacheError::InvalidInput("metadata row byte length overflowed usize".to_string())
            })?;
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::index::MappedIndex;
    use crate::test_support::TestDirectory;

    /// Persist multiple batches while index bytes grow and retained metadata stays bounded.
    #[test]
    fn streams_rows_and_index_before_finish() -> CacheResult<()> {
        let directory = TestDirectory::new()?;
        let metadata =
            BTreeMap::from([("text".to_string(), MetadataValue::String("x".repeat(4096)))]);
        let mut writer = CacheTablesWriter::create(directory.path(), &metadata)?;
        let total = 10_000_u64;
        for sample in 0..total {
            writer.push(
                IndexEntry {
                    shard_id: 0,
                    offset: sample * 24,
                    byte_len: 24,
                },
                writer.metadata_row(metadata.clone())?,
            )?;
            assert!(writer.rows.len() <= MAX_BATCH_ROWS);
            assert!(writer.row_bytes <= MAX_BATCH_BYTES);
        }
        assert!(
            std::fs::metadata(directory.path().join("metadata.arrow"))?.len()
                > MAX_BATCH_BYTES as u64
        );
        assert!(std::fs::metadata(directory.path().join("index.bin"))?.len() > 24 * 9_000);
        let finished = writer.finish()?;
        assert_eq!(finished.sample_count, total);
        let index = MappedIndex::open(&directory.path().join("index.bin"))?;
        assert_eq!(index.len(), total as usize);
        assert_eq!(index.get(9_999)?.offset, 9_999 * 24);
        let reader = arrow_ipc::reader::FileReader::try_new(
            File::open(directory.path().join("metadata.arrow"))?,
            None,
        )?;
        let mut rows = 0;
        for batch in reader {
            let batch = batch?;
            assert!(batch.num_rows() <= MAX_BATCH_ROWS);
            rows += batch.num_rows();
        }
        assert_eq!(rows, total as usize);
        Ok(())
    }

    /// A single oversized row is written alone without expanding the retained batch.
    #[test]
    fn oversized_row_does_not_accumulate_with_other_rows() -> CacheResult<()> {
        let directory = TestDirectory::new()?;
        let metadata = BTreeMap::from([("text".to_string(), MetadataValue::String(String::new()))]);
        let mut writer = CacheTablesWriter::create(directory.path(), &metadata)?;
        let entry = IndexEntry {
            shard_id: 0,
            offset: 0,
            byte_len: 1,
        };
        writer.push(
            entry.clone(),
            vec![MetadataValue::String("small".to_string())],
        )?;
        writer.push(
            entry,
            vec![MetadataValue::String("x".repeat(MAX_BATCH_BYTES + 1))],
        )?;
        assert!(writer.rows.is_empty());
        assert_eq!(writer.row_bytes, 0);
        writer.finish()?;
        let reader = arrow_ipc::reader::FileReader::try_new(
            File::open(directory.path().join("metadata.arrow"))?,
            None,
        )?;
        assert_eq!(
            reader
                .map(|batch| batch.map(|batch| batch.num_rows()))
                .collect::<Result<Vec<_>, _>>()?,
            vec![1, 1]
        );
        Ok(())
    }

    /// Empty schemas retain physical row counts across more than one Arrow batch.
    #[test]
    fn empty_schema_preserves_rows_and_rejects_drift() -> CacheResult<()> {
        let directory = TestDirectory::new()?;
        let mut writer = CacheTablesWriter::create(directory.path(), &BTreeMap::new())?;
        assert!(writer
            .metadata_row(BTreeMap::from([(
                "late".to_string(),
                MetadataValue::Int(1)
            )]))
            .is_err());
        for sample in 0..MAX_BATCH_ROWS + 1 {
            writer.push(
                IndexEntry {
                    shard_id: 0,
                    offset: sample as u64,
                    byte_len: 1,
                },
                Vec::new(),
            )?;
        }
        writer.finish()?;
        let reader = arrow_ipc::reader::FileReader::try_new(
            File::open(directory.path().join("metadata.arrow"))?,
            None,
        )?;
        assert_eq!(
            reader
                .map(|batch| batch.map(|batch| batch.num_rows()))
                .collect::<Result<Vec<_>, _>>()?,
            vec![MAX_BATCH_ROWS, 1]
        );
        Ok(())
    }
}
