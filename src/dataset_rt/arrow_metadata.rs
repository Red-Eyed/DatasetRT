use std::path::Path;
use std::sync::Arc;

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{Array, RecordBatch, RecordBatchReader, UInt64Array};
use arrow_schema::{ArrowError, SchemaRef};
use arrow_select::take::take_record_batch;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

use crate::types::{CacheError, CacheResult};

pub mod mapped;

/// Immutable Arrow buffers and an optional active-row view; owners outlive every reader.
#[derive(Clone)]
pub struct ArrowMetadata {
    pub schema: SchemaRef,
    pub batches: Arc<Vec<RecordBatch>>,
    offsets: Arc<Vec<usize>>,
    rows: usize,
    selection: Option<UInt64Array>,
}

impl ArrowMetadata {
    /// Retain validated Arrow batches without serializing or copying their column buffers.
    pub fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> CacheResult<Self> {
        let mut offsets = Vec::with_capacity(batches.len());
        let mut rows = 0_usize;
        for batch in &batches {
            offsets.push(rows);
            rows = rows.checked_add(batch.num_rows()).ok_or_else(|| {
                CacheError::InvalidInput("metadata row count overflowed usize".to_string())
            })?;
        }
        Ok(Self {
            schema,
            batches: Arc::new(batches),
            offsets: Arc::new(offsets),
            rows,
            selection: None,
        })
    }

    /// Select shared row positions without copying any metadata column.
    pub fn select(mut self, selection: UInt64Array) -> CacheResult<Self> {
        if selection.null_count() != 0 {
            return Err(CacheError::InvalidInput(
                "metadata selection contains nulls".to_string(),
            ));
        }
        for row in selection.values() {
            if *row >= self.rows as u64 {
                return Err(CacheError::InvalidInput(
                    "metadata selection is out of range".to_string(),
                ));
            }
        }
        self.selection = Some(selection);
        Ok(self)
    }

    /// Return the active population size, including deliberately repeated positions.
    pub fn len(&self) -> usize {
        self.selection.as_ref().map_or(self.rows, Array::len)
    }

    /// Locate one selected row with logarithmic lookup in the record-batch offsets.
    pub fn row(&self, position: usize) -> CacheResult<(&RecordBatch, usize)> {
        let (batch_index, row) = self.location(position)?;
        let batch = self
            .batches
            .get(batch_index)
            .ok_or(CacheError::WorkerFailed)?;
        Ok((batch, row))
    }

    /// Resolve a view position to a batch and local row without scanning physical samples.
    pub fn location(&self, position: usize) -> CacheResult<(usize, usize)> {
        if position >= self.len() {
            return Err(CacheError::InvalidInput(
                "metadata row is out of range".to_string(),
            ));
        }
        let row = match &self.selection {
            Some(selection) => usize::try_from(selection.value(position)).map_err(|_| {
                CacheError::InvalidInput("metadata row does not fit usize".to_string())
            })?,
            None => position,
        };
        let batch_index = self
            .offsets
            .partition_point(|offset| *offset <= row)
            .checked_sub(1)
            .ok_or_else(|| CacheError::InvalidInput("metadata batch is missing".to_string()))?;
        let offset = self
            .offsets
            .get(batch_index)
            .ok_or(CacheError::WorkerFailed)?;
        Ok((batch_index, row - offset))
    }

    /// Export buffers through the standard Arrow capsule with reference-counted ownership.
    pub fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let reader = self.reader();
        PyCapsule::new_with_value(
            py,
            FFI_ArrowArrayStream::new(Box::new(reader)),
            c"arrow_array_stream",
        )
    }

    /// Create an independent cursor whose arrays retain this snapshot's owners.
    pub fn reader(&self) -> impl RecordBatchReader + Send + use<> {
        MetadataReader {
            table: self.clone(),
            position: 0,
        }
    }

    /// Write a new explicit snapshot with bounded chunks and durable file contents.
    pub fn write_file(&self, path: &Path) -> CacheResult<()> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut writer = arrow_ipc::writer::FileWriter::try_new(file, &self.schema)?;
        for batch in self.reader() {
            writer.write(&batch?)?;
        }
        writer.finish()?;
        writer.get_ref().sync_all()?;
        Ok(())
    }
}

/// Accept one immutable uint64 selection buffer without copying mapped permutation positions.
pub fn import_selection(capsule: Bound<'_, PyCapsule>) -> PyResult<UInt64Array> {
    let selection = import_table(capsule)?;
    let batch = selection
        .batches
        .first()
        .filter(|_| selection.batches.len() == 1)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("selection must contain one Arrow batch")
        })?;
    batch
        .columns()
        .first()
        .filter(|_| batch.num_columns() == 1)
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .cloned()
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("selection must contain one uint64 column")
        })
}

/// Consume a fresh standard Arrow stream capsule; imported arrays retain producer owners.
pub fn import_table(capsule: Bound<'_, PyCapsule>) -> PyResult<ArrowMetadata> {
    let pointer = capsule.pointer_checked(Some(c"arrow_array_stream"))?;
    if !pointer.cast::<FFI_ArrowArrayStream>().is_aligned() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "unaligned Arrow stream capsule",
        ));
    }
    // The named Arrow capsule transfers a valid stream once; from_raw clears its release callback.
    // The capsule remains alive during the move, and Arrow retains each imported array's owner.
    let stream = unsafe { FFI_ArrowArrayStream::from_raw(pointer.cast().as_ptr()) };
    let reader = ArrowArrayStreamReader::try_new(stream)
        .map_err(CacheError::from)
        .map_err(CacheError::into_py_err)?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(CacheError::from)
        .map_err(CacheError::into_py_err)?;
    ArrowMetadata::new(schema, batches).map_err(CacheError::into_py_err)
}

struct MetadataReader {
    table: ArrowMetadata,
    position: usize,
}

impl Iterator for MetadataReader {
    type Item = Result<RecordBatch, ArrowError>;

    /// Share whole batches by default; only explicit selected-table exports gather columns.
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.table.len() {
            return None;
        }
        if self.table.selection.is_none() {
            let result = self.shared_batch();
            if result.is_err() {
                self.position = self.table.len();
            }
            return Some(
                result.map_err(|error| ArrowError::InvalidArgumentError(error.to_string())),
            );
        }
        let result = self.selected_batch();
        if result.is_err() {
            self.position = self.table.len();
        }
        Some(result.map_err(|error| ArrowError::InvalidArgumentError(error.to_string())))
    }
}

impl MetadataReader {
    /// Slice large producer batches into bounded exports while sharing their column buffers.
    fn shared_batch(&mut self) -> CacheResult<RecordBatch> {
        let (batch, row) = self.table.row(self.position)?;
        let len = (batch.num_rows() - row).min(crate::samples_metadata::METADATA_BATCH_ROWS);
        let output = batch.slice(row, len);
        self.position += len;
        Ok(output)
    }

    /// Gather a bounded run from one source batch while preserving selected row order.
    fn selected_batch(&mut self) -> CacheResult<RecordBatch> {
        let (batch, first_row) = self.table.row(self.position)?;
        let mut indices = vec![first_row as u64];
        let mut end = self.position + 1;
        while end < self.table.len() && indices.len() < crate::samples_metadata::METADATA_BATCH_ROWS
        {
            let (next_batch, row) = self.table.row(end)?;
            if !std::ptr::eq(batch, next_batch) {
                break;
            }
            indices.push(row as u64);
            end += 1;
        }
        let output = take_record_batch(batch, &UInt64Array::from(indices))?;
        self.position = end;
        Ok(output)
    }
}

impl RecordBatchReader for MetadataReader {
    /// Preserve the original schema even for an empty stream.
    fn schema(&self) -> SchemaRef {
        self.table.schema.clone()
    }
}
