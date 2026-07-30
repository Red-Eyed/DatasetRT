use std::io::Cursor;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, UInt32Array,
    UInt64Array,
};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

use crate::sampling::validate_weights;
use crate::storage::LoadedCache;
use crate::types::{CacheError, CacheResult, MetadataField, MetadataKind};

/// Sampling weight storage that avoids materializing uniform all-one vectors.
pub enum WeightState {
    Uniform,
    Custom(Vec<f64>),
}

struct WeightUpdate {
    cache_id: u64,
    sample_id: u64,
    weight: f64,
}

enum WeightSlice<'a> {
    Uniform { len: usize },
    Custom(&'a [f64]),
}

/// Report whether a weight state contains caller-supplied custom weights.
pub fn has_custom_weights(weights: &WeightState) -> bool {
    matches!(weights, WeightState::Custom(_))
}

/// Materialize weights only for API paths that explicitly need a full vector.
pub fn weight_values(weights: &WeightState, total_samples: usize) -> CacheResult<Vec<f64>> {
    match weights {
        WeightState::Uniform => Ok(vec![1.0; total_samples]),
        WeightState::Custom(values) => {
            validate_weights(values, total_samples)?;
            Ok(values.clone())
        }
    }
}

/// Build one Arrow IPC table containing identity, metadata, and current weights.
pub fn build_weight_table_ipc(
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    schema: &[MetadataField],
    weights: &WeightState,
) -> CacheResult<Vec<u8>> {
    let arrow_schema = weight_table_schema(schema);
    let batches = weight_table_batches(caches, cache_offsets, schema, &arrow_schema, weights)?;
    write_record_batches_ipc(&arrow_schema, &batches)
}

/// Parse a columnar weight update and validate that it covers the dataset exactly once.
pub fn extract_weight_table_ipc(
    ipc: &[u8],
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    total_samples: usize,
) -> CacheResult<Vec<f64>> {
    let mut weights = vec![0.0; total_samples];
    let mut seen = vec![false; total_samples];
    let mut row_count = 0_usize;
    let reader = FileReader::try_new(Cursor::new(ipc), None)?;

    for batch in reader {
        let batch = batch?;
        row_count = row_count.checked_add(batch.num_rows()).ok_or_else(|| {
            CacheError::InvalidInput("weight table row count overflowed usize".to_string())
        })?;
        apply_weight_batch(&batch, caches, cache_offsets, &mut weights, &mut seen)?;
    }

    validate_complete_weight_table(row_count, total_samples, &seen)?;
    validate_weights(&weights, total_samples)?;
    Ok(weights)
}

/// Build Arrow batches for every cache without expanding metadata cells into Rust objects.
fn weight_table_batches(
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    metadata_schema: &[MetadataField],
    arrow_schema: &Arc<Schema>,
    weights: &WeightState,
) -> CacheResult<Vec<RecordBatch>> {
    let mut output_batches = Vec::new();
    for (cache_index, cache) in caches.iter().enumerate() {
        let cache_offset = cache_offset(cache_offsets, cache_index)?;
        let mut sample_start = 0_usize;
        let reader = cache.metadata_batches()?;
        for batch in reader {
            let metadata_batch = batch?;
            let row_count = metadata_batch.num_rows();
            let physical_offset = physical_offset(cache_offset, sample_start)?;
            let weights = weight_slice(weights, physical_offset, row_count)?;
            output_batches.push(weight_table_batch(
                cache_index,
                sample_start,
                arrow_schema,
                metadata_schema,
                &metadata_batch,
                weights,
            )?);
            sample_start = sample_start.checked_add(row_count).ok_or_else(|| {
                CacheError::InvalidInput("metadata row count overflowed usize".to_string())
            })?;
        }
        validate_cache_metadata_rows(cache_index, cache.sample_count(), sample_start)?;
    }
    Ok(output_batches)
}

/// Return the stable public weight table schema used by Rust IPC exports.
fn weight_table_schema(schema: &[MetadataField]) -> Arc<Schema> {
    Arc::new(Schema::new(weight_table_fields(schema)))
}

/// Build the public weight-table fields in stable column order.
fn weight_table_fields(schema: &[MetadataField]) -> Vec<Field> {
    let mut fields = Vec::with_capacity(schema.len() + 3);
    fields.push(Field::new("cache_id", DataType::UInt64, false));
    fields.push(Field::new("sample_id", DataType::UInt64, false));
    fields.extend(
        schema
            .iter()
            .map(|field| Field::new(&field.name, arrow_type(&field.kind), false)),
    );
    fields.push(Field::new("weight", DataType::Float64, false));
    fields
}

/// Map DatasetRT metadata kinds onto Arrow types used by Polars.
fn arrow_type(kind: &MetadataKind) -> DataType {
    match kind {
        MetadataKind::Bool => DataType::Boolean,
        MetadataKind::Int => DataType::Int64,
        MetadataKind::Float => DataType::Float64,
        MetadataKind::String => DataType::Utf8,
    }
}

/// Return the first physical sample offset for one cache.
fn cache_offset(cache_offsets: &[usize], cache_index: usize) -> CacheResult<usize> {
    cache_offsets
        .get(cache_index)
        .copied()
        .ok_or_else(|| CacheError::InvalidCache("cache offset is out of range".to_string()))
}

/// Add a cache-local batch offset to a cache's physical base offset.
fn physical_offset(cache_offset: usize, sample_start: usize) -> CacheResult<usize> {
    cache_offset.checked_add(sample_start).ok_or_else(|| {
        CacheError::InvalidInput("physical metadata offset overflowed usize".to_string())
    })
}

/// Ensure `metadata.arrow` produced exactly the manifest/index row count for one cache.
fn validate_cache_metadata_rows(
    cache_index: usize,
    expected_rows: usize,
    actual_rows: usize,
) -> CacheResult<()> {
    if actual_rows != expected_rows {
        return Err(CacheError::InvalidCache(format!(
            "metadata row count for cache {cache_index} expected {expected_rows}, got {actual_rows}"
        )));
    }
    Ok(())
}

/// Borrow the weight range for one cache without materializing uniform weights.
fn weight_slice(weights: &WeightState, offset: usize, len: usize) -> CacheResult<WeightSlice<'_>> {
    match weights {
        WeightState::Uniform => Ok(WeightSlice::Uniform { len }),
        WeightState::Custom(values) => values
            .get(
                offset..offset.checked_add(len).ok_or_else(|| {
                    CacheError::InvalidInput("weight slice offset overflowed usize".to_string())
                })?,
            )
            .map(WeightSlice::Custom)
            .ok_or_else(|| CacheError::InvalidInput("weight slice is out of range".to_string())),
    }
}

/// Build one output batch by reusing metadata arrays and adding identity plus weight columns.
fn weight_table_batch(
    cache_index: usize,
    sample_start: usize,
    arrow_schema: &Arc<Schema>,
    metadata_schema: &[MetadataField],
    metadata_batch: &RecordBatch,
    weights: WeightSlice<'_>,
) -> CacheResult<RecordBatch> {
    let row_count = metadata_batch.num_rows();
    let cache_id = u64::try_from(cache_index)
        .map_err(|_| CacheError::InvalidInput("cache_id does not fit in u64".to_string()))?;
    if row_count != weight_len(&weights) {
        return Err(CacheError::InvalidCache(
            "metadata and weight row counts diverged".to_string(),
        ));
    }

    let mut columns = Vec::with_capacity(metadata_schema.len() + 3);
    columns.push(Arc::new(UInt64Array::from(vec![cache_id; row_count])) as ArrayRef);
    columns.push(Arc::new(sample_id_array(sample_start, row_count)?) as ArrayRef);
    columns.extend(metadata_columns(metadata_batch, metadata_schema)?);
    columns.push(Arc::new(weight_array(weights)) as ArrayRef);

    RecordBatch::try_new(arrow_schema.clone(), columns).map_err(CacheError::from)
}

/// Build the sample_id column for one metadata batch within a cache.
fn sample_id_array(sample_start: usize, row_count: usize) -> CacheResult<UInt64Array> {
    let sample_end = sample_start
        .checked_add(row_count)
        .ok_or_else(|| CacheError::InvalidInput("sample_id range overflowed usize".to_string()))?;
    let values = (sample_start..sample_end)
        .map(|sample_index| {
            u64::try_from(sample_index)
                .map_err(|_| CacheError::InvalidInput("sample_id does not fit in u64".to_string()))
        })
        .collect::<CacheResult<Vec<_>>>()?;
    Ok(UInt64Array::from(values))
}

/// Reuse metadata columns after verifying they match the manifest schema.
fn metadata_columns(
    metadata_batch: &RecordBatch,
    metadata_schema: &[MetadataField],
) -> CacheResult<Vec<ArrayRef>> {
    if metadata_batch.num_columns() != metadata_schema.len() {
        return Err(CacheError::InvalidCache(
            "metadata batch column count does not match manifest".to_string(),
        ));
    }
    metadata_schema
        .iter()
        .enumerate()
        .map(|(column_index, field)| metadata_column(metadata_batch, column_index, field))
        .collect()
}

/// Return one metadata array if its Arrow field matches the expected manifest field.
fn metadata_column(
    metadata_batch: &RecordBatch,
    column_index: usize,
    expected: &MetadataField,
) -> CacheResult<ArrayRef> {
    let schema = metadata_batch.schema();
    let actual = schema
        .fields()
        .get(column_index)
        .ok_or_else(|| CacheError::InvalidCache("metadata field is out of range".to_string()))?;
    let expected_type = arrow_type(&expected.kind);
    if actual.name() != &expected.name || actual.data_type() != &expected_type {
        return Err(CacheError::InvalidCache(format!(
            "metadata column '{}' does not match manifest schema",
            expected.name
        )));
    }
    metadata_batch
        .columns()
        .get(column_index)
        .cloned()
        .ok_or_else(|| CacheError::InvalidCache("metadata column is out of range".to_string()))
}

/// Build the current weight column for one cache.
fn weight_array(weights: WeightSlice<'_>) -> Float64Array {
    match weights {
        WeightSlice::Uniform { len } => Float64Array::from(vec![1.0; len]),
        WeightSlice::Custom(values) => Float64Array::from(values.to_vec()),
    }
}

/// Return the number of rows represented by a weight slice.
fn weight_len(weights: &WeightSlice<'_>) -> usize {
    match weights {
        WeightSlice::Uniform { len } => *len,
        WeightSlice::Custom(values) => values.len(),
    }
}

/// Serialize Arrow record batches sharing one schema to an IPC file payload.
fn write_record_batches_ipc(schema: &Arc<Schema>, batches: &[RecordBatch]) -> CacheResult<Vec<u8>> {
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, schema)?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.finish()?;
    }
    Ok(bytes)
}

/// Apply one Arrow record batch of weight updates without converting rows through Python.
fn apply_weight_batch(
    batch: &RecordBatch,
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    weights: &mut [f64],
    seen: &mut [bool],
) -> CacheResult<()> {
    let cache_ids = required_column(batch, "cache_id")?;
    let sample_ids = required_column(batch, "sample_id")?;
    let weight_values = required_column(batch, "weight")?;

    for row_index in 0..batch.num_rows() {
        let update = WeightUpdate {
            cache_id: read_u64_cell(cache_ids, row_index, "cache_id")?,
            sample_id: read_u64_cell(sample_ids, row_index, "sample_id")?,
            weight: read_f64_cell(weight_values, row_index, "weight")?,
        };
        apply_weight_update(update, caches, cache_offsets, weights, seen)?;
    }

    Ok(())
}

/// Store one validated weight update at its physical offset and reject duplicate identities.
fn apply_weight_update(
    update: WeightUpdate,
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    weights: &mut [f64],
    seen: &mut [bool],
) -> CacheResult<()> {
    let physical_index = physical_index(caches, cache_offsets, update.cache_id, update.sample_id)?;
    let already_seen = seen
        .get(physical_index)
        .copied()
        .ok_or_else(|| CacheError::InvalidCache("weight seen index is out of range".to_string()))?;
    if already_seen {
        return Err(CacheError::InvalidInput(format!(
            "duplicate weight row for cache_id={} sample_id={}",
            update.cache_id, update.sample_id
        )));
    }
    let seen_slot = seen
        .get_mut(physical_index)
        .ok_or_else(|| CacheError::InvalidCache("weight seen index is out of range".to_string()))?;
    *seen_slot = true;
    let weight_slot = weights
        .get_mut(physical_index)
        .ok_or_else(|| CacheError::InvalidCache("weight index is out of range".to_string()))?;
    *weight_slot = update.weight;
    Ok(())
}

/// Convert a `(cache_id, sample_id)` identity into the compact physical index space.
fn physical_index(
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    cache_id: u64,
    sample_id: u64,
) -> CacheResult<usize> {
    let cache_index = usize::try_from(cache_id)
        .map_err(|_| CacheError::InvalidInput("cache_id does not fit in usize".to_string()))?;
    let sample_index = usize::try_from(sample_id)
        .map_err(|_| CacheError::InvalidInput("sample_id does not fit in usize".to_string()))?;
    let cache = caches.get(cache_index).ok_or_else(|| {
        CacheError::InvalidInput(format!(
            "unknown weight row identity cache_id={cache_id} sample_id={sample_id}"
        ))
    })?;
    if sample_index >= cache.sample_count() {
        return Err(CacheError::InvalidInput(format!(
            "unknown weight row identity cache_id={cache_id} sample_id={sample_id}"
        )));
    }
    let offset = cache_offsets
        .get(cache_index)
        .copied()
        .ok_or_else(|| CacheError::InvalidCache("cache offset is out of range".to_string()))?;
    offset.checked_add(sample_index).ok_or_else(|| {
        CacheError::InvalidInput("physical weight index overflowed usize".to_string())
    })
}

/// Ensure a weight table has exactly one accepted row for every physical sample.
fn validate_complete_weight_table(
    row_count: usize,
    expected_rows: usize,
    seen: &[bool],
) -> CacheResult<()> {
    if row_count != expected_rows {
        return Err(CacheError::InvalidInput(format!(
            "expected {expected_rows} weight rows, got {row_count}"
        )));
    }
    if seen.iter().any(|value| !value) {
        return Err(CacheError::InvalidInput(
            "weight table must include every physical sample exactly once".to_string(),
        ));
    }
    Ok(())
}

/// Fetch a required Arrow column by name with a user-facing validation error.
fn required_column<'a>(batch: &'a RecordBatch, name: &str) -> CacheResult<&'a ArrayRef> {
    batch
        .column_by_name(name)
        .ok_or_else(|| CacheError::InvalidInput(format!("weight table missing '{name}' column")))
}

/// Read a non-null integer identity cell from an Arrow column as `u64`.
fn read_u64_cell(column: &ArrayRef, row_index: usize, name: &str) -> CacheResult<u64> {
    reject_null_cell(column, row_index, name)?;
    if let Some(values) = column.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row_index));
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt32Array>() {
        return Ok(u64::from(values.value(row_index)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
        return non_negative_i64_as_u64(values.value(row_index), name);
    }
    if let Some(values) = column.as_any().downcast_ref::<Int32Array>() {
        return non_negative_i64_as_u64(i64::from(values.value(row_index)), name);
    }
    Err(CacheError::InvalidInput(format!(
        "weight table column '{name}' must be an integer"
    )))
}

/// Read a non-null numeric weight cell from an Arrow column as `f64`.
fn read_f64_cell(column: &ArrayRef, row_index: usize, name: &str) -> CacheResult<f64> {
    reject_null_cell(column, row_index, name)?;
    if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row_index));
    }
    if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
        return Ok(f64::from(values.value(row_index)));
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row_index) as f64);
    }
    if let Some(values) = column.as_any().downcast_ref::<UInt32Array>() {
        return Ok(f64::from(values.value(row_index)));
    }
    if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row_index) as f64);
    }
    if let Some(values) = column.as_any().downcast_ref::<Int32Array>() {
        return Ok(f64::from(values.value(row_index)));
    }
    Err(CacheError::InvalidInput(format!(
        "weight table column '{name}' must be numeric"
    )))
}

/// Reject null identity or weight cells before type-specific extraction.
fn reject_null_cell(column: &ArrayRef, row_index: usize, name: &str) -> CacheResult<()> {
    if column.is_null(row_index) {
        return Err(CacheError::InvalidInput(format!(
            "weight table column '{name}' contains null"
        )));
    }
    Ok(())
}

/// Convert signed identity values only when they are representable cache/sample IDs.
fn non_negative_i64_as_u64(value: i64, name: &str) -> CacheResult<u64> {
    u64::try_from(value).map_err(|_| {
        CacheError::InvalidInput(format!("weight table column '{name}' must be non-negative"))
    })
}
