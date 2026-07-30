use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, UInt32Array,
    UInt64Array,
};
use arrow_ipc::reader::FileReader;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyDict};

use crate::runtime::{EpochPlan, RuntimeIterator};
use crate::sampling::{plan_epoch, validate_weights};
use crate::storage::{load_cache, LoadedCache};
use crate::types::{
    CacheError, CacheResult, MetadataField, MetadataValue, NumWorkers, PrefetchSize,
};

#[pyclass(name = "CachedDataset")]
pub struct PyCachedDataset {
    inner: Arc<DatasetState>,
}

struct DatasetState {
    caches: Arc<Vec<LoadedCache>>,
    cache_offsets: Arc<Vec<usize>>,
    total_samples: usize,
    schema: Vec<MetadataField>,
    seed: u64,
    prefetch_size: PrefetchSize,
    num_workers: NumWorkers,
    shuffle: bool,
    mutable: Mutex<MutableDatasetState>,
}

struct MutableDatasetState {
    weights: WeightState,
    next_epoch: u64,
}

enum WeightState {
    Uniform,
    Custom(Vec<f64>),
}

#[pymethods]
impl PyCachedDataset {
    #[new]
    fn new(
        paths: Vec<String>,
        seed: u64,
        prefetch_size: usize,
        num_workers: usize,
        shuffle: bool,
        validate_cache: bool,
    ) -> PyResult<Self> {
        DatasetState::load(
            paths,
            seed,
            prefetch_size,
            num_workers,
            shuffle,
            validate_cache,
        )
        .map(|state| Self {
            inner: Arc::new(state),
        })
        .map_err(CacheError::into_py_err)
    }

    fn __len__(&self) -> usize {
        self.inner.total_samples
    }

    fn __iter__(&self) -> PyResult<PyDatasetIterator> {
        let iterator = self
            .inner
            .start_iterator()
            .map_err(CacheError::into_py_err)?;
        Ok(PyDatasetIterator {
            schema: self.inner.schema.clone(),
            runtime: Mutex::new(iterator),
        })
    }

    fn has_custom_weights(&self) -> PyResult<bool> {
        let guard = self
            .inner
            .mutable
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        Ok(matches!(guard.weights, WeightState::Custom(_)))
    }

    fn get_weights(&self) -> PyResult<Vec<f64>> {
        let guard = self
            .inner
            .mutable
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        self.inner
            .weight_values(&guard.weights)
            .map_err(CacheError::into_py_err)
    }

    fn set_weight_table_ipc(&self, ipc: Bound<'_, PyBytes>) -> PyResult<()> {
        let weights = self
            .inner
            .extract_weight_table_ipc(ipc.as_bytes())
            .map_err(CacheError::into_py_err)?;
        let mut guard = self
            .inner
            .mutable
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        guard.weights = WeightState::Custom(weights);
        Ok(())
    }
}

impl DatasetState {
    fn load(
        paths: Vec<String>,
        seed: u64,
        prefetch_size: usize,
        num_workers: usize,
        shuffle: bool,
        validate_cache: bool,
    ) -> CacheResult<Self> {
        if paths.is_empty() {
            return Err(CacheError::InvalidInput(
                "CachedDataset requires at least one cache path".to_string(),
            ));
        }

        let caches = load_caches(paths, validate_cache)?;
        let schema = common_schema(&caches)?;
        let (cache_offsets, total_samples) = collect_cache_offsets(&caches)?;
        if total_samples == 0 {
            return Err(CacheError::InvalidCache(
                "dataset contains no physical samples".to_string(),
            ));
        }

        Ok(Self {
            caches: Arc::new(caches),
            cache_offsets: Arc::new(cache_offsets),
            total_samples,
            schema,
            seed,
            prefetch_size: PrefetchSize::new(prefetch_size)?,
            num_workers: NumWorkers::new(num_workers)?,
            shuffle,
            mutable: Mutex::new(MutableDatasetState {
                weights: WeightState::Uniform,
                next_epoch: 0,
            }),
        })
    }

    fn start_iterator(&self) -> CacheResult<RuntimeIterator> {
        let plan = if self.shuffle {
            EpochPlan::Planned(self.shuffled_plan()?)
        } else {
            EpochPlan::PhysicalOrder {
                len: self.total_samples,
            }
        };
        Ok(RuntimeIterator::start(
            self.caches.clone(),
            self.cache_offsets.clone(),
            plan,
            self.prefetch_size,
            self.num_workers,
        ))
    }

    fn shuffled_plan(&self) -> CacheResult<Vec<usize>> {
        let (weights, epoch) = {
            let mut guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
            let snapshot = self.weight_values(&guard.weights)?;
            let epoch = guard.next_epoch;
            guard.next_epoch += 1;
            (snapshot, epoch)
        };
        plan_epoch(&weights, self.seed, epoch)
    }

    fn weight_values(&self, weights: &WeightState) -> CacheResult<Vec<f64>> {
        match weights {
            WeightState::Uniform => Ok(vec![1.0; self.total_samples]),
            WeightState::Custom(values) => {
                validate_weights(values, self.total_samples)?;
                Ok(values.clone())
            }
        }
    }

    fn extract_weight_table_ipc(&self, ipc: &[u8]) -> CacheResult<Vec<f64>> {
        let mut weights = vec![0.0; self.total_samples];
        let mut seen = vec![false; self.total_samples];
        let mut row_count = 0_usize;
        let reader = FileReader::try_new(Cursor::new(ipc), None)?;

        for batch in reader {
            let batch = batch?;
            row_count = row_count.checked_add(batch.num_rows()).ok_or_else(|| {
                CacheError::InvalidInput("weight table row count overflowed usize".to_string())
            })?;
            apply_weight_batch(
                &batch,
                &self.caches,
                &self.cache_offsets,
                &mut weights,
                &mut seen,
            )?;
        }

        validate_complete_weight_table(row_count, self.total_samples, &seen)?;
        validate_weights(&weights, self.total_samples)?;
        Ok(weights)
    }
}

struct WeightUpdate {
    cache_id: u64,
    sample_id: u64,
    weight: f64,
}

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

fn cache_offsets(caches: &[LoadedCache]) -> CacheResult<Vec<usize>> {
    let mut offsets = Vec::with_capacity(caches.len());
    let mut next_offset = 0_usize;
    for cache in caches {
        offsets.push(next_offset);
        next_offset = next_offset
            .checked_add(cache.sample_count())
            .ok_or_else(|| {
                CacheError::InvalidInput("total sample count overflowed usize".to_string())
            })?;
    }
    Ok(offsets)
}

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

fn required_column<'a>(batch: &'a RecordBatch, name: &str) -> CacheResult<&'a ArrayRef> {
    batch
        .column_by_name(name)
        .ok_or_else(|| CacheError::InvalidInput(format!("weight table missing '{name}' column")))
}

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

fn reject_null_cell(column: &ArrayRef, row_index: usize, name: &str) -> CacheResult<()> {
    if column.is_null(row_index) {
        return Err(CacheError::InvalidInput(format!(
            "weight table column '{name}' contains null"
        )));
    }
    Ok(())
}

fn non_negative_i64_as_u64(value: i64, name: &str) -> CacheResult<u64> {
    u64::try_from(value).map_err(|_| {
        CacheError::InvalidInput(format!("weight table column '{name}' must be non-negative"))
    })
}

type PySampleTuple<'py> = (Bound<'py, PyBytes>, Bound<'py, PyDict>, u64, u64);

#[pyclass]
pub struct PyDatasetIterator {
    schema: Vec<MetadataField>,
    runtime: Mutex<RuntimeIterator>,
}

#[pymethods]
impl PyDatasetIterator {
    fn __iter__(self_: PyRef<'_, Self>) -> PyRef<'_, Self> {
        self_
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<PySampleTuple<'py>>> {
        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        match guard.next() {
            Some(Ok(sample)) => sample_to_python(py, &self.schema, sample).map(Some),
            Some(Err(error)) => Err(error.into_py_err()),
            None => Ok(None),
        }
    }
}

fn load_caches(paths: Vec<String>, validate_cache: bool) -> CacheResult<Vec<LoadedCache>> {
    paths
        .into_iter()
        .map(|path| load_cache(PathBuf::from(path), validate_cache))
        .collect()
}

fn common_schema(caches: &[LoadedCache]) -> CacheResult<Vec<MetadataField>> {
    let first = caches
        .first()
        .ok_or_else(|| CacheError::InvalidInput("no caches provided".to_string()))?;
    for cache in caches.iter().skip(1) {
        if cache.manifest.metadata_schema.len() != first.manifest.metadata_schema.len() {
            return Err(CacheError::InvalidCache(
                "all caches must share the same metadata schema".to_string(),
            ));
        }
        for (left, right) in cache
            .manifest
            .metadata_schema
            .iter()
            .zip(first.manifest.metadata_schema.iter())
        {
            if left.name != right.name || left.kind != right.kind {
                return Err(CacheError::InvalidCache(
                    "all caches must share the same metadata schema".to_string(),
                ));
            }
        }
    }
    Ok(first.manifest.metadata_schema.clone())
}

fn collect_cache_offsets(caches: &[LoadedCache]) -> CacheResult<(Vec<usize>, usize)> {
    let offsets = cache_offsets(caches)?;
    let total = caches.iter().try_fold(0_usize, |total, cache| {
        total.checked_add(cache.sample_count()).ok_or_else(|| {
            CacheError::InvalidInput("total sample count overflowed usize".to_string())
        })
    })?;
    Ok((offsets, total))
}

fn sample_to_python<'py>(
    py: Python<'py>,
    schema: &[MetadataField],
    sample: crate::types::LoadedSample,
) -> PyResult<PySampleTuple<'py>> {
    let metadata = metadata_to_dict(py, schema, &sample.metadata)?;
    Ok((
        PyBytes::new(py, &sample.data),
        metadata,
        sample.cache_id.as_u64(),
        sample.sample_id.as_u64(),
    ))
}

fn metadata_to_dict<'py>(
    py: Python<'py>,
    schema: &[MetadataField],
    metadata: &[MetadataValue],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (field, value) in schema.iter().zip(metadata.iter()) {
        match value {
            MetadataValue::Bool(value) => dict.set_item(&field.name, *value)?,
            MetadataValue::Int(value) => dict.set_item(&field.name, *value)?,
            MetadataValue::Float(value) => dict.set_item(&field.name, *value)?,
            MetadataValue::String(value) => dict.set_item(&field.name, value)?,
        }
    }
    Ok(dict)
}
