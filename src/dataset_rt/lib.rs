#![deny(clippy::expect_used)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::unwrap_used)]

mod dataset;
mod runtime;
mod sampling;
mod storage;
mod types;
mod writer;

use dataset::PyCachedDataset;
use pyo3::prelude::*;

#[pyfunction]
fn write_cache(
    sources: Bound<'_, PyAny>,
    base_cache_dir: String,
    max_shard_bytes: u64,
    prefetch_size: usize,
    num_threads: usize,
    shard_compression: Bound<'_, PyAny>,
    reuse_existing: bool,
) -> PyResult<Vec<String>> {
    writer::write_cache(
        sources,
        base_cache_dir,
        max_shard_bytes,
        prefetch_size,
        num_threads,
        shard_compression,
        reuse_existing,
    )
    .map_err(types::CacheError::into_py_err)
}

#[pymodule]
fn _dataset_rt(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(write_cache, module)?)?;
    module.add_class::<PyCachedDataset>()?;
    Ok(())
}
