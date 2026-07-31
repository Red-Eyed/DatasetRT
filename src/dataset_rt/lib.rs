#![deny(clippy::expect_used)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::unwrap_used)]

mod compression;
mod dataset;
mod dataset_runtime;
mod runtime;
mod samples_metadata;
mod sampling;
mod storage;
mod types;
mod worker_pool;
mod writer;

use dataset::PyCachedDataset;
use dataset_runtime::PyDatasetRuntime;
use pyo3::prelude::*;

#[pyfunction]
fn write_cache(
    runtime: PyRef<'_, PyDatasetRuntime>,
    sources: Bound<'_, PyAny>,
    base_cache_dir: String,
    writer_config: Bound<'_, PyAny>,
    reuse_existing: bool,
) -> PyResult<Vec<(String, String, String)>> {
    writer::write_cache(
        runtime.pool(),
        runtime.num_workers(),
        sources,
        base_cache_dir,
        writer_config,
        reuse_existing,
    )
    .map_err(types::CacheError::into_py_err)
}

#[pymodule]
fn _dataset_rt(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(write_cache, module)?)?;
    module.add_class::<PyDatasetRuntime>()?;
    module.add_class::<PyCachedDataset>()?;
    Ok(())
}
