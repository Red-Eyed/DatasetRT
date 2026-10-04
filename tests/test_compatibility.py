"""Freeze the 0.3.2 public facade before moving its implementation modules."""

from __future__ import annotations

import inspect
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import dataset_rt
import dataset_rt.api as api
from dataset_rt import config, dataset, records, runtime

EXPORTS = (
    "CacheInput",
    "CacheSource",
    "CacheSourcesDatasetError",
    "CacheSourcesDatasetResult",
    "CacheSourcesDatasetSuccess",
    "CacheWriteError",
    "CacheWriteResult",
    "CacheWriteSuccess",
    "CachedDataset",
    "CachedSample",
    "DatasetRuntime",
    "ReaderConfig",
    "ShardCompression",
    "SizedTorchIterableDataset",
    "WriterConfig",
    "WriterProfilerConfig",
)


def test_facade_preserves_existing_exports_and_identity() -> None:
    """Old imports must resolve to the same objects through both public paths."""
    assert set(EXPORTS) <= set(dataset_rt.__all__)
    for name in EXPORTS:
        assert getattr(dataset_rt, name) is getattr(api, name)


@pytest.mark.parametrize(
    ("implementation", "names"),
    [
        (config, ("ReaderConfig", "WriterConfig", "ShardCompression", "WriterProfilerConfig")),
        (dataset, ("CachedDataset",)),
        (runtime, ("DatasetRuntime",)),
        (records, ("CacheSource", "SizedTorchIterableDataset")),
    ],
)
def test_moved_classes_keep_historical_lookup_paths(
    implementation: object, names: tuple[str, ...]
) -> None:
    """Moving definitions must preserve public class identity and old pickle lookups."""
    for name in names:
        public_class = getattr(api, name)
        assert getattr(implementation, name) is public_class
        assert public_class.__module__ == "dataset_rt.api"
        assert pickle.loads(pickle.dumps(public_class)) is public_class


@pytest.mark.parametrize(
    ("record", "fields"),
    [
        (api.CacheInput, ("data", "metadata")),
        (api.CachedSample, ("data", "metadata", "cache_id", "sample_id")),
        (api.CacheWriteSuccess, ("source_name", "path")),
        (api.CacheWriteError, ("source_name", "message")),
        (api.CacheSourcesDatasetSuccess, ("dataset", "results")),
        (api.CacheSourcesDatasetError, ("results", "message")),
    ],
)
def test_record_lookup_paths_and_fields(record: type[tuple], fields: tuple[str, ...]) -> None:
    """Pickled record classes retain the historical facade lookup path."""
    assert record.__module__ == "dataset_rt.api"
    assert record._fields == fields
    assert pickle.loads(pickle.dumps(record)) is record


@pytest.mark.parametrize(
    "value",
    [
        api.CacheInput(b"sample", {"index": 1}),
        api.CachedSample(b"sample", {"index": 1}, 0, 1),
        api.CacheWriteSuccess("source", Path("cache/source")),
        api.CacheWriteError("source", "failed"),
        api.CacheSourcesDatasetError([], "no caches"),
        api.ReaderConfig(seed=7),
        api.WriterConfig(),
        api.ShardCompression(),
        api.WriterProfilerConfig(),
    ],
)
def test_public_values_round_trip_through_pickle(value: object) -> None:
    """Serializable records/configuration preserve their exact public type."""
    restored = pickle.loads(pickle.dumps(value))
    assert type(restored) is type(value)
    assert restored == value


def test_existing_method_parameters_and_default_objects() -> None:
    """Additive APIs must leave existing call signatures and defaults intact."""
    assert str(inspect.signature(api.DatasetRuntime)) == "(*, num_workers: 'int') -> 'None'"
    for method in (api.DatasetRuntime.write_cache, api.DatasetRuntime.from_cache_sources):
        signature = inspect.signature(method)
        assert signature.parameters["writer_config"].default is api.DEFAULT_WRITER_CONFIG
    assert tuple(inspect.signature(api.CachedDataset.get_item).parameters) == (
        "self",
        "cache_id",
        "sample_id",
    )
    assert api.DEFAULT_SHARD_COMPRESSION == api.ShardCompression()
    assert api.DEFAULT_WRITER_PROFILER_CONFIG == api.WriterProfilerConfig()


def test_configs_remain_frozen() -> None:
    """Mutating validated settings must fail rather than change runtime policy."""
    reader = api.ReaderConfig(seed=7)
    with pytest.raises(ValidationError, match="frozen"):
        reader.__setattr__("seed", 8)


def test_fresh_import_does_not_require_torch() -> None:
    """An import blocker tests optionality even when Torch is installed locally."""
    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockTorch(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("Torch deliberately unavailable")
sys.meta_path.insert(0, BlockTorch())
import dataset_rt
import dataset_rt.api
assert "torch" not in sys.modules
assert dataset_rt.ReaderConfig(seed=1).seed == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
