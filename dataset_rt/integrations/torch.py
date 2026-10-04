"""Lazy legacy PyTorch view; multiprocessing loading is not enabled here."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dataset_rt.dataset import CachedDataset
    from dataset_rt.records import CachedSample, SizedTorchIterableDataset


def to_torch_iterable_dataset(dataset: CachedDataset) -> SizedTorchIterableDataset:
    """Wrap the existing dataset, preserving the legacy zero-worker contract."""
    try:
        torch_data = import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError(
            "CachedDataset.to_torch_iterable_dataset requires PyTorch to be installed"
        ) from error

    iterable_dataset = cast("type[object]", torch_data.IterableDataset)

    class DatasetRTTorchIterableDataset(iterable_dataset):
        """Sized PyTorch iterable view over a `CachedDataset`."""

        def __iter__(self) -> Iterator[CachedSample]:
            """Use the original dataset only in the calling process."""
            if torch_data.get_worker_info() is not None:
                raise RuntimeError(
                    "Use Dataloader with num workers = 0. "
                    "Use DatasetRuntime(num_workers=N) for parallel cache reads."
                )
            return iter(dataset)

        def __len__(self) -> int:
            """Expose the original dataset's current epoch length."""
            return len(dataset)

    return DatasetRTTorchIterableDataset()
