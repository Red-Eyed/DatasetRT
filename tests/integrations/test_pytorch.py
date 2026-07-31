from __future__ import annotations

import io
from pathlib import Path

import pytest

from dataset_rt import CachedDataset, CacheInput, CacheWriteSuccess, ReaderConfig, write_cache

torch = pytest.importorskip("torch")


class TensorSource:
    name = "tensors"

    def __iter__(self):
        for index in range(4):
            yield CacheInput(
                tensor_to_bytes(torch.tensor([index, index + 1])),
                {"index": index, "split": "train"},
            )


def tensor_to_bytes(tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def tensor_from_bytes(data: bytes):
    buffer = io.BytesIO(data)
    try:
        return torch.load(buffer, weights_only=True)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer)


def test_dataset_rt_streams_into_pytorch_iterable_dataset(tmp_path: Path) -> None:
    results = write_cache(TensorSource(), tmp_path / "cache", num_workers=4)
    match results[0]:
        case CacheWriteSuccess(path=path):
            cache_paths = [path]
        case result:
            raise AssertionError(result)
    dataset = CachedDataset(
        cache_paths,
        num_workers=4,
        reader_config=ReaderConfig(seed=3, prefetch_size=2, shuffle=False),
    )
    torch_dataset = dataset.to_torch_iterable_dataset()
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=None)

    samples = list(loader)
    tensors = [tensor_from_bytes(sample.data) for sample in samples]
    indices = [sample.metadata["index"] for sample in samples]

    assert len(torch_dataset) == 4
    assert len(samples) == 4
    assert torch.equal(torch.stack(tensors), torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]))
    assert indices == [0, 1, 2, 3]
