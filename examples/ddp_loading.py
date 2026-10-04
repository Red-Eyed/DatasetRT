"""Self-contained Mac CPU DDP example; the application owns training step policy."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict
from torch.nn.parallel import DistributedDataParallel

from dataset_rt import (
    CachedSample,
    CacheInput,
    CacheWriteSuccess,
    DatasetRuntime,
    ReaderConfig,
    WriterConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class Source:
    """Generate a bounded fixture cache so the example needs no manual preparation."""

    name = "training"

    def __iter__(self) -> Iterator[CacheInput]:
        """Serialize payloads before DatasetRT's native write pipeline."""
        for index in range(48):
            yield CacheInput(str(index).encode(), {"index": index})


def transform(sample: CachedSample) -> torch.Tensor:
    """Decode a tiny domain input only after successful native sample delivery."""
    return torch.tensor([int(sample.data) / 48])


class Example(BaseSettings):
    """Expose application topology separately from native reader thread count."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    ranks: int = Field(default=2, ge=2, le=4)
    workers: int = Field(default=2, ge=0, le=2)
    worker_context: Literal["fork", "spawn", "forkserver"] = "forkserver"
    gloo_interface: str = "lo0"
    steps: int = Field(default=4, gt=0)
    shuffle: CliImplicitFlag[bool] = True
    quiet: CliImplicitFlag[bool] = False
    json_output: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Publish a temporary fixture cache, then pass only paths/config to ranks."""
        with TemporaryDirectory(prefix="dataset-rt-ddp-") as directory:
            root = Path(directory)
            runtime = DatasetRuntime(num_workers=1)
            outcomes = runtime.write_cache(
                [Source()], root, writer_config=WriterConfig(show_progress=not self.quiet)
            )
            outcome = outcomes[0]
            if not isinstance(outcome, CacheWriteSuccess):
                raise RuntimeError(outcome)
            mp.spawn(
                train_rank,
                args=(self, outcome.path, (root / "rendezvous").as_uri()),
                nprocs=self.ranks,
                join=True,
            )


def train_rank(rank: int, options: Example, cache: Path, rendezvous: str) -> None:
    """Initialize the group before creating the loader, with one native reader per PID."""
    torch.set_num_threads(1)
    # This single-host Mac example uses loopback explicitly; training on other
    # machines must choose the interface that connects its replicas.
    os.environ["GLOO_SOCKET_IFNAME"] = options.gloo_interface
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=options.ranks,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = DatasetRuntime(num_workers=1)
        dataset = runtime.cached_dataset(
            [cache], reader_config=ReaderConfig(seed=7, prefetch_size=2)
        )
        loader = dataset.to_torch_dataloader(
            shuffle=options.shuffle,
            seed=7,
            batch_size=4,
            num_workers=options.workers,
            multiprocessing_context=options.worker_context if options.workers else None,
            persistent_workers=options.workers > 0,
            sample_transform_fn=transform,
        )
        model = DistributedDataParallel(torch.nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        completed = 0
        # Finite validation partitions may yield different step counts. The caller
        # supplies join; the loader neither pads samples nor coordinates collectives.
        with model.join():
            for batch in islice(loader, options.steps):
                if not isinstance(batch, torch.Tensor):
                    raise TypeError("expected the transform's collated tensor")
                optimizer.zero_grad()
                model(batch).square().mean().backward()
                optimizer.step()
                completed += 1
                if rank == 0 and not options.quiet:
                    print(f"step {completed}/{options.steps}", file=sys.stderr)
        if options.json_output:
            print(
                json.dumps({"rank": rank, "steps": completed, "shuffle": options.shuffle}),
                flush=True,
            )
        elif not options.quiet:
            print(f"rank {rank}: completed {completed} optimizer steps", flush=True)
        del loader
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    CliApp.run(Example)
