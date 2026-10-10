"""Run portable reader/training benchmarks with python -m dataset_rt.run_benchmark."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_rt.benchmarks.config import BenchmarkConfig


def launch(config: BenchmarkConfig) -> None:
    """Let torchrun supervise local ranks; kill only this launch's session on timeout."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--local-addr",
        "127.0.0.1",
        "--nproc-per-node",
        str(config.ranks),
        "--max-restarts",
        "0",
        "--module",
        "dataset_rt.run_benchmark",
        *sys.argv[1:],
    ]
    with subprocess.Popen(command, start_new_session=True) as process:
        try:
            status = process.wait(timeout=config.timeout_seconds)
            if status:
                raise SystemExit(status)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            raise


def main() -> None:
    """Parse CLI before launch; optional benchmark dependencies stay outside the base API."""
    try:
        from pydantic_settings import CliApp

        from dataset_rt.benchmarks.config import BenchmarkConfig
        from dataset_rt.benchmarks.distributed import run_rank
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing benchmark dependency {error.name!r}; install dataset-rt[benchmark]."
        ) from error
    config = CliApp.run(BenchmarkConfig)
    if "RANK" not in os.environ:
        launch(config)
        return
    run_rank(config)


if __name__ == "__main__":
    main()
