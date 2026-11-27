"""Run the suite against an installed wheel without a source package shadowing it."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Stage test inputs only; subprocess imports must resolve into the selected environment."""
    with tempfile.TemporaryDirectory(prefix="datasetrt-wheel-tests-") as temporary:
        directory = Path(temporary)
        for name in ("tests", "scripts"):
            shutil.copytree(
                PROJECT_ROOT / name,
                directory / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copy2(PROJECT_ROOT / "pytest.ini", directory / "pytest.ini")
        shutil.copy2(PROJECT_ROOT / "pyproject.toml", directory / "pyproject.toml")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import dataset_rt,sys; from pathlib import Path; "
                "p=Path(dataset_rt.__file__).resolve(); print('Installed package:',p,flush=True); "
                "assert p.is_relative_to(Path(sys.prefix).resolve()), p",
            ],
            cwd=directory,
            env=environment,
            check=True,
        )
        subprocess.run(
            # Documentation generation reads repository sources and is checked separately.
            [sys.executable, "-m", "pytest", "--ignore=tests/test_api_docs.py"],
            cwd=directory,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
