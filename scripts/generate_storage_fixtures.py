"""Freeze caches and reader oracles from isolated producer revisions; never overwrite fixtures."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict

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

    from dataset_rt.records import Metadata

REPOSITORY = Path(__file__).resolve().parents[1]
PINNED_REVISIONS = {
    2: "94f2c3e1d22930f9dfd5402b098bc45e702e3b1e",
    3: "38864c828fb639ea58b6966c9c96a94d701320a3",
    4: "bfe84fef1da1aa75ddbdcfd3e82000681cc79b3f",
}
PAYLOADS = (
    b"",
    b"A short training sentence.",
    "Сумісність кешу: 한글, 日本語, café.".encode(),
    bytes(range(256)),
    b"\0" * 4096,
    b"independent-record-" * 73,
)


class StrictModel(BaseModel):
    """Reject accidental oracle/schema drift at JSON boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)


class LegacySettings(StrictModel):
    """Historical payload-compression settings preserve the legacy ratio field."""

    algo: Literal["none", "lz4"]
    ratio: float = 1.0


class PlainSettings(StrictModel):
    """Modern no-compression policy retains its explicit discriminator."""

    algo: Literal["none"] = "none"
    mode: Literal["none"] = "none"


class FastSettings(StrictModel):
    """Exercise the independent LZ4 fast wire variant."""

    algo: Literal["lz4"] = "lz4"
    mode: Literal["fast"] = "fast"
    acceleration: int = 1


class HcSettings(StrictModel):
    """Exercise the LZ4 HC wire variant independently of fast mode."""

    algo: Literal["lz4"] = "lz4"
    mode: Literal["hc"] = "hc"
    level: int = 9


class BrotliSettings(StrictModel):
    """Exercise the pure-Rust Brotli record framing."""

    algo: Literal["brotli"] = "brotli"
    mode: Literal["brotli"] = "brotli"
    level: int = 5


class ZstdSettings(StrictModel):
    """Exercise sized, checksummed Zstd frames."""

    algo: Literal["zstd"] = "zstd"
    mode: Literal["zstd"] = "zstd"
    level: int = 3


class CodecCase(StrictModel):
    """A named codec case supplies only settings accepted by its producer."""

    name: str
    settings: (
        LegacySettings | PlainSettings | FastSettings | HcSettings | BrotliSettings | ZstdSettings
    )


class ProducerSource(StrictModel):
    """Identify actual build inputs, including an explicitly marked uncommitted snapshot."""

    revision: str
    working_tree: bool
    source_sha256: str
    build_profile: Literal["release"] = "release"
    strip: Literal["none"] = "none"


class Producer(ProducerSource):
    """Runtime provenance is reported by the installed producer itself."""

    package_version: str
    python_version: str


class SampleMetadata(StrictModel):
    """The same primitive schema is used across all formats and codecs."""

    index: int
    negative: int
    flag: bool
    score: float
    label: str
    source: str

    def as_metadata(self) -> Metadata:
        """Supply precisely typed Python primitives to historical CacheInput contracts."""
        return {
            "index": self.index,
            "negative": self.negative,
            "flag": self.flag,
            "score": self.score,
            "label": self.label,
            "source": self.source,
        }


class Identity(StrictModel):
    """Retain producer-resolved physical identity, including positional v2 IDs."""

    cache_id: int
    sample_id: int


class ExpectedSample(Identity):
    """Record complete reader output once; epoch oracles reference only identities."""

    data_hex: str
    metadata: SampleMetadata


class ExpectedCache(StrictModel):
    """Keep input path order and codec settings as part of the compatibility contract."""

    name: str
    codec: CodecCase
    sample_count: int


class GroundTruth(StrictModel):
    """Immutable output of one producer's writer and reader, with file-integrity evidence."""

    format_version: int
    producer: Producer
    seed: int
    caches: tuple[ExpectedCache, ...]
    sequential: tuple[ExpectedSample, ...]
    weighted_epochs: tuple[tuple[Identity, ...], ...]
    file_sha256: dict[str, str]


class WorkerSpec(StrictModel):
    """The producer subprocess receives explicit provenance and an exclusive output directory."""

    format_version: int
    producer: ProducerSource
    output: Path
    codecs: tuple[CodecCase, ...]


class ManifestHeader(BaseModel):
    """Read the writer's version marker without duplicating historical manifest schemas."""

    format_version: int


class Settings(BaseSettings):
    """Append new format oracles; historical versions use pinned producer revisions."""

    model_config = SettingsConfigDict(cli_parse_args=False, cli_kebab_case=True)
    versions: tuple[int, ...] = (2, 3, 4, 5)
    revision: str | None = None
    output: Path = REPOSITORY / "tests/fixtures/storage-compatibility"
    quiet: CliImplicitFlag[bool] = False
    capture: Path | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> Settings:
        """Prevent ambiguous revisions and accidental v1 support promises."""
        if (
            not self.versions
            or len(set(self.versions)) != len(self.versions)
            or min(self.versions) < 2
        ):
            raise ValueError("versions must be distinct integers starting at 2")
        if self.revision is not None and len(self.versions) != 1:
            raise ValueError("--revision requires exactly one selected version")
        return self


@dataclass
class Source:
    """Tiny recognizable inputs exercise empty, Unicode, binary and multi-shard payloads."""

    name: str

    def __iter__(self) -> Iterator[CacheInput]:
        """Keep fixture inputs independent of whichever reader captures ground truth."""
        for index, data in enumerate(PAYLOADS):
            yield CacheInput(data, input_metadata(self.name, index).as_metadata())


def input_metadata(source: str, index: int) -> SampleMetadata:
    """Exercise bool, signed integer, exact float and empty/non-ASCII string values."""
    return SampleMetadata(
        index=index,
        negative=-(2**62) + index,
        flag=index % 2 == 0,
        score=index + 0.25,
        label="" if index == 0 else "échantillon 한글",
        source=source,
    )


def codec_cases(format_version: int) -> tuple[CodecCase, ...]:
    """Freeze every codec wire variant supported by the selected format family."""
    if format_version <= 3:
        return (
            CodecCase(name="none", settings=LegacySettings(algo="none")),
            CodecCase(name="lz4", settings=LegacySettings(algo="lz4")),
        )
    return (
        CodecCase(name="none", settings=PlainSettings()),
        CodecCase(name="lz4-fast", settings=FastSettings()),
        CodecCase(name="lz4-hc", settings=HcSettings()),
        CodecCase(name="brotli", settings=BrotliSettings()),
        CodecCase(name="zstd", settings=ZstdSettings()),
    )


def record_sample(sample: CachedSample) -> ExpectedSample:
    """Capture actual producer output and independently compare it to the known input."""
    metadata = SampleMetadata.model_validate(sample.metadata)
    if sample.data != PAYLOADS[sample.sample_id] or metadata != input_metadata(
        metadata.source, sample.sample_id
    ):
        raise AssertionError("producer failed to reproduce the known fixture input")
    return ExpectedSample(
        cache_id=sample.cache_id,
        sample_id=sample.sample_id,
        data_hex=sample.data.hex(),
        metadata=metadata,
    )


def file_checksums(directory: Path) -> dict[str, str]:
    """Hash only tiny fixture artifacts; the oracle itself is excluded to avoid self-reference."""
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "oracle.json"
    }


def capture(spec: WorkerSpec) -> None:
    """Run exclusively inside the producer environment, never against the candidate reader."""
    spec.output.mkdir(parents=True, exist_ok=False)
    runtime = DatasetRuntime(num_workers=1)
    caches = []
    paths = []
    for codec in spec.codecs:
        name = f"compat-v{spec.format_version}-{codec.name}-é"
        config = WriterConfig.model_validate(
            {
                "show_progress": False,
                "max_shard_bytes": 512,
                "shard_compression": codec.settings.model_dump(),
            }
        )
        result = runtime.write_cache(Source(name), spec.output, writer_config=config)[0]
        if not isinstance(result, CacheWriteSuccess):
            raise AssertionError(result)
        header = ManifestHeader.model_validate_json((result.path / "manifest.json").read_text())
        if header.format_version != spec.format_version:
            raise AssertionError(
                f"expected v{spec.format_version}, producer wrote v{header.format_version}"
            )
        caches.append(ExpectedCache(name=name, codec=codec, sample_count=len(PAYLOADS)))
        paths.append(result.path)
    seed = 17
    dataset = runtime.cached_dataset(
        paths, reader_config=ReaderConfig(seed=seed, shuffle=False, validate_cache=True)
    )
    sequential = tuple(record_sample(sample) for sample in dataset)
    shuffled = runtime.cached_dataset(
        paths, reader_config=ReaderConfig(seed=seed, shuffle=True, validate_cache=True)
    )
    weights = shuffled.get_metadata().with_columns(
        (pl.col("index") + 1).cast(pl.Float64).alias("weight")
    )
    weights.write_ipc(spec.output / "weights.arrow")
    shuffled.update_metadata(weights)
    shuffled.set_epoch_len(23)
    epochs = tuple(
        tuple(Identity(cache_id=s.cache_id, sample_id=s.sample_id) for s in shuffled)
        for _ in range(2)
    )
    producer = Producer(
        revision=spec.producer.revision,
        working_tree=spec.producer.working_tree,
        source_sha256=spec.producer.source_sha256,
        package_version=version("dataset-rt"),
        python_version=sys.version.split()[0],
    )
    oracle = GroundTruth(
        format_version=spec.format_version,
        producer=producer,
        seed=seed,
        caches=tuple(caches),
        sequential=sequential,
        weighted_epochs=epochs,
        file_sha256=file_checksums(spec.output),
    )
    (spec.output / "oracle.json").write_text(oracle.model_dump_json(indent=2) + "\n")


def source_checksum(directory: Path) -> str:
    """Identify the exported source tree without build outputs or environment dependencies."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def export_source(revision: str, directory: Path) -> tuple[str, bool]:
    """Export commits read-only; snapshot current build inputs without disturbing dirty work."""
    if revision != "working-tree":
        resolved = subprocess.check_output(
            ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=REPOSITORY, text=True
        ).strip()
        archive = directory.parent / "source.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", "--output", str(archive), resolved],
            cwd=REPOSITORY,
            check=True,
        )
        with tarfile.open(archive) as source:
            source.extractall(directory, filter="data")
        return resolved, False
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    files = [
        REPOSITORY / name
        for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md", "LICENSE")
    ]
    files.extend(
        path
        for root in (REPOSITORY / "src", REPOSITORY / "dataset_rt")
        for path in root.rglob("*")
        if path.is_file() and (path.suffix in {".rs", ".py", ".pyi"} or path.name == "py.typed")
    )
    for source in files:
        target = directory / source.relative_to(REPOSITORY)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return revision, True


def build_and_capture(format_version: int, revision: str, output: Path, quiet: bool) -> None:
    """Build one exported producer and execute the capture harness in its own installed wheel."""
    with tempfile.TemporaryDirectory(prefix=f"datasetrt-producer-v{format_version}-") as temporary:
        directory = Path(temporary)
        source = directory / "source"
        resolved, working_tree = export_source(revision, source)
        checksum = source_checksum(source)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["CARGO_TARGET_DIR"] = str(REPOSITORY / "target/compatibility-producers")
        # Match the project's LINKEDIT workaround without editing historical Cargo manifests.
        environment["CARGO_PROFILE_RELEASE_STRIP"] = "none"
        wheels = directory / "wheels"
        run_producer_command(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "--with",
                "maturin>=1.7,<2",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(source / "Cargo.toml"),
                "--out",
                str(wheels),
            ],
            cwd=source,
            environment=environment,
            quiet=quiet,
        )
        (wheel,) = wheels.glob("*.whl")
        virtualenv = directory / "environment"
        run_producer_command(
            ["uv", "venv", "--python", sys.executable, str(virtualenv)],
            cwd=directory,
            environment=environment,
            quiet=quiet,
        )
        executable = virtualenv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run_producer_command(
            [
                "uv",
                "pip",
                "install",
                "--only-binary=:all:",
                "--python",
                str(executable),
                str(wheel),
                "pydantic-settings",
                "pyarrow",
            ],
            cwd=directory,
            environment=environment,
            quiet=quiet,
        )
        # Producer identity is captured by the producer, not guessed from the checkout's manifest.
        spec = WorkerSpec(
            format_version=format_version,
            output=directory / "captured",
            codecs=codec_cases(format_version),
            producer=ProducerSource(
                revision=resolved, working_tree=working_tree, source_sha256=checksum
            ),
        )
        specification = directory / "capture.json"
        specification.write_text(spec.model_dump_json())
        if not quiet:
            print(f"Capture v{format_version}: {resolved} (snapshot={working_tree})", flush=True)
        run_producer_command(
            [str(executable), str(Path(__file__).resolve()), "--capture", str(specification)],
            cwd=directory,
            environment=environment,
            quiet=quiet,
        )
        publish_fixture(spec.output, output)


def run_producer_command(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    quiet: bool,
) -> None:
    """Suppress successful build chatter in quiet mode while retaining failure diagnostics."""
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    if result.returncode != 0 and result.stdout is not None:
        print(result.stdout, file=sys.stderr, end="")
    result.check_returncode()


def publish_fixture(captured: Path, output: Path) -> None:
    """Publish a complete oracle on the destination filesystem; preserve all existing fixtures."""
    if output.exists():
        raise ValueError(f"immutable fixtures already exist: {output}")
    with tempfile.TemporaryDirectory(prefix=".new-oracle-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        shutil.copytree(captured, staged)
        staged.rename(output)


def main() -> None:
    """Generate only missing version directories; ordinary tests never invoke this command."""
    settings = CliApp.run(Settings)
    if settings.capture is not None:
        spec = WorkerSpec.model_validate_json(settings.capture.read_text())
        capture(spec)
        return
    plans = [
        (number, settings.revision or PINNED_REVISIONS.get(number, "working-tree"))
        for number in settings.versions
    ]
    for number, _ in plans:
        if (settings.output / f"v{number}").exists():
            raise ValueError(
                f"immutable fixtures already exist: {settings.output / f'v{number}'}; choose a new output directory"
            )
    settings.output.mkdir(parents=True, exist_ok=True)
    for number, revision in plans:
        build_and_capture(
            number, revision, settings.output.resolve() / f"v{number}", settings.quiet
        )


if __name__ == "__main__":
    main()
