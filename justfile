set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

python := "3.11"
py := "uv run --python " + python + " --extra dev"

default:
    just --list

sync:
    uv sync --python {{python}} --extra dev

develop:
    {{py}} maturin develop

fmt:
    {{py}} ruff format dataset_rt tests scripts
    cargo fmt

fmt-check:
    {{py}} ruff format --check dataset_rt tests scripts
    cargo fmt --check

lint:
    {{py}} ruff check dataset_rt tests scripts

typecheck:
    {{py}} pyrefly check

test-rust:
    cargo test

clippy:
    cargo clippy --all-targets -- -D warnings

test-python: develop
    {{py}} pytest

test: test-rust test-python

check: fmt-check lint typecheck test-rust clippy test-python

smoke: develop
    {{py}} scripts/bench_smoke.py

release:
    {{py}} scripts/release_local.py

release-all: release

release-sdist:
    DATASETRT_SKIP_MACOS=true DATASETRT_SKIP_LINUX=true {{py}} scripts/release_local.py

release-macos:
    DATASETRT_SKIP_LINUX=true {{py}} scripts/release_local.py

release-linux:
    DATASETRT_SKIP_MACOS=true {{py}} scripts/release_local.py
